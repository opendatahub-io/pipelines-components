"""Build metrics/back_testing.json for AutoGluon time series models."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_NUM_VAL_WINDOWS = 3
# Limit forecast points per window to control JSON artifact size.
# At 500 points: ~21 days (hourly data), ~1.4 years (daily data), ~8 hours (minute data).
# For longer horizons, downstream consumers (UI, notebooks) should aggregate or paginate.
# This prevents multi-MB JSON files that would slow down KFP artifact rendering.
MAX_FORECAST_POINTS_PER_WINDOW = 500
# Point-forecast metrics computed per series; used internally for best/worst selection.
_SERIES_POINT_METRICS = frozenset({"mean_absolute_percentage_error", "root_mean_squared_error", "mean_absolute_error"})
_DEFAULT_SERIES_RANKING_METRIC = "mean_absolute_percentage_error"

# Metrics where lower values indicate better performance (AutoGluon time series defaults).
# Includes both uppercase acronyms (AutoGluon evaluate() output) and snake_case names
# (component-normalized keys used in series_analysis metrics).
_LOWER_IS_BETTER = frozenset(
    {
        "MASE",
        "mean_absolute_scaled_error",
        "MAPE",
        "mean_absolute_percentage_error",
        "SMAPE",
        "symmetric_mean_absolute_percentage_error",
        "MSE",
        "mean_squared_error",
        "RMSE",
        "root_mean_squared_error",
        "MAE",
        "mean_absolute_error",
        "WQL",
        "weighted_quantile_loss",
        "SQL",
        "symmetric_quantile_loss",
        "RMSSE",
        "root_mean_squared_scaled_error",
        "WAPE",
        "weighted_absolute_percentage_error",
        # AutoGluon 1.6 losses. BIAS is a signed diagnostic and stays unflipped.
        "MAEB",
        "WAPEB",
        "MQL",
    }
)


def to_finite_float(value: Any) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _normalize_metric_name(metric: str) -> str:
    return (metric or "").strip().lstrip("-").upper()


def filter_finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep only finite scalar metric values (KFP-safe).

    Normalizes AutoGluon's "higher-is-better" convention by converting error metrics
    back to their natural positive form. AutoGluon negates error metrics like MAPE/RMSE/MAE
    in .evaluate() output, so this function multiplies them by -1 to restore standard signs.

    Used for ``back_testing.json`` only. ``metrics.json`` keeps raw AutoGluon signs for
    leaderboard compatibility.
    """
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        number = to_finite_float(value)
        if number is not None:
            # AutoGluon negates error metrics for "higher is better" convention.
            # Convert back to natural form: MAPE/RMSE/etc should be positive.
            metric_normalized = _normalize_metric_name(key)
            if metric_normalized in _LOWER_IS_BETTER:
                # This metric is naturally "lower is better", so AutoGluon negated it.
                # Multiply by -1 to restore positive error values.
                cleaned[key] = -number
            else:
                cleaned[key] = number
    return cleaned


def serialize_timestamp(value: Any) -> str:
    """Serialize timestamps to ISO-8601 UTC-style strings."""
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        text = value.isoformat()
        if text.endswith("Z") or "+" in text:
            return text.replace("+00:00", "Z")
        return f"{text}Z" if "T" in text else f"{text}T00:00:00Z"
    text = str(value)
    if text.endswith("+00:00"):
        return text.replace("+00:00", "Z")
    if text.endswith("Z"):
        return text
    if "T" in text:
        return f"{text}Z"
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return f"{text[:10]}T00:00:00Z"
    return text


def serialize_date(value: Any) -> str:
    """Serialize timestamps to YYYY-MM-DD for compact window bounds."""
    timestamp = serialize_timestamp(value)
    return timestamp[:10] if len(timestamp) >= 10 else timestamp


def _is_higher_better(eval_metric: str) -> bool:
    original = (eval_metric or "").strip()
    if original.startswith("-"):
        return True
    # Check raw name first (handles snake_case), then normalized uppercase (handles acronyms).
    return original not in _LOWER_IS_BETTER and _normalize_metric_name(original) not in _LOWER_IS_BETTER


def _mean_prediction_column(predictions: pd.DataFrame) -> str:
    """Select the point-forecast column from predictions.

    Prefers 'mean' if available. If not, falls back to the first numeric column (likely a quantile),
    which may introduce systematic bias in computed error metrics.
    """
    if "mean" in predictions.columns:
        return "mean"
    numeric_cols = [c for c in predictions.columns if to_finite_float(c) is not None]
    if numeric_cols:
        col = str(numeric_cols[0])
        logger.warning(
            "No 'mean' column found in predictions; using quantile column %r as point forecast. "
            "Computed mean_absolute_percentage_error/root_mean_squared_error/mean_absolute_error may be biased.",
            col,
        )
        return col
    return str(predictions.columns[0])


def _quantile_bounds(predictions: pd.DataFrame) -> tuple[str | None, str | None]:
    """Extract lower/upper quantile columns aligned with ``TimeSeriesPredictor.plot`` defaults.

    Prefers quantile levels closest to 0.1 and 0.9 (AutoGluon's typical P10/P90 band).
    """
    levels: list[tuple[float, str]] = []
    for col in predictions.columns:
        try:
            levels.append((float(col), str(col)))
        except (TypeError, ValueError):
            continue
    if not levels:
        return None, None

    def _closest(candidates: list[tuple[float, str]], target: float) -> str | None:
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item[0] - target))[1]

    lower = _closest(levels, 0.1)
    if lower is None:
        return None, None

    remaining = [(value, name) for value, name in levels if name != lower]
    upper = _closest(remaining, 0.9)
    return lower, upper


def _item_ids(tsdf: pd.DataFrame) -> list[Any]:
    if isinstance(tsdf.index, pd.MultiIndex):
        return list(tsdf.index.get_level_values(0).unique())
    return [None]


def _holdout_frame(tsdf: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    """Return the forecast horizon rows (last prediction_length steps per series)."""
    if prediction_length <= 0:
        raise ValueError("prediction_length must be a positive integer.")

    if not isinstance(tsdf.index, pd.MultiIndex):
        return tsdf.tail(prediction_length)

    item_ids = _item_ids(tsdf)
    parts = [tsdf.loc[item_id].tail(prediction_length) for item_id in item_ids]
    return pd.concat(parts, keys=item_ids, names=tsdf.index.names)


def _window_date_bounds(targets_window: pd.DataFrame, prediction_length: int) -> tuple[str, str]:
    holdout = _holdout_frame(targets_window, prediction_length)
    timestamps = holdout.index.get_level_values(-1) if isinstance(holdout.index, pd.MultiIndex) else holdout.index
    timestamps = pd.to_datetime(timestamps, errors="coerce").dropna()
    if timestamps.empty:
        return "", ""
    return serialize_date(timestamps.min()), serialize_date(timestamps.max())


def _forecast_data_for_item(
    predictions: pd.DataFrame,
    targets_window: pd.DataFrame,
    item_id: Any,
    target: str,
    prediction_length: int,
) -> list[dict[str, Any]]:
    if isinstance(predictions.index, pd.MultiIndex):
        if item_id not in predictions.index.get_level_values(0):
            return []
        pred_item = predictions.loc[item_id]
    else:
        pred_item = predictions

    holdout = _holdout_frame(targets_window, prediction_length)
    if isinstance(holdout.index, pd.MultiIndex):
        if item_id not in holdout.index.get_level_values(0):
            return []
        target_item = holdout.loc[item_id]
    else:
        target_item = holdout

    pred_col = _mean_prediction_column(pred_item)
    lower_col, upper_col = _quantile_bounds(pred_item)

    rows: list[dict[str, Any]] = []
    for ts in target_item.index[:MAX_FORECAST_POINTS_PER_WINDOW]:
        if ts not in pred_item.index:
            continue
        predicted = to_finite_float(pred_item.loc[ts, pred_col])
        if predicted is None:
            continue
        actual = None
        if ts in target_item.index and target in target_item.columns:
            actual = to_finite_float(target_item.loc[ts, target])
        row: dict[str, Any] = {
            "timestamp": serialize_timestamp(ts),
            "predicted": predicted,
        }
        if actual is not None:
            row["actual"] = actual
        if lower_col is not None:
            lower = to_finite_float(pred_item.loc[ts, lower_col])
            if lower is not None:
                row["lower_bound"] = lower
                row["lower_quantile"] = float(lower_col)
        if upper_col is not None:
            upper = to_finite_float(pred_item.loc[ts, upper_col])
            if upper is not None:
                row["upper_bound"] = upper
                row["upper_quantile"] = float(upper_col)
        rows.append(row)
    return rows


def _point_errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return {"mean_absolute_percentage_error": None, "root_mean_squared_error": None, "mean_absolute_error": None}
    actual_f = actual[mask]
    predicted_f = predicted[mask]
    mae = float(np.mean(np.abs(actual_f - predicted_f)))
    rmse = float(np.sqrt(np.mean((actual_f - predicted_f) ** 2)))
    denom = np.abs(actual_f)
    mape_mask = denom > 0
    if mape_mask.any():
        mape = float(np.mean(np.abs((actual_f[mape_mask] - predicted_f[mape_mask]) / actual_f[mape_mask]) * 100))
    else:
        mape = None
    return {"mean_absolute_percentage_error": mape, "root_mean_squared_error": rmse, "mean_absolute_error": mae}


def _compute_metrics_from_forecast_data(forecast_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute error metrics from pre-generated forecast data rows.

    Args:
        forecast_rows: List of forecast dicts with "actual" and "predicted" keys

    Returns:
        Dict of computed metrics (mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error) with None values filtered out
    """  # noqa: E501
    paired = [r for r in forecast_rows if "actual" in r]
    if not paired:
        return {}
    actual = np.array([r["actual"] for r in paired], dtype=float)
    predicted = np.array([r["predicted"] for r in paired], dtype=float)
    return {k: v for k, v in _point_errors(actual, predicted).items() if v is not None}


def _item_window_metrics(
    predictions: pd.DataFrame,
    targets_window: pd.DataFrame,
    item_id: Any,
    target: str,
    prediction_length: int,
) -> dict[str, float]:
    """Compute point-forecast error metrics for a single item/window.

    Only rows with both valid actual and predicted values are included in the computation.
    This prevents silent metric loss when some actual values are NaN/Inf.
    """
    rows = _forecast_data_for_item(predictions, targets_window, item_id, target, prediction_length)
    return _compute_metrics_from_forecast_data(rows)


def _average_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = {key for metrics in metric_dicts for key in metrics}
    averaged: dict[str, float] = {}
    for key in keys:
        values = [metrics[key] for metrics in metric_dicts if key in metrics]
        if values:
            averaged[key] = float(np.mean(values))
    return averaged


def _series_ranking_metric(eval_metric: str, series_averages: dict[Any, dict[str, float]]) -> str:
    """Select ranking metric with fallback when primary metric unavailable.

    Args:
        eval_metric: Desired evaluation metric from component config
        series_averages: Per-series averaged metrics (used to check availability)

    Returns:
        Metric name to use for ranking (mean_absolute_percentage_error, root_mean_squared_error, or mean_absolute_error)

    Falls back through: eval_metric → mean_absolute_percentage_error → root_mean_squared_error → mean_absolute_error
    if metrics can't be computed. This handles cases where mean_absolute_percentage_error fails due to zero denominators.
    """  # noqa: E501
    # Collect all available metrics across all series
    available_metrics: set[str] = set()
    for metrics in series_averages.values():
        available_metrics.update(metrics.keys())

    if eval_metric in _SERIES_POINT_METRICS and eval_metric in available_metrics:
        return eval_metric

    # Fallback chain: mean_absolute_percentage_error → root_mean_squared_error → mean_absolute_error
    for fallback in ["mean_absolute_percentage_error", "root_mean_squared_error", "mean_absolute_error"]:
        if fallback in available_metrics:
            return fallback

    # Last resort: return mean_absolute_percentage_error even if not available (will sort to infinity)
    return _DEFAULT_SERIES_RANKING_METRIC


def _select_best_worst(
    series_averages: dict[Any, dict[str, float]],
    ranking_metric: str,
) -> tuple[Any | None, Any | None]:
    if not series_averages:
        return None, None

    higher_is_better = _is_higher_better(ranking_metric)

    def sort_key(item: tuple[Any, dict[str, float]]) -> float:
        value = item[1].get(ranking_metric)
        if value is None:
            # Always sort missing values to the end (worst), regardless of metric direction
            return float("inf")
        return -value if higher_is_better else value

    ordered = sorted(series_averages.items(), key=sort_key)
    return ordered[0][0], ordered[-1][0]


def _build_series_analysis(
    predictions_windows: list[pd.DataFrame],
    targets_windows: list[pd.DataFrame],
    *,
    target: str,
    prediction_length: int,
    eval_metric: str,
) -> dict[str, Any]:
    """Build series analysis with cached metrics and forecast data.

    Pre-computes forecast data once, then derives metrics from it. This avoids
    double computation: forecast data generation is the expensive operation
    (DataFrame slicing, type conversion, bounds extraction), while metrics
    computation from the generated rows is cheap (numpy array operations).
    """
    item_ids: set[Any] = set()
    for targets_window in targets_windows:
        item_ids.update(_item_ids(targets_window))

    # Cache structure: {(item_id, window_id): {"metrics": {...}, "forecast_data": [...]}}
    cache: dict[tuple[Any, int], dict[str, Any]] = {}
    per_item_window_metrics: dict[Any, list[dict[str, float]]] = {item_id: [] for item_id in item_ids}

    # Single pass: generate forecast data once, compute metrics from it
    for window_id, (predictions, targets_window) in enumerate(zip(predictions_windows, targets_windows, strict=True)):
        for item_id in item_ids:
            # Generate forecast data once
            forecast_data = _forecast_data_for_item(predictions, targets_window, item_id, target, prediction_length)
            # Compute metrics from the forecast data (no redundant generation)
            metrics = _compute_metrics_from_forecast_data(forecast_data)
            if metrics:
                per_item_window_metrics[item_id].append(metrics)
                # Cache both for later use
                cache[(item_id, window_id)] = {
                    "metrics": metrics,
                    "forecast_data": forecast_data,
                }

    series_averages = {
        item_id: _average_metrics(metrics_list)
        for item_id, metrics_list in per_item_window_metrics.items()
        if metrics_list
    }
    ranking_metric = _series_ranking_metric(eval_metric, series_averages)
    best_id, worst_id = _select_best_worst(series_averages, ranking_metric)

    def _series_payload(item_id: Any | None) -> dict[str, Any] | None:
        """Build performer payload from cached data (no recomputation)."""
        if item_id not in series_averages:
            return None
        windows_payload = []
        for window_id in range(len(predictions_windows)):
            cached = cache.get((item_id, window_id))
            if cached:
                windows_payload.append(
                    {
                        "window_id": window_id,
                        "metrics": cached["metrics"],
                        "forecast_data": cached["forecast_data"],
                    }
                )
        if not windows_payload:
            return None
        return {
            "item_id": item_id.item() if hasattr(item_id, "item") else item_id,
            "avg_metrics": series_averages.get(item_id, {}),
            "windows": windows_payload,
        }

    return {
        "num_series_evaluated": len(series_averages),
        "best_performer": _series_payload(best_id),
        "worst_performer": _series_payload(worst_id),
    }


def build_back_testing_json(
    predictor: Any,
    *,
    model_name: str,
    model_name_full: str,
    train_data: Any,
    eval_metric: str,
    target: str,
    id_column: str,
    timestamp_column: str,
    prediction_length: int,
    num_val_windows: int = DEFAULT_NUM_VAL_WINDOWS,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Build back_testing.json aligned with the model insights ADR schema."""
    if num_val_windows <= 0:
        raise ValueError("num_val_windows must be a positive integer.")
    if not hasattr(predictor, "backtest_predictions") or not hasattr(predictor, "backtest_targets"):
        raise AttributeError("TimeSeriesPredictor backtest API is unavailable on this AutoGluon version.")

    backtest_kwargs: dict[str, Any] = {
        "data": train_data,
        "num_val_windows": num_val_windows,
        "model": model_name,
        "use_cache": False,
    }

    predictions_windows = predictor.backtest_predictions(**backtest_kwargs)
    targets_windows = predictor.backtest_targets(data=train_data, num_val_windows=num_val_windows)
    if len(predictions_windows) != len(targets_windows):
        raise ValueError(
            f"backtest predictions/targets window count mismatch: {len(predictions_windows)} vs {len(targets_windows)}"
        )

    metric_names = metrics or [eval_metric]
    per_window_metrics: list[dict[str, Any]] = []
    num_windows = len(predictions_windows)
    # Generate cutoff points for each validation window.
    # AutoGluon's cutoff parameter is a negative integer indicating where evaluation starts:
    # cutoff=-N means evaluate from the -N-th to the (-N + prediction_length)-th time step.
    # For num_val_windows=3 and prediction_length=2, this generates [-6, -4, -2],
    # evaluating: window 0: steps -6 to -4, window 1: steps -4 to -2, window 2: steps -2 to end.
    cutoffs = range(-num_windows * prediction_length, 0, prediction_length)
    for window_id, cutoff in enumerate(cutoffs):
        evaluate_kwargs: dict[str, Any] = {
            "data": train_data,
            "model": model_name,
            "metrics": metric_names,
            "cutoff": cutoff,
        }
        try:
            window_scores = filter_finite_metrics(predictor.evaluate(**evaluate_kwargs))
        except Exception as exc:
            logger.warning("Backtest evaluate failed for window %s (cutoff=%s): %s", window_id, cutoff, exc)
            window_scores = {}
        test_start, test_end = "", ""
        if window_id < len(targets_windows):
            test_start, test_end = _window_date_bounds(targets_windows[window_id], prediction_length)
        per_window_metrics.append(
            {
                "window_id": window_id,
                "cutoff": cutoff,
                "test_start": test_start,
                "test_end": test_end,
                "metrics": window_scores,
            }
        )

    series_analysis = _build_series_analysis(
        predictions_windows,
        targets_windows,
        target=target,
        prediction_length=prediction_length,
        eval_metric=eval_metric,
    )

    return {
        "schema_version": 1,
        "model_name": model_name_full,
        "prediction_length": prediction_length,
        "num_val_windows": len(predictions_windows),
        "eval_metric": eval_metric,
        "target": target,
        "id_column": id_column,
        "timestamp_column": timestamp_column,
        "per_window_metrics": per_window_metrics,
        "series_analysis": series_analysis,
    }
