"""Matplotlib charts for ``metrics/back_testing.json`` (same library as tabular ROC/PR curves)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

_ACCENT = "#EE0000"
_NEUTRAL = "#161616"
_CUTOFF = "gray"


def _matplotlib():
    """Import matplotlib on demand (same pattern as ROC/PR cells in tabular notebooks)."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for back-testing charts. Install it with: pip install matplotlib"
        ) from exc

    return plt, mdates


def _metric_display_name(metric: str) -> str:
    """Return a short label for plot titles (abbreviates snake_case to initials)."""
    if "_" not in metric:
        return metric
    metric = metric.replace("_with_bias", "_bias")  # AutoGluon's MAEB/WAPEB aliases.
    return "".join(w[0] for w in metric.split("_") if w).upper()


def _normalize_metric(metric: str) -> str:
    return _metric_display_name((metric or "").strip().lstrip("-")).upper()


def pick_window_metric(
    metrics: dict[str, Any], eval_metric: str, *, allow_fallback: bool = True
) -> tuple[str, float] | None:
    """Return the actual metric name and finite value, preferring ``eval_metric``."""
    finite = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    wanted = _normalize_metric(eval_metric)
    for key, value in finite.items():
        if _normalize_metric(key) == wanted:
            return key, value
    return next(iter(finite.items()), None) if allow_fallback else None


def _resolve_item_id(item_id: Any, available_item_ids: list[Any]) -> Any | None:
    """Map a backtest ``item_id`` to the matching ID in ``available_item_ids``."""
    if item_id is None:
        return None
    if item_id in available_item_ids:
        return item_id
    item_text = str(item_id)
    for candidate in available_item_ids:
        if str(candidate) == item_text:
            return candidate
    return None


def backtest_highlight_item_ids(
    back_testing: dict[str, Any],
    available_item_ids: list[Any],
    *,
    max_items: int = 4,
) -> list[Any]:
    """Return best/worst performer IDs from backtest JSON that exist in the sample data."""
    if not available_item_ids or max_items <= 0:
        return []

    series_analysis = back_testing.get("series_analysis") or {}
    resolved: list[Any] = []
    for role in ("best_performer", "worst_performer"):
        performer = series_analysis.get(role)
        if not performer:
            continue
        item_id = _resolve_item_id(performer.get("item_id"), available_item_ids)
        if item_id is not None and item_id not in resolved:
            resolved.append(item_id)
        if len(resolved) >= max_items:
            break
    return resolved[:max_items]


def forecast_data_to_frame(forecast_data: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse ``forecast_data`` rows from ``back_testing.json``."""
    if not forecast_data:
        return pd.DataFrame(columns=["timestamp", "actual", "predicted", "lower_bound", "upper_bound"])
    frame = pd.DataFrame(forecast_data)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp")


def per_window_metrics_table(
    per_window_metrics: list[dict[str, Any]],
    eval_metric: str,
) -> pd.DataFrame:
    """Build a per-cutoff metrics table aligned with AutoGluon backtest tutorials."""
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(per_window_metrics):
        metrics = window.get("metrics") or {}
        metric = pick_window_metric(metrics, eval_metric, allow_fallback=False)
        rows.append(
            {
                "cutoff": window.get("cutoff"),
                "window_id": window.get("window_id", index),
                "test_start": window.get("test_start", ""),
                "test_end": window.get("test_end", ""),
                eval_metric: metric[1] if metric else None,
            }
        )
    return pd.DataFrame(rows)


def mean_window_metric(per_window_metrics: list[dict[str, Any]], eval_metric: str) -> float | None:
    """Return the mean ``eval_metric`` across validation windows (each window is all-series)."""
    table = per_window_metrics_table(per_window_metrics, eval_metric)
    if eval_metric not in table.columns:
        return None
    values = table[eval_metric].dropna()
    if values.empty:
        return None
    mean = float(values.mean())
    return mean if math.isfinite(mean) else None


def _present_frame(frame: pd.DataFrame) -> None:
    try:
        from IPython.display import display

        display(frame)
    except ImportError:
        print(frame.to_string(index=False))


def _show_overall_backtest_summary(
    per_window_metrics: list[dict[str, Any]],
    eval_metric: str,
    *,
    num_series_evaluated: int | None = None,
) -> None:
    """Print mean ``eval_metric`` across windows (AutoGluon-style cross-series aggregate)."""
    overall = mean_window_metric(per_window_metrics, eval_metric)
    if overall is None:
        return
    window_count = int(per_window_metrics_table(per_window_metrics, eval_metric)[eval_metric].count())
    window_label = "window" if window_count == 1 else "windows"
    summary = f"Overall {eval_metric} (mean across {window_count} validation {window_label}, all series): {overall:.6g}"
    if num_series_evaluated is not None:
        summary += f" | Series evaluated: {num_series_evaluated}"
    print(summary)


def _show_per_window_metrics(
    per_window_metrics: list[dict[str, Any]],
    eval_metric: str,
    *,
    num_series_evaluated: int | None = None,
) -> None:
    """Print cutoff scores and show a table (AutoGluon uses print + pivot, not bar charts)."""
    table = per_window_metrics_table(per_window_metrics, eval_metric)
    if eval_metric not in table.columns or table[eval_metric].notna().sum() == 0:
        print(f"No finite {eval_metric} values per validation window.")
        return

    for _, row in table.iterrows():
        value = row[eval_metric]
        if pd.notna(value):
            print(f"Cutoff {row['cutoff']}: {eval_metric} = {value}")

    _present_frame(table)
    _show_overall_backtest_summary(
        per_window_metrics,
        eval_metric,
        num_series_evaluated=num_series_evaluated,
    )


def _style_date_axis(ax: Any) -> None:
    """Rotate x labels to reduce overlap; keep matplotlib's default date formatting."""
    plt, _ = _matplotlib()
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")


def _resolve_cutoff_timestamp(cutoff_start: str | None, timestamps: pd.Series) -> pd.Timestamp | None:
    """Map a window ``test_start`` bound onto plotted timestamps.

    ``test_start`` is stored as YYYY-MM-DD for compact tables; for hourly (or sub-daily)
    data that resolves to midnight and can sit left of the first plotted point.
    """
    if timestamps.empty:
        return pd.to_datetime(cutoff_start, utc=True) if cutoff_start else None

    first_ts = pd.Timestamp(timestamps.min())
    if first_ts.tzinfo is not None:
        first_ts = first_ts.tz_convert("UTC").tz_localize(None)

    if not cutoff_start:
        return first_ts

    cutoff = pd.to_datetime(cutoff_start, utc=True)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)

    if cutoff.normalize() == cutoff and cutoff < first_ts:
        return first_ts
    return cutoff


def _target_column_name(data: pd.DataFrame) -> str:
    if data.columns.empty:
        raise ValueError("Cannot extract target column name from DataFrame with no columns")
    return str(data.columns[0])


def _point_forecast_column(predictions: pd.DataFrame) -> str:
    if "0.5" in predictions.columns:
        return "0.5"
    if "mean" in predictions.columns:
        return "mean"
    raise ValueError("predictions must include a 'mean' or '0.5' forecast column")


def plot_timeseries_forecasts(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    item_ids: list[Any] | None = None,
    quantile_levels: list[float] | None = None,
    max_history_length: int | None = None,
    target: str | None = None,
) -> None:
    """Plot history and forecast quantiles (AutoGluon ``predictor.plot``-compatible axes)."""
    plt, _ = _matplotlib()
    quantile_levels = quantile_levels or [0.1, 0.9]
    q_low, q_high = (str(level) for level in quantile_levels[:2])
    point_column = _point_forecast_column(predictions)

    if item_ids is None:
        if hasattr(data, "item_ids"):
            plot_ids = list(data.item_ids)[:4]
        elif isinstance(data.index, pd.MultiIndex):
            plot_ids = list(data.index.get_level_values(0).unique()[:4])
        else:
            plot_ids = [None]
    else:
        plot_ids = list(item_ids)

    target_name = target or _target_column_name(data)
    count = max(len(plot_ids), 1)
    figure, axes = plt.subplots(1, count, figsize=(max(8.0, 4.0 * count), 4.5), squeeze=False)

    for index, item_id in enumerate(plot_ids):
        axis = axes[0, index]
        if item_id is None:
            history = data
            preds = predictions
        else:
            history = data.loc[item_id]
            preds = predictions.loc[item_id]

        if max_history_length:
            history = history.iloc[-max_history_length:]

        axis.plot(history.index, history.iloc[:, 0], label="Observed", color=_NEUTRAL)
        point_forecast = preds[point_column]
        axis.plot(
            preds.index,
            point_forecast,
            marker="s",
            linestyle="--",
            color=_ACCENT,
            label="Forecast",
        )

        if q_low in preds.columns and q_high in preds.columns:
            axis.fill_between(
                preds.index,
                preds[q_low],
                preds[q_high],
                alpha=0.1,
                color=_ACCENT,
                label=f"P{float(q_low) * 100:.0f}-P{float(q_high) * 100:.0f} interval",
            )

        if not preds.index.empty:
            axis.axvline(
                preds.index[0],
                color=_CUTOFF,
                linestyle=":",
                alpha=0.8,
                label="Forecast start",
            )

        title = str(item_id) if item_id is not None else target_name
        axis.set(title=title, xlabel="Date", ylabel=target_name if index == 0 else "")
        axis.legend(loc="best", fontsize=8)
        axis.grid(linestyle="--", alpha=0.3)
        _style_date_axis(axis)

    figure.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.show()


def _interval_label(frame: pd.DataFrame) -> str:
    if {"lower_quantile", "upper_quantile"}.issubset(frame.columns):
        lower = frame["lower_quantile"].dropna()
        upper = frame["upper_quantile"].dropna()
        if not lower.empty and not upper.empty:
            lo = float(lower.iloc[0])
            hi = float(upper.iloc[0])
            return f"{lo * 100:.0f}%-{hi * 100:.0f}% interval"
    return "10%-90% interval"


def _draw_cutoff(ax: Any, cutoff_start: str | None, timestamps: pd.Series | None = None) -> None:
    if timestamps is not None:
        cutoff = _resolve_cutoff_timestamp(cutoff_start, timestamps)
    elif cutoff_start:
        cutoff = pd.to_datetime(cutoff_start)
    else:
        return
    if cutoff is None:
        return
    ax.axvline(cutoff, color=_CUTOFF, linestyle="--", alpha=0.7, label="Cutoff")


def _draw_forecast(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    title: str,
    target: str,
    cutoff_start: str | None = None,
) -> None:
    frame = forecast_data_to_frame(rows)
    if frame.empty:
        raise ValueError("forecast_data is empty")

    timestamps = frame["timestamp"]
    if "actual" in frame.columns and frame["actual"].notna().any():
        ax.plot(timestamps, frame["actual"], marker="o", color=_NEUTRAL, label=f"Actual ({target})")
    ax.plot(timestamps, frame["predicted"], marker="s", linestyle="--", color=_ACCENT, label="Predicted")

    if {"lower_bound", "upper_bound"}.issubset(frame.columns):
        lower, upper = frame["lower_bound"], frame["upper_bound"]
        if lower.notna().any() and upper.notna().any():
            ax.fill_between(timestamps, lower, upper, alpha=0.1, color=_ACCENT, label=_interval_label(frame))

    _draw_cutoff(ax, cutoff_start, timestamps)

    ax.set(title=title, xlabel="Date", ylabel=target)
    ax.legend(loc="best", fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    _style_date_axis(ax)


def render_back_testing_metrics(back_testing: dict[str, Any]) -> None:
    """Print per-window backtest scores and summary table (no plots)."""
    eval_metric = back_testing.get("eval_metric", "mean_absolute_scaled_error")
    per_window = back_testing.get("per_window_metrics") or []
    series_analysis = back_testing.get("series_analysis") or {}
    num_series = series_analysis.get("num_series_evaluated")
    _show_per_window_metrics(
        per_window,
        eval_metric,
        num_series_evaluated=num_series if isinstance(num_series, int) else None,
    )


def render_back_testing_forecast_charts(back_testing: dict[str, Any]) -> None:
    """Render best/worst holdout forecast matplotlib panels."""
    eval_metric = back_testing.get("eval_metric", "mean_absolute_scaled_error")
    target = back_testing.get("target", "target")
    per_window = back_testing.get("per_window_metrics") or []
    series_analysis = back_testing.get("series_analysis") or {}

    plotted_ids: set[Any] = set()
    plt, _ = _matplotlib()
    for heading, role in (("Best", "best_performer"), ("Worst", "worst_performer")):
        performer = series_analysis.get(role)
        if not performer:
            continue
        item_id = performer.get("item_id")
        if item_id in plotted_ids:
            continue
        plotted_ids.add(item_id)

        item_windows = performer.get("windows") or []
        if not item_windows:
            continue

        count = len(item_windows)
        figure, axes = plt.subplots(1, count, figsize=(max(8.0, 4.0 * count), 4.2), squeeze=False)
        window_starts = {
            window.get("window_id"): window.get("test_start")
            for window in per_window
            if window.get("window_id") is not None
        }
        for index, window in enumerate(item_windows):
            axis = axes[0, index]
            window_id = window.get("window_id", index)
            metric = pick_window_metric(window.get("metrics") or {}, series_analysis.get("ranking_metric", eval_metric))
            metric_text = f"{_metric_display_name(metric[0])}={metric[1]:.3g}" if metric else "No finite metric"
            try:
                _draw_forecast(
                    axis,
                    window.get("forecast_data") or [],
                    title=f"Window {window_id} ({metric_text})",
                    target=target,
                    cutoff_start=window_starts.get(window_id),
                )
            except ValueError as exc:
                print(exc)
            if index:
                axis.set_ylabel("")

        ranking_metric = series_analysis.get("ranking_metric")
        ranking_label = "ranking metric unavailable (legacy artifact)"
        if ranking_metric:
            ranking_label = f"ranked by {_metric_display_name(ranking_metric)}"
            if _normalize_metric(ranking_metric) != _normalize_metric(eval_metric):
                ranking_label += f" (fallback for {_metric_display_name(eval_metric)})"
        figure.suptitle(f"{heading} performer: {item_id} — {ranking_label}", fontsize=12)
        figure.tight_layout(rect=[0, 0.04, 1, 0.92])
        plt.show()


def render_back_testing_charts(back_testing: dict[str, Any]) -> None:
    """Render per-window metrics and best/worst holdout forecast panels."""
    render_back_testing_metrics(back_testing)
    render_back_testing_forecast_charts(back_testing)


def notebook_backtest_charts_source() -> str:
    """Return plotting module source copied into generated inference notebooks."""
    source = Path(__file__).read_text(encoding="utf-8")
    marker = "def notebook_backtest_charts_source"
    return source.split(marker, maxsplit=1)[0]
