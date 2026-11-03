"""Helpers for AutoGluon time series inference notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _coerce_sample_timestamps(score_df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Parse sample-row timestamps written as epoch milliseconds by ``DataFrame.to_json``."""
    series = score_df[timestamp_column]
    if not pd.api.types.is_numeric_dtype(series):
        out = score_df.copy()
        out[timestamp_column] = pd.to_datetime(series, errors="coerce")
        return out

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        out = score_df.copy()
        out[timestamp_column] = pd.to_datetime(series, errors="coerce")
        return out

    if numeric.abs().max() > 1e11:
        out = score_df.copy()
        out[timestamp_column] = pd.to_datetime(numeric, unit="ms")
        return out

    out = score_df.copy()
    out[timestamp_column] = pd.to_datetime(series, errors="coerce")
    return out


def _is_datetime_series(series: pd.Series) -> bool:
    """Return whether a series holds datetime values (works with test pandas mocks)."""
    dtype = getattr(series, "dtype", None)
    if dtype is not None and str(dtype).startswith("datetime64"):
        return True
    api = getattr(pd, "api", None)
    if api is not None:
        return api.types.is_datetime64_any_dtype(series)
    return False


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe row dicts (ISO timestamps)."""
    out = df.copy()
    for col in out.columns:
        if _is_datetime_series(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
    return out.to_dict(orient="records")


def _future_horizon_frame(future_cov: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``make_future_data_frame`` output to flat item_id + timestamp columns."""
    if {"item_id", "timestamp"}.issubset(future_cov.columns):
        out = future_cov.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        return out

    if isinstance(future_cov.index, pd.MultiIndex):
        out = future_cov.reset_index()
        id_name = future_cov.index.names[0] or "item_id"
        ts_name = future_cov.index.names[1] or "timestamp"
        rename = {}
        if out.columns[0] != id_name:
            rename[out.columns[0]] = id_name
        if len(out.columns) > 1 and out.columns[1] != ts_name:
            rename[out.columns[1]] = ts_name
        if rename:
            out = out.rename(columns=rename)
        out = out.rename(columns={id_name: "item_id", ts_name: "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        return out

    raise ValueError(
        "future_cov must come from make_future_data_frame (item_id and timestamp columns) "
        "or use a MultiIndex (item_id, timestamp)."
    )


def _historical_covariates_long(
    ts_df: pd.DataFrame,
    known_covariates_names: list[str],
) -> pd.DataFrame:
    """Expand ts_df covariates to flat item_id + timestamp columns."""
    if not isinstance(ts_df.index, pd.MultiIndex):
        raise ValueError("ts_df must use a MultiIndex (item_id, timestamp).")

    hist = ts_df[known_covariates_names].reset_index()
    id_name = ts_df.index.names[0] or "item_id"
    ts_name = ts_df.index.names[1] or "timestamp"
    hist = hist.rename(columns={hist.columns[0]: id_name, hist.columns[1]: ts_name})
    hist = hist.rename(columns={id_name: "item_id", ts_name: "timestamp"})
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    return hist


def fill_known_covariates_on_future_frame(
    future_cov: pd.DataFrame,
    ts_df: pd.DataFrame,
    known_covariates_names: list[str],
) -> pd.DataFrame:
    """Fill known covariates on the flat forecast frame returned by ``make_future_data_frame``."""
    missing = [col for col in known_covariates_names if col not in ts_df.columns]
    if missing:
        raise ValueError(f"Known covariates missing from ts_df: {missing}")

    future_df = _future_horizon_frame(future_cov)[["item_id", "timestamp"]].copy()
    hist = _historical_covariates_long(ts_df, known_covariates_names)
    last_by_item = hist.groupby("item_id", sort=False)[known_covariates_names].last()

    for col in known_covariates_names:
        merged = future_df.merge(
            hist[["item_id", "timestamp", col]],
            on=["item_id", "timestamp"],
            how="left",
        )
        future_df[col] = merged[col].fillna(future_df["item_id"].map(last_by_item[col]))

    return future_df


def build_predict_sample_artifact(
    predictor: Any,
    sample_data: list[dict[str, Any]],
    id_column: str,
    timestamp_column: str,
    known_covariates_names: list[str] | None,
) -> dict[str, Any]:
    """Build predict demo payload embedded in the generated notebook.

    Includes historical sample rows and, when required, known covariates for the
    forecast horizon so the notebook only loads data and calls ``predict()``.
    An empty preview returns metadata with no history or forecast covariates.
    """
    covariate_names = known_covariates_names or []
    payload = {
        "id_column": id_column,
        "timestamp_column": timestamp_column,
        "known_covariates_names": covariate_names,
        "history": [],
        "known_covariates": None,
    }
    # sample_rows defaults to "[]" in the training component. There is no
    # timestamp column or forecast horizon to build when that preview is absent.
    if not sample_data:
        return payload

    from autogluon.timeseries import TimeSeriesDataFrame

    score_df = _coerce_sample_timestamps(pd.DataFrame(sample_data), timestamp_column)
    ts_df = TimeSeriesDataFrame.from_data_frame(
        score_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    if covariate_names:
        missing = [col for col in covariate_names if col not in score_df.columns]
        if not missing:
            future_cov = predictor.make_future_data_frame(ts_df)
            known_df = fill_known_covariates_on_future_frame(future_cov, ts_df, covariate_names)
            payload["known_covariates"] = _json_records(known_df)

    payload["history"] = _json_records(score_df)
    return payload


def notebook_timeseries_sample_helpers_source() -> str:
    """Return sample/covariate helpers copied into generated inference notebooks."""
    source = Path(__file__).read_text(encoding="utf-8")
    end = source.index("\n\ndef build_predict_sample_artifact")
    return source[:end]
