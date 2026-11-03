"""Tests for time series inference notebook helpers."""

from __future__ import annotations

import ast
from unittest import mock

import pandas as pd
import pytest

from ..timeseries_notebook_utils import (
    _coerce_sample_timestamps,
    build_predict_sample_artifact,
    fill_known_covariates_on_future_frame,
    notebook_timeseries_sample_helpers_source,
)


def _future_frame_flat(item_ids: list, timestamps: list[str]) -> pd.DataFrame:
    """Flat make_future_data_frame-style output (AutoGluon >= 1.3)."""
    rows = [(item_id, ts) for item_id in item_ids for ts in timestamps]
    return pd.DataFrame(rows, columns=["item_id", "timestamp"])


def _future_frame_multiindex(item_ids: list, timestamps: list[str]) -> pd.DataFrame:
    """Legacy MultiIndex future frame."""
    index = pd.MultiIndex.from_tuples(
        [(item_id, pd.Timestamp(ts)) for item_id in item_ids for ts in timestamps],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame(index=index)


def _ts_frame(
    item_ids: list,
    timestamps: list[str],
    covariates: dict[str, list[float]],
) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(item_id, pd.Timestamp(ts)) for item_id, ts in zip(item_ids, timestamps, strict=False)],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame(covariates, index=index)


class TestFillKnownCovariatesOnFutureFrame:
    """Tests for known-covariate filling on the forecast horizon."""

    def test_uses_exact_sample_values_for_matching_timestamps(self):
        """Covariates match historical rows when the forecast timestamp is in ts_df."""
        ts_df = _ts_frame(
            [1, 1, 1],
            ["2025-01-01", "2025-01-02", "2025-01-03"],
            {"promo": [0.0, 1.0, 0.5]},
        )
        future_cov = _future_frame_flat([1], ["2025-01-03", "2025-01-04"])

        result = fill_known_covariates_on_future_frame(future_cov, ts_df, ["promo"])

        assert result["promo"].tolist() == [0.5, 0.5]

    def test_fills_future_steps_with_last_known_value_per_item(self):
        """Future horizon steps use the last known covariate value per item."""
        ts_df = _ts_frame([1, 1], ["2025-01-01", "2025-01-02"], {"promo": [0.0, 1.0]})
        future_cov = _future_frame_flat([1], ["2025-01-03", "2025-01-04"])

        result = fill_known_covariates_on_future_frame(future_cov, ts_df, ["promo"])

        assert result["promo"].tolist() == [1.0, 1.0]

    def test_multiple_items_get_item_specific_last_values(self):
        """Each item receives its own last-known covariate value."""
        ts_df = _ts_frame(
            [1, 1, 2, 2],
            ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
            {"promo": [0.0, 1.0, 10.0, 20.0]},
        )
        future_cov = _future_frame_flat([1, 2], ["2025-01-03"])

        result = fill_known_covariates_on_future_frame(future_cov, ts_df, ["promo"])

        promo_by_item = result.set_index("item_id")["promo"]
        assert promo_by_item.loc[1] == 1.0
        assert promo_by_item.loc[2] == 20.0

    def test_string_item_ids(self):
        """String item ids such as 'strawberry' work on flat future frames."""
        ts_df = _ts_frame(
            ["strawberry", "strawberry", "strawberry"],
            ["2025-01-01", "2025-01-02", "2025-01-03"],
            {"price_max": [6.7, 6.5, 6.3]},
        )
        future_cov = _future_frame_flat(["strawberry"], ["2025-01-04", "2025-01-05"])

        result = fill_known_covariates_on_future_frame(future_cov, ts_df, ["price_max"])

        assert list(result.columns)[:3] == ["item_id", "timestamp", "price_max"]
        assert result["price_max"].tolist() == [6.3, 6.3]

    def test_flat_future_frame_with_integer_row_index(self):
        """Regression: flat future frame RangeIndex must not be used as item ids."""
        ts_df = _ts_frame(
            ["strawberry", "strawberry", "strawberry"],
            ["2025-01-01", "2025-01-02", "2025-01-03"],
            {"price_max": [6.7, 6.5, 6.3]},
        )
        future_cov = pd.DataFrame(
            {
                "item_id": ["strawberry", "strawberry"],
                "timestamp": pd.to_datetime(["2025-01-04", "2025-01-05"]),
            }
        )

        result = fill_known_covariates_on_future_frame(future_cov, ts_df, ["price_max"])

        assert result["price_max"].tolist() == [6.3, 6.3]

    def test_multiindex_future_frame_still_supported(self):
        """Older MultiIndex future templates remain supported."""
        ts_df = _ts_frame([1, 1], ["2025-01-01", "2025-01-02"], {"promo": [0.0, 1.0]})
        future_cov = _future_frame_multiindex([1], ["2025-01-03"])

        result = fill_known_covariates_on_future_frame(future_cov, ts_df, ["promo"])

        assert result["promo"].iloc[0] == 1.0

    def test_raises_when_covariates_missing_from_ts_df(self):
        """Missing covariate columns on ts_df fail fast with a clear error."""
        ts_df = _ts_frame([1], ["2025-01-01"], {"other_col": [1.0]})
        future_cov = _future_frame_flat([1], ["2025-01-02"])

        try:
            fill_known_covariates_on_future_frame(future_cov, ts_df, ["promo"])
        except ValueError as exc:
            assert "missing from ts_df" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing covariate column")


class TestCoerceSampleTimestamps:
    """Tests for epoch-millisecond timestamp parsing in notebook samples."""

    def test_parses_positive_epoch_milliseconds(self):
        """Post-1970 epoch ms values are parsed as UTC timestamps."""
        frame = pd.DataFrame({"timestamp": [1704067200000, 1704070800000]})
        result = _coerce_sample_timestamps(frame, "timestamp")
        assert result["timestamp"].tolist() == [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 01:00:00"),
        ]

    def test_parses_negative_epoch_milliseconds(self):
        """Pre-1970 epoch ms values (fractional-year hourly data) are parsed correctly."""
        frame = pd.DataFrame({"timestamp": [-6939997200000, -6939982800000]})
        result = _coerce_sample_timestamps(frame, "timestamp")
        assert result["timestamp"].tolist() == [
            pd.Timestamp("1750-01-29 23:00:00"),
            pd.Timestamp("1750-01-30 03:00:00"),
        ]

    def test_leaves_iso_strings_unchanged(self):
        """ISO timestamp strings are parsed without treating them as epoch integers."""
        frame = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"]})
        result = _coerce_sample_timestamps(frame, "timestamp")
        assert result["timestamp"].tolist() == [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 01:00:00"),
        ]


class TestNotebookTimeseriesSampleHelpersSource:
    """Tests for helper source embedded in generated notebooks."""

    def test_embedded_source_includes_runtime_helpers(self):
        """Notebook helper bundle exposes coercion and covariate fill helpers."""
        source = notebook_timeseries_sample_helpers_source()
        assert "def _coerce_sample_timestamps" in source
        assert "def fill_known_covariates_on_future_frame" in source
        assert "def build_predict_sample_artifact" not in source


class TestBuildPredictSampleArtifact:
    """Tests for notebook-embedded predict sample payload."""

    @pytest.mark.parametrize("covariates", [None, [], ["promo"]])
    @pytest.mark.parametrize("timestamp_column", ["date", "timestamp"])
    def test_empty_sample_preserves_metadata_without_forecasting(self, covariates, timestamp_column, monkeypatch):
        """The training component's default empty preview must not fail a refit."""
        import sys
        import types

        ts_df_cls = mock.Mock()
        mock_mod = types.ModuleType("autogluon.timeseries")
        mock_mod.TimeSeriesDataFrame = ts_df_cls
        monkeypatch.setitem(sys.modules, "autogluon.timeseries", mock_mod)
        predictor = mock.Mock()

        payload = build_predict_sample_artifact(predictor, [], "product_id", timestamp_column, covariates)

        assert payload == {
            "id_column": "product_id",
            "timestamp_column": timestamp_column,
            "known_covariates_names": covariates or [],
            "history": [],
            "known_covariates": None,
        }
        ts_df_cls.from_data_frame.assert_not_called()
        predictor.make_future_data_frame.assert_not_called()

    def test_builds_embeddable_payload(self, monkeypatch):
        """Payload contains history and pre-filled known covariates."""
        import sys
        import types

        def _from_data_frame(score_df, id_column, timestamp_column):
            covariate_cols = [c for c in score_df.columns if c not in (id_column, timestamp_column)]
            return _ts_frame(
                list(score_df[id_column]),
                [str(ts)[:10] for ts in score_df[timestamp_column]],
                {col: list(score_df[col]) for col in covariate_cols},
            )

        mock_mod = types.ModuleType("autogluon.timeseries")
        mock_mod.TimeSeriesDataFrame = types.SimpleNamespace(from_data_frame=_from_data_frame)
        monkeypatch.setitem(sys.modules, "autogluon.timeseries", mock_mod)

        class _Predictor:
            def make_future_data_frame(self, _ts_df):
                return _future_frame_flat(["strawberry"], ["2025-01-03"])

        payload = build_predict_sample_artifact(
            _Predictor(),
            [{"Fruit": "strawberry", "Date": "2025-01-01", "Price_max": 6.7}],
            "Fruit",
            "Date",
            ["Price_max"],
        )

        assert payload["id_column"] == "Fruit"
        assert payload["known_covariates"] is not None
        assert ast.literal_eval(str(payload)) == payload
