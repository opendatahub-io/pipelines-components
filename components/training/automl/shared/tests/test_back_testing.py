"""Tests for time series back_testing.json builder."""

from __future__ import annotations

import math
from unittest import mock

import pandas as pd
import pytest

from ..back_testing import (
    _build_series_analysis,
    _forecast_data_for_item,
    _holdout_frame,
    _item_window_metrics,
    _mean_prediction_column,
    _series_ranking_metric,
    build_back_testing_json,
    filter_finite_metrics,
    serialize_date,
    serialize_timestamp,
)


def _make_panel(item_ids: list[str], timestamps: list[str], target_values: list[float]) -> pd.DataFrame:
    rows = []
    for item_id in item_ids:
        for ts, value in zip(timestamps, target_values, strict=True):
            rows.append((item_id, pd.Timestamp(ts), value))
    index = pd.MultiIndex.from_tuples(
        [(item_id, ts) for item_id, ts, _ in rows],
        names=["item_id", "timestamp"],
    )
    return pd.DataFrame({"target": [value for _, _, value in rows]}, index=index)


class TestSerialization:
    """Tests for serialization helpers."""

    def test_filter_finite_metrics_drops_nan(self):
        """Non-finite metric values are omitted."""
        # AutoGluon negates error metrics, so MASE=-0.5 becomes 0.5
        assert filter_finite_metrics({"MASE": -0.5, "MAPE": float("nan"), "RMSE": float("inf")}) == {"MASE": 0.5}

    def test_filter_finite_metrics_normalizes_autogluon_signs(self):
        """Error metrics are converted from AutoGluon's negative 'higher-is-better' to positive natural form."""
        # AutoGluon .evaluate() returns negated error metrics (higher-is-better convention)
        autogluon_output = {"MASE": -0.42, "MAPE": -5.0, "RMSE": -10.5, "MAE": -3.2}
        # After normalization, error metrics should be positive
        normalized = filter_finite_metrics(autogluon_output)
        assert normalized == {"MASE": 0.42, "MAPE": 5.0, "RMSE": 10.5, "MAE": 3.2}
        # All values positive (natural error form)
        assert all(v > 0 for v in normalized.values())

    def test_filter_finite_metrics_normalizes_autogluon_1_6_losses(self):
        """MAEB, WAPEB, and MQL are restored to positive losses; signed BIAS is unchanged."""
        assert filter_finite_metrics({"MAEB": -1.5, "WAPEB": -0.2, "MQL": -0.8, "BIAS": 0.4}) == {
            "MAEB": 1.5,
            "WAPEB": 0.2,
            "MQL": 0.8,
            "BIAS": 0.4,
        }
        assert filter_finite_metrics({"BIAS": -0.3}) == {"BIAS": -0.3}

    def test_back_testing_normalizes_autogluon_signs(self):
        """back_testing.json converts AutoGluon negated error metrics to natural positive form."""
        raw_autogluon = {"MASE": -0.42, "MAPE": -5.0, "RMSE": -10.5}
        assert filter_finite_metrics(raw_autogluon) == {
            "MASE": 0.42,
            "MAPE": 5.0,
            "RMSE": 10.5,
        }

    def test_metrics_json_and_backtesting_use_different_sign_conventions(self):
        """metrics.json keeps raw AutoGluon signs; back_testing.json normalizes to natural form."""
        raw_autogluon = {"MASE": -0.42, "MAPE": -5.0, "RMSE": -10.5}
        metrics_json_values = {
            k: v for k, v in raw_autogluon.items() if isinstance(v, (int, float)) and math.isfinite(v)
        }

        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["A"], timestamps, [100.0, 110.0, 120.0, 130.0])
        window_targets = _holdout_frame(train_data, prediction_length=2)
        predictions = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["A"].index)

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        predictor.evaluate.return_value = raw_autogluon

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
            metrics=list(raw_autogluon.keys()),
        )

        backtesting_values = payload["per_window_metrics"][0]["metrics"]
        assert metrics_json_values == raw_autogluon
        assert backtesting_values == filter_finite_metrics(raw_autogluon)
        assert metrics_json_values != backtesting_values

    def test_serialize_timestamp_utc(self):
        """Timestamps serialize to ISO strings with UTC suffix."""
        assert serialize_timestamp(pd.Timestamp("2025-12-08T00:00:00Z")) == "2025-12-08T00:00:00Z"
        assert serialize_timestamp(pd.Timestamp("2025-12-08")) == "2025-12-08T00:00:00Z"

    def test_serialize_date(self):
        """Window bounds serialize to YYYY-MM-DD."""
        assert serialize_date(pd.Timestamp("2025-12-08T15:30:00Z")) == "2025-12-08"


class TestHoldoutHelpers:
    """Tests for holdout and forecast helpers."""

    def test_holdout_frame_takes_last_prediction_length_per_item(self):
        """Holdout uses the last prediction_length rows per series."""
        ts = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        panel = _make_panel(["A"], ts, [1.0, 2.0, 3.0, 4.0])
        holdout = _holdout_frame(panel, prediction_length=2)
        assert len(holdout) == 2
        assert holdout["target"].tolist() == [3.0, 4.0]

    def test_holdout_frame_single_series_no_multiindex(self):
        """Holdout works for single time-series (non-MultiIndex DataFrame)."""
        timestamps = pd.date_range("2025-01-01", periods=5, freq="D")
        single_series = pd.DataFrame(
            {"target": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=timestamps,
        )
        holdout = _holdout_frame(single_series, prediction_length=2)
        assert len(holdout) == 2
        assert holdout["target"].tolist() == [40.0, 50.0]
        assert not isinstance(holdout.index, pd.MultiIndex)

    def test_holdout_frame_shorter_than_prediction_length(self):
        """Holdout returns all available rows when the series is shorter than prediction_length."""
        ts = ["2025-01-01", "2025-01-02"]
        panel = _make_panel(["A"], ts, [1.0, 2.0])
        holdout = _holdout_frame(panel, prediction_length=5)
        assert len(holdout) == 2
        assert holdout["target"].tolist() == [1.0, 2.0]

    def test_item_window_metrics_computes_mape(self):
        """Per-item window metrics include MAPE from point forecasts."""
        timestamps = ["2025-01-03", "2025-01-04"]
        targets = _make_panel(["A"], timestamps, [100.0, 200.0])
        predictions = pd.DataFrame(
            {"mean": [110.0, 180.0]},
            index=targets.index,
        )
        metrics = _item_window_metrics(predictions, targets, "A", "target", prediction_length=2)
        assert "mean_absolute_percentage_error" in metrics
        assert metrics["mean_absolute_percentage_error"] == pytest.approx(10.0)

    def test_forecast_data_includes_actual_and_predicted(self):
        """Forecast rows include actual, predicted, and optional quantile bounds."""
        timestamps = ["2025-01-03"]
        targets = _make_panel(["A"], timestamps, [100.0])
        predictions = pd.DataFrame({"mean": [105.0], "0.1": [95.0], "0.9": [115.0]}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=1)
        assert rows[0]["actual"] == 100.0
        assert rows[0]["predicted"] == 105.0
        assert rows[0]["lower_bound"] == 95.0
        assert rows[0]["upper_bound"] == 115.0
        assert rows[0]["lower_quantile"] == 0.1
        assert rows[0]["upper_quantile"] == 0.9

    def test_quantile_bounds_prefer_p10_p90(self):
        """P10/P90 quantiles are preferred over other levels when all are present."""
        from ..back_testing import _quantile_bounds

        predictions = pd.DataFrame(
            {"mean": [1.0], "0.1": [0.5], "0.2": [0.6], "0.8": [1.4], "0.9": [1.5]},
            index=[0],
        )
        lower, upper = _quantile_bounds(predictions)
        assert lower == "0.1"
        assert upper == "0.9"

    def test_quantile_bounds_single_quantile_omits_upper(self):
        """A lone quantile column is used only as the lower bound (no collapsed band)."""
        from ..back_testing import _quantile_bounds

        predictions = pd.DataFrame({"mean": [1.0], "0.5": [1.0]}, index=[0])
        lower, upper = _quantile_bounds(predictions)
        assert lower == "0.5"
        assert upper is None

    def test_quantile_bounds_picks_distinct_lower_and_upper(self):
        """Lower and upper bounds are never the same quantile column."""
        from ..back_testing import _quantile_bounds

        predictions = pd.DataFrame({"0.1": [0.9], "0.5": [1.0], "0.9": [1.1]}, index=[0])
        lower, upper = _quantile_bounds(predictions)
        assert lower == "0.1"
        assert upper == "0.9"
        assert lower != upper

    def test_forecast_data_limits_to_holdout_horizon(self):
        """Forecast rows cover holdout steps only, not the full prediction window history."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]
        targets = _make_panel(["A"], timestamps, [1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = pd.DataFrame({"mean": [1.1, 2.1, 3.1, 4.1, 5.1]}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=2)
        assert len(rows) == 2
        assert rows[0]["timestamp"].startswith("2025-01-04")
        assert rows[1]["timestamp"].startswith("2025-01-05")

    def test_forecast_data_omits_bounds_without_quantiles(self):
        """Forecast rows omit bounds when prediction output has no quantile columns."""
        timestamps = ["2025-01-03"]
        targets = _make_panel(["A"], timestamps, [100.0])
        predictions = pd.DataFrame({"mean": [105.0]}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=1)
        assert "lower_bound" not in rows[0]
        assert "upper_bound" not in rows[0]

    def test_forecast_data_skips_timestamp_missing_from_predictions(self):
        """Forecast rows omit holdout timestamps with no matching prediction."""
        timestamps = ["2025-01-03", "2025-01-04"]
        targets = _make_panel(["A"], timestamps, [100.0, 200.0])
        predictions = pd.DataFrame({"mean": [105.0]}, index=targets.index[:1])
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=2)
        assert len(rows) == 1
        assert rows[0]["actual"] == 100.0
        assert rows[0]["predicted"] == 105.0

    def test_mean_prediction_column_warns_without_mean(self):
        """A warning is logged when falling back from missing mean to a quantile column."""
        from .. import back_testing

        predictions = pd.DataFrame({0.1: [1.0], 0.9: [2.0]})
        with mock.patch.object(back_testing.logger, "warning") as mock_warning:
            col = _mean_prediction_column(predictions)
        assert col == "0.1"
        mock_warning.assert_called_once()
        assert "quantile column" in mock_warning.call_args[0][0].lower()

    def test_forecast_data_capped_at_max_points(self):
        """Forecast rows never exceed MAX_FORECAST_POINTS_PER_WINDOW."""
        from ..back_testing import MAX_FORECAST_POINTS_PER_WINDOW

        n = MAX_FORECAST_POINTS_PER_WINDOW + 10
        timestamps = [f"2025-01-{i:02d}" for i in range(1, min(n + 1, 32))]  # Cap at 31 for valid dates
        if len(timestamps) < n:
            # For larger n, use hours instead
            timestamps = pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S").tolist()

        targets = _make_panel(["A"], timestamps[:n], list(range(n)))
        predictions = pd.DataFrame({"mean": list(range(n))}, index=targets.index)
        rows = _forecast_data_for_item(predictions, targets, "A", "target", prediction_length=n)
        assert len(rows) <= MAX_FORECAST_POINTS_PER_WINDOW


class TestSeriesRanking:
    """Tests for series ranking metric selection."""

    def test_series_ranking_metric_uses_eval_metric_when_available(self):
        """Uses eval_metric when it's a point metric and available."""
        series_averages = {
            "A": {"mean_absolute_percentage_error": 5.0, "root_mean_squared_error": 10.0},
            "B": {"mean_absolute_percentage_error": 8.0, "root_mean_squared_error": 12.0},
        }
        assert (
            _series_ranking_metric("mean_absolute_percentage_error", series_averages)
            == "mean_absolute_percentage_error"
        )
        assert _series_ranking_metric("root_mean_squared_error", series_averages) == "root_mean_squared_error"

    def test_series_ranking_metric_falls_back_when_mape_unavailable(self):
        """Falls back to root_mean_squared_error when mean_absolute_percentage_error can't be computed (zero denominators)."""  # noqa: E501
        # mean_absolute_percentage_error missing due to zero denominators
        series_averages = {
            "A": {"root_mean_squared_error": 10.0, "mean_absolute_error": 5.0},
            "B": {"root_mean_squared_error": 12.0, "mean_absolute_error": 6.0},
        }
        assert _series_ranking_metric("mean_absolute_percentage_error", series_averages) == "root_mean_squared_error"

    def test_series_ranking_metric_uses_mae_as_last_resort(self):
        """Uses mean_absolute_error when both mean_absolute_percentage_error and root_mean_squared_error unavailable."""
        series_averages = {"A": {"mean_absolute_error": 5.0}, "B": {"mean_absolute_error": 6.0}}
        assert _series_ranking_metric("mean_absolute_percentage_error", series_averages) == "mean_absolute_error"

    def test_series_ranking_metric_defaults_to_mape_for_non_point_metrics(self):
        """For non-point metrics (mean_absolute_scaled_error, weighted_quantile_loss), falls back to mean_absolute_percentage_error if available."""  # noqa: E501
        series_averages = {
            "A": {"mean_absolute_percentage_error": 5.0},
            "B": {"mean_absolute_percentage_error": 8.0},
        }
        assert _series_ranking_metric("mean_absolute_scaled_error", series_averages) == "mean_absolute_percentage_error"


class TestSeriesAnalysis:
    """Tests for series analysis payload construction."""

    def test_no_performers_when_no_series_metrics(self):
        """Best/worst performers are omitted when no series metrics are available."""
        analysis = _build_series_analysis([], [], target="target", prediction_length=2, eval_metric="MAPE")
        assert analysis["num_series_evaluated"] == 0
        assert analysis["best_performer"] is None
        assert analysis["worst_performer"] is None

    def test_item_window_metrics_with_nan_actuals(self):
        """Metrics are computed on the valid subset even when some actuals are NaN."""
        timestamps = ["2025-01-03", "2025-01-04"]
        targets = _make_panel(["A"], timestamps, [100.0, float("nan")])
        predictions = pd.DataFrame({"mean": [105.0, 180.0]}, index=targets.index)
        metrics = _item_window_metrics(predictions, targets, "A", "target", prediction_length=2)
        # Should compute mean_absolute_percentage_error on the 1 valid point (100 → 105), not silently return {}
        assert "mean_absolute_percentage_error" in metrics
        assert metrics["mean_absolute_percentage_error"] == pytest.approx(5.0)

    def test_select_best_worst_missing_value_ranks_as_worst(self):
        """A series with no metric value is ranked worst regardless of metric direction."""
        from ..back_testing import _select_best_worst

        # Test with lower-is-better metric (mean_absolute_percentage_error)
        series_averages_lower = {"A": {"mean_absolute_percentage_error": 5.0}, "B": {}}
        best, worst = _select_best_worst(series_averages_lower, "mean_absolute_percentage_error")
        assert best == "A"
        assert worst == "B"

        # Test with higher-is-better metric (R2)
        series_averages_higher = {"A": {"R2": 0.9}, "B": {}}
        best, worst = _select_best_worst(series_averages_higher, "R2")
        assert best == "A"
        assert worst == "B"

        # Missing lower-is-better metric still ranks as worst
        series_averages_empty = {
            "A": {"mean_absolute_percentage_error": 5.0, "root_mean_squared_error": 10.0},
            "B": {"mean_absolute_percentage_error": 8.0},
        }
        best, worst = _select_best_worst(series_averages_empty, "root_mean_squared_error")
        assert best == "A"
        assert worst == "B"

    def test_build_series_analysis_caches_computation(self):
        """Series analysis generates forecast data once per (item, window), not twice."""
        from unittest.mock import patch

        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["good", "bad"], timestamps, [100.0, 110.0, 120.0, 130.0])
        window_targets = _holdout_frame(train_data, prediction_length=2)
        good_preds = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["good"].index)
        bad_preds = pd.DataFrame({"mean": [200.0, 210.0]}, index=window_targets.loc["bad"].index)
        predictions = pd.concat({"good": good_preds, "bad": bad_preds}, names=["item_id", "timestamp"])

        # Patch the expensive forecast generation function to count calls
        with patch(
            "components.training.automl.shared.back_testing._forecast_data_for_item",
            wraps=_forecast_data_for_item,
        ) as mock_forecast:
            analysis = _build_series_analysis(
                [predictions],
                [window_targets],
                target="target",
                prediction_length=2,
                eval_metric="mean_absolute_percentage_error",
            )

            # _forecast_data_for_item should be called exactly once per (item, window) pair
            # 2 items × 1 window = 2 calls total
            # WITHOUT optimization, it would be called 4 times:
            #   - 2 calls in first pass (computing metrics)
            #   - 2 more calls when building best/worst payloads
            assert mock_forecast.call_count == 2, (
                f"Expected 2 calls (1 per item×window), got {mock_forecast.call_count}. Double computation detected!"
            )

            # Verify analysis structure is still correct
            assert analysis["num_series_evaluated"] == 2
            assert analysis["best_performer"]["item_id"] == "good"
            assert analysis["worst_performer"]["item_id"] == "bad"
            # Verify forecast data is present in the output
            assert len(analysis["best_performer"]["windows"]) == 1
            assert "forecast_data" in analysis["best_performer"]["windows"][0]
            assert len(analysis["best_performer"]["windows"][0]["forecast_data"]) > 0


class TestBuildBackTestingJson:
    """Tests for build_back_testing_json orchestration."""

    def test_cutoff_calculation_matches_autogluon_api(self):
        """Cutoff values are calculated correctly for AutoGluon evaluate() API.

        AutoGluon's cutoff parameter: negative integer where evaluation starts.
        cutoff=-N evaluates from -N-th to (-N + prediction_length)-th time step.

        For num_val_windows=3, prediction_length=2:
        - cutoffs should be [-6, -4, -2]
        - window 0: evaluate steps -6 to -4
        - window 1: evaluate steps -4 to -2
        - window 2: evaluate steps -2 to 0 (end)
        """
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]
        train_data = _make_panel(["A"], timestamps, [100.0, 110.0, 120.0, 130.0, 140.0, 150.0])

        predictor = mock.MagicMock()
        # Create 3 prediction windows
        pred1 = pd.DataFrame({"mean": [121.0, 131.0]}, index=train_data.tail(2).index)
        pred2 = pd.DataFrame({"mean": [131.0, 141.0]}, index=train_data.tail(4).head(2).index)
        pred3 = pd.DataFrame({"mean": [111.0, 121.0]}, index=train_data.tail(6).head(2).index)

        predictor.backtest_predictions.return_value = [pred3, pred2, pred1]  # chronological order
        predictor.backtest_targets.return_value = [
            train_data.tail(6).head(2),
            train_data.tail(4).head(2),
            train_data.tail(2),
        ]

        # Track cutoff values passed to evaluate()
        cutoff_calls = []

        def mock_evaluate(**kwargs):
            cutoff_calls.append(kwargs.get("cutoff"))
            # AutoGluon returns negated error metrics (higher-is-better convention)
            return {"MASE": -0.5}

        predictor.evaluate.side_effect = mock_evaluate

        build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=3,
        )

        # Verify cutoff values: should be [-6, -4, -2] for 3 windows, prediction_length=2
        assert cutoff_calls == [-6, -4, -2], f"Expected [-6, -4, -2], got {cutoff_calls}"

    def test_builds_schema_with_mock_predictor(self):
        """Builder emits ADR-shaped payload from mocked AutoGluon backtest APIs."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["good", "bad"], timestamps, [100.0, 110.0, 120.0, 130.0])

        window_targets = _holdout_frame(train_data, prediction_length=2)
        good_preds = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["good"].index)
        bad_preds = pd.DataFrame({"mean": [200.0, 210.0]}, index=window_targets.loc["bad"].index)
        predictions = pd.concat({"good": good_preds, "bad": bad_preds}, names=["item_id", "timestamp"])

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        # AutoGluon returns negated error metrics (higher-is-better convention)
        predictor.evaluate.return_value = {"MASE": -0.42, "MAPE": -5.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
            metrics=["MASE", "MAPE"],
        )

        assert payload["model_name"] == "DeepAR_FULL"
        assert payload["num_val_windows"] == 1
        assert payload["per_window_metrics"][0]["test_start"] == "2025-01-03"
        assert payload["per_window_metrics"][0]["cutoff"] == -2
        assert payload["per_window_metrics"][0]["metrics"]["MASE"] == 0.42
        assert payload["series_analysis"]["num_series_evaluated"] == 2
        assert payload["series_analysis"]["best_performer"]["item_id"] == "good"
        assert payload["series_analysis"]["worst_performer"]["item_id"] == "bad"
        assert payload["series_analysis"]["best_performer"]["windows"][0]["forecast_data"]
        assert payload["series_analysis"]["best_performer"]["windows"][0]["forecast_data"][0]["timestamp"].endswith("Z")
        assert payload["schema_version"] == 1
        assert "ranking_metric" not in payload["series_analysis"]

    def test_ranks_by_point_metric_matching_eval_metric(self):
        """Best/worst selection uses eval_metric when it is a computed point-forecast metric."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["good", "bad"], timestamps, [100.0, 110.0, 120.0, 130.0])

        window_targets = _holdout_frame(train_data, prediction_length=2)
        good_preds = pd.DataFrame({"mean": [121.0, 131.0]}, index=window_targets.loc["good"].index)
        bad_preds = pd.DataFrame({"mean": [200.0, 210.0]}, index=window_targets.loc["bad"].index)
        predictions = pd.concat({"good": good_preds, "bad": bad_preds}, names=["item_id", "timestamp"])

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        # AutoGluon returns negated error metrics (higher-is-better convention)
        predictor.evaluate.return_value = {"RMSE": -1.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="root_mean_squared_error",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
        )

        assert payload["series_analysis"]["best_performer"]["item_id"] == "good"
        assert "ranking_metric" not in payload["series_analysis"]

    def test_requires_backtest_api(self):
        """Missing backtest methods raise AttributeError."""
        predictor = mock.MagicMock(spec=[])
        with pytest.raises(AttributeError, match="backtest API"):
            build_back_testing_json(
                predictor,
                model_name="DeepAR",
                model_name_full="DeepAR_FULL",
                train_data=pd.DataFrame(),
                eval_metric="MASE",
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                prediction_length=1,
            )

    def test_build_back_testing_json_with_zero_backtest_windows(self):
        """Empty backtest window lists produce an empty but valid payload."""
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        train_data = _make_panel(["A"], timestamps, [100.0, 110.0, 120.0, 130.0])

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = []
        predictor.backtest_targets.return_value = []

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=3,
        )

        assert payload["num_val_windows"] == 0
        assert payload["per_window_metrics"] == []
        assert payload["series_analysis"]["num_series_evaluated"] == 0
        assert payload["series_analysis"]["best_performer"] is None
        assert payload["series_analysis"]["worst_performer"] is None
        predictor.evaluate.assert_not_called()

    def test_single_time_series_no_multiindex(self):
        """Builder handles single time-series (non-MultiIndex) correctly."""
        timestamps = pd.date_range("2025-01-01", periods=6, freq="D")
        train_data = pd.DataFrame(
            {"target": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]},
            index=timestamps,
        )

        # Mock predictions for last 2 points (holdout)
        holdout_timestamps = timestamps[-2:]
        predictions = pd.DataFrame(
            {"mean": [141.0, 151.0]},
            index=holdout_timestamps,
        )

        window_targets = train_data.tail(2)

        predictor = mock.MagicMock()
        predictor.backtest_predictions.return_value = [predictions]
        predictor.backtest_targets.return_value = [window_targets]
        # AutoGluon returns negated error metrics (higher-is-better convention)
        predictor.evaluate.return_value = {"MASE": -0.5, "MAPE": -2.0}

        payload = build_back_testing_json(
            predictor,
            model_name="DeepAR",
            model_name_full="DeepAR_FULL",
            train_data=train_data,
            eval_metric="MASE",
            target="target",
            id_column=None,
            timestamp_column="timestamp",
            prediction_length=2,
            num_val_windows=1,
            metrics=["MASE", "MAPE"],
        )

        assert payload["model_name"] == "DeepAR_FULL"
        assert payload["num_val_windows"] == 1
        assert payload["per_window_metrics"][0]["test_start"] == "2025-01-05"
        assert payload["per_window_metrics"][0]["test_end"] == "2025-01-06"
        assert payload["per_window_metrics"][0]["metrics"]["MASE"] == 0.5
        # Single series: series_analysis should show num_series_evaluated = 1
        assert payload["series_analysis"]["num_series_evaluated"] == 1
        # For single series, best and worst should be None (only one series)
        assert payload["series_analysis"]["best_performer"] is not None
        assert payload["series_analysis"]["worst_performer"] is not None
