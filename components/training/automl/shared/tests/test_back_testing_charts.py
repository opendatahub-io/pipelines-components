"""Tests for back_testing.json matplotlib charts."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd
import pytest

from ..back_testing_charts import (
    _draw_forecast,
    _interval_label,
    _show_per_window_metrics,
    backtest_highlight_item_ids,
    forecast_data_to_frame,
    mean_window_metric,
    notebook_backtest_charts_source,
    per_window_metrics_table,
    pick_window_metric,
    plot_timeseries_forecasts,
    render_back_testing_charts,
    render_back_testing_forecast_charts,
    render_back_testing_metrics,
)


class TestBackTestingCharts:
    """Tests for backtest chart helpers."""

    @staticmethod
    def _sample_forecast_rows():
        return [
            {
                "timestamp": "2025-01-03T00:00:00Z",
                "actual": 100.0,
                "predicted": 105.0,
                "lower_bound": 95.0,
                "upper_bound": 115.0,
                "lower_quantile": 0.1,
                "upper_quantile": 0.9,
            },
            {
                "timestamp": "2025-01-04T00:00:00Z",
                "actual": 110.0,
                "predicted": 108.0,
                "lower_bound": 98.0,
                "upper_bound": 118.0,
                "lower_quantile": 0.1,
                "upper_quantile": 0.9,
            },
        ]

    def test_per_window_metrics_table_includes_cutoff(self):
        """Per-window table exposes AutoGluon-style cutoff identifiers."""
        per_window = [
            {
                "window_id": 0,
                "cutoff": -6,
                "test_start": "2025-03-01",
                "test_end": "2025-03-02",
                "metrics": {"MASE": 0.38},
            },
            {
                "window_id": 1,
                "cutoff": -4,
                "test_start": "2025-03-03",
                "test_end": "2025-03-04",
                "metrics": {"MASE": 0.45},
            },
        ]
        table = per_window_metrics_table(per_window, "MASE")
        assert list(table["cutoff"]) == [-6, -4]
        assert list(table["MASE"]) == [0.38, 0.45]

    def test_mean_window_metric_averages_finite_values(self):
        """Overall summary uses the mean of per-window eval metric values."""
        per_window = [
            {"window_id": 0, "metrics": {"MASE": 0.38}},
            {"window_id": 1, "metrics": {"MASE": 0.42}},
        ]
        assert mean_window_metric(per_window, "MASE") == pytest.approx(0.4)

    def test_show_per_window_metrics_prints_cutoff_lines(self, capsys):
        """Output matches AutoGluon tutorial print format."""
        per_window = [
            {
                "window_id": 0,
                "cutoff": -6,
                "test_start": "2025-03-01",
                "test_end": "2025-03-02",
                "metrics": {"MASE": 0.38},
            },
        ]
        _show_per_window_metrics(per_window, "MASE")
        output = capsys.readouterr().out
        assert "Cutoff -6: MASE = 0.38" in output

    def test_show_per_window_metrics_prints_overall_summary(self, capsys):
        """Overall line aggregates window scores across all series."""
        per_window = [
            {"window_id": 0, "cutoff": -6, "metrics": {"MASE": 0.38}},
            {"window_id": 1, "cutoff": -4, "metrics": {"MASE": 0.42}},
        ]
        _show_per_window_metrics(per_window, "MASE", num_series_evaluated=12)
        output = capsys.readouterr().out
        assert "Overall MASE (mean across 2 validation windows, all series): 0.4" in output
        assert "Series evaluated: 12" in output

    def test_forecast_data_to_frame_parses_timestamps(self):
        """Forecast rows are sorted by timestamp."""
        frame = forecast_data_to_frame(self._sample_forecast_rows())
        assert "lower_quantile" in frame.columns
        assert frame["timestamp"].is_monotonic_increasing

    def test_interval_label_uses_quantile_metadata(self):
        """Interval legend matches AutoGluon-style P10-P90 labeling."""
        frame = forecast_data_to_frame(self._sample_forecast_rows())
        assert _interval_label(frame) == "10%-90% interval"

    def test_draw_forecast_adds_cutoff_line(self):
        """Cutoff vline is drawn at the validation window start."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        _draw_forecast(
            axis,
            self._sample_forecast_rows(),
            title="Window 0",
            target="sales",
            cutoff_start="2025-01-03",
        )
        assert any(line.get_linestyle() == "--" for line in axis.lines)
        plt.close(figure)

    def test_draw_forecast_cutoff_aligns_with_hourly_timestamps(self):
        """Date-only test_start aligns to the first hourly forecast point."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        hourly_rows = [
            {
                "timestamp": "2025-01-03T14:00:00Z",
                "actual": 100.0,
                "predicted": 105.0,
            },
            {
                "timestamp": "2025-01-03T15:00:00Z",
                "actual": 110.0,
                "predicted": 108.0,
            },
        ]
        figure, axis = plt.subplots()
        _draw_forecast(
            axis,
            hourly_rows,
            title="Window 0",
            target="sales",
            cutoff_start="2025-01-03",
        )
        cutoff_lines = [
            line for line in axis.lines if line.get_linestyle() == "--" and line.get_color() in {"gray", "0.5"}
        ]
        assert cutoff_lines
        cutoff_ts = pd.Timestamp(cutoff_lines[0].get_xdata()[0], tz="UTC").tz_convert(None)
        assert cutoff_ts == pd.Timestamp("2025-01-03 14:00:00")
        plt.close(figure)

    def test_pick_window_metric_prefers_eval_metric(self):
        """Metric lookup matches normalized eval metric names."""
        metrics = {"MASE": 0.42, "MAPE": 5.0}
        assert pick_window_metric(metrics, "MASE") == ("MASE", 0.42)
        assert pick_window_metric(metrics, "-MASE") == ("MASE", 0.42)
        assert pick_window_metric(metrics, "mean_absolute_scaled_error") == ("MASE", 0.42)
        assert pick_window_metric({"MAPE": 5.0}, "MASE") == ("MAPE", 5.0)
        assert pick_window_metric({"MAEB": 1.0}, "mean_absolute_error_with_bias") == ("MAEB", 1.0)
        assert pick_window_metric({"WAPEB": 0.2}, "weighted_absolute_percentage_error_with_bias") == ("WAPEB", 0.2)

    def test_aggregate_does_not_substitute_another_metric(self):
        """Missing MASE must not become a MAPE value in a MASE mean."""
        windows = [{"metrics": {"MASE": 0.4}}, {"metrics": {"MAPE": 5.0}}]
        assert mean_window_metric(windows, "mean_absolute_scaled_error") == 0.4
        assert pd.isna(per_window_metrics_table(windows, "MASE").iloc[1]["MASE"])

    def test_notebook_backtest_charts_source_compiles(self):
        """Notebook injection source is valid Python."""
        ast.parse(notebook_backtest_charts_source())

    def test_backtest_highlight_item_ids_prefers_best_and_worst(self):
        """Plot helper returns performers that exist in the sample history."""
        payload = {
            "series_analysis": {
                "best_performer": {"item_id": "north"},
                "worst_performer": {"item_id": "south"},
            }
        }
        assert backtest_highlight_item_ids(payload, ["north", "south", "east"]) == ["north", "south"]

    def test_backtest_highlight_item_ids_resolves_string_ids(self):
        """Numeric JSON ids match string sample ids."""
        payload = {"series_analysis": {"best_performer": {"item_id": 101}}}
        assert backtest_highlight_item_ids(payload, ["101", "102"]) == ["101"]

    def test_backtest_highlight_item_ids_deduplicates_same_performer(self):
        """Single-series backtests do not duplicate plot panels."""
        payload = {
            "series_analysis": {
                "best_performer": {"item_id": "A"},
                "worst_performer": {"item_id": "A"},
            }
        }
        assert backtest_highlight_item_ids(payload, ["A", "B"]) == ["A"]

    def test_render_back_testing_metrics_prints_overall(self, capsys):
        """Metrics entry point prints table output without plotting."""
        payload = {
            "eval_metric": "MASE",
            "per_window_metrics": [
                {"window_id": 0, "cutoff": -6, "metrics": {"MASE": 0.38}},
                {"window_id": 1, "cutoff": -4, "metrics": {"MASE": 0.42}},
            ],
            "series_analysis": {"num_series_evaluated": 3},
        }
        render_back_testing_metrics(payload)
        output = capsys.readouterr().out
        assert "Cutoff -6: MASE = 0.38" in output
        assert "Overall MASE (mean across 2 validation windows, all series): 0.4" in output

    @pytest.mark.parametrize(
        ("eval_metric", "ranking_metric", "ranking_label"),
        [
            ("MASE", "mean_absolute_percentage_error", "ranked by MAPE (fallback for MASE)"),
            ("mean_absolute_percentage_error", "mean_absolute_percentage_error", "ranked by MAPE"),
            ("MASE", None, "ranking metric unavailable (legacy artifact)"),
        ],
    )
    def test_render_back_testing_forecast_charts_runs(self, eval_metric, ranking_metric, ranking_label):
        """Forecast charts entry point renders matplotlib panels only."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        payload = {
            "eval_metric": eval_metric,
            "target": "sales",
            "per_window_metrics": [{"window_id": 0, "test_start": "2025-01-03"}],
            "series_analysis": {
                "best_performer": {
                    "item_id": "A",
                    "windows": [
                        {
                            "window_id": 0,
                            "metrics": {"mean_absolute_percentage_error": 5.0},
                            "forecast_data": self._sample_forecast_rows(),
                        },
                    ],
                },
            },
        }
        if ranking_metric:
            payload["series_analysis"]["ranking_metric"] = ranking_metric
        render_back_testing_forecast_charts(payload)
        figure = plt.gcf()
        assert figure.axes[0].get_title() == "Window 0 (MAPE=5)"
        assert figure._suptitle.get_text() == f"Best performer: A — {ranking_label}"
        # Labels change, actual/predicted curves and interval bounds stay intact.
        assert list(figure.axes[0].lines[0].get_ydata()) == [100.0, 110.0]
        assert list(figure.axes[0].lines[1].get_ydata()) == [105.0, 108.0]
        band = figure.axes[0].collections[0].get_paths()[0].vertices
        assert band[:, 1].min() == 95.0
        assert band[:, 1].max() == 118.0
        plt.close("all")

    def test_render_back_testing_charts_runs(self):
        """Orchestrator renders charts for a minimal payload."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        payload = {
            "eval_metric": "MASE",
            "target": "sales",
            "per_window_metrics": [
                {
                    "window_id": 0,
                    "cutoff": -2,
                    "test_start": "2025-01-01",
                    "test_end": "2025-01-02",
                    "metrics": {"MASE": 0.4},
                },
                {
                    "window_id": 1,
                    "cutoff": -4,
                    "test_start": "2025-01-03",
                    "test_end": "2025-01-04",
                    "metrics": {"MASE": 0.5},
                },
            ],
            "series_analysis": {
                "num_series_evaluated": 5,
                "best_performer": {
                    "item_id": "A",
                    "windows": [
                        {
                            "window_id": 0,
                            "metrics": {"mean_absolute_percentage_error": 5.0},
                            "forecast_data": self._sample_forecast_rows(),
                        },
                        {
                            "window_id": 1,
                            "metrics": {"mean_absolute_percentage_error": 6.0},
                            "forecast_data": self._sample_forecast_rows(),
                        },
                    ],
                },
                "worst_performer": None,
            },
        }
        render_back_testing_charts(payload)
        plt.close("all")

    def test_plot_timeseries_forecasts_styles_date_axis(self):
        """Forecast plot uses rotated short date ticks and marks forecast points."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        history_rows = []
        for offset in range(10):
            day = f"2025-02-{15 + offset:02d}"
            history_rows.append({"sales": 100.0 + offset, "timestamp": day})
        history = pd.DataFrame(history_rows)
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        ts_df = history.set_index("timestamp")

        forecast_rows = []
        for offset in range(2):
            day = f"2025-02-{25 + offset:02d}"
            forecast_rows.append({"mean": 110.0 + offset, "0.1": 105.0, "0.9": 115.0, "timestamp": day})
        forecasts = pd.DataFrame(forecast_rows)
        forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"])
        forecasts = forecasts.set_index("timestamp")

        captured: list[Any] = []

        def capture_show() -> None:
            captured.append(plt.gcf())
            plt.close("all")

        original_show = plt.show
        plt.show = capture_show
        try:
            plot_timeseries_forecasts(ts_df, forecasts, quantile_levels=[0.1, 0.9])
        finally:
            plt.show = original_show

        assert captured
        axis = captured[0].axes[0]
        labels = [label.get_rotation() for label in axis.get_xticklabels()]
        assert labels and labels[0] == 45

        forecast_line = next(line for line in axis.lines if line.get_label() == "Forecast")
        assert forecast_line.get_marker() == "s"
        assert len(forecast_line.get_xdata()) == 2
