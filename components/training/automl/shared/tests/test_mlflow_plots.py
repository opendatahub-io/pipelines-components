"""Unit tests for headless MLflow plot rendering."""

import json

import pytest
from kfp_components.components.training.automl.shared import mlflow_plots

pytest.importorskip("matplotlib")


def _write_metrics(model_dir, filename, payload):
    metrics_dir = model_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


class TestRenderConfusionMatrix:
    """Confusion-matrix heatmap rendering."""

    def test_renders_png(self, tmp_path):
        """Render a confusion-matrix PNG from a DataFrame.to_dict() payload."""
        payload = {"cat": {"cat": 5, "dog": 1}, "dog": {"cat": 2, "dog": 7}}
        out = tmp_path / "cm.png"
        result = mlflow_plots.render_confusion_matrix(payload, out)
        assert result == out
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_empty_payload_returns_none(self, tmp_path):
        """Return None for an empty confusion matrix."""
        assert mlflow_plots.render_confusion_matrix({}, tmp_path / "cm.png") is None


class TestRenderRocCurve:
    """ROC-curve rendering for binary and multiclass payloads."""

    def test_binary_roc(self, tmp_path):
        """Render a binary ROC curve."""
        curves = {
            "task_type": "binary",
            "roc_curve": {"auc": 0.9, "fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0]},
        }
        out = tmp_path / "roc.png"
        assert mlflow_plots.render_roc_curve(curves, out) == out
        assert out.is_file()

    def test_multiclass_roc(self, tmp_path):
        """Render a multiclass one-vs-rest ROC curve."""
        curves = {
            "task_type": "multiclass",
            "roc_curve": {
                "auc_macro": 0.88,
                "per_class": {
                    "a": {"auc": 0.9, "fpr": [0.0, 0.4, 1.0], "tpr": [0.0, 0.7, 1.0]},
                    "b": {"auc": 0.85, "fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.6, 1.0]},
                },
            },
        }
        out = tmp_path / "roc_mc.png"
        assert mlflow_plots.render_roc_curve(curves, out) == out
        assert out.is_file()

    def test_missing_roc_returns_none(self, tmp_path):
        """Return None when no roc_curve block is present."""
        assert mlflow_plots.render_roc_curve({"task_type": "binary"}, tmp_path / "roc.png") is None


class TestRenderTimeseriesPlots:
    """Back-testing forecast-vs-actual rendering."""

    def test_actual_only_window_is_still_rendered(self, tmp_path):
        """A window with only actual data (no forecast mean) must still produce a plot."""
        model_dir = tmp_path / "ETS_FULL"
        _write_metrics(
            model_dir,
            "back_testing.json",
            {
                "windows": [
                    {
                        "forecast_data": [
                            {
                                "timestamps": ["2020-01-01", "2020-01-02", "2020-01-03"],
                                "actual": [1.0, 2.0, 3.0],
                            }
                        ]
                    }
                ]
            },
        )
        out_dir = tmp_path / "plots"
        written = mlflow_plots.render_timeseries_plots(model_dir, out_dir)
        assert [p.name for p in written] == ["back_testing.png"]
        assert all(p.is_file() for p in written)

    def test_empty_windows_returns_empty(self, tmp_path):
        """Return no plots when no window supplies any drawable series."""
        model_dir = tmp_path / "ETS_FULL"
        _write_metrics(model_dir, "back_testing.json", {"windows": [{"forecast_data": [{}]}]})
        assert mlflow_plots.render_timeseries_plots(model_dir, tmp_path / "plots") == []


class TestRenderModelPlots:
    """Dispatch by task type over a model directory."""

    def test_classification_dispatch(self, tmp_path):
        """Render confusion matrix and ROC for a classification model dir."""
        model_dir = tmp_path / "LightGBM_BAG_L1_FULL"
        _write_metrics(model_dir, "confusion_matrix.json", {"a": {"a": 3, "b": 1}, "b": {"a": 0, "b": 4}})
        _write_metrics(
            model_dir,
            "curves.json",
            {"task_type": "binary", "roc_curve": {"auc": 0.9, "fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}},
        )
        out_dir = tmp_path / "plots"
        written = mlflow_plots.render_model_plots("binary", model_dir, out_dir)
        names = {p.name for p in written}
        assert names == {"confusion_matrix.png", "roc_curve.png"}
        assert all(p.is_file() for p in written)

    def test_unknown_task_returns_empty(self, tmp_path):
        """Return no plots for an unsupported task type."""
        assert mlflow_plots.render_model_plots("regression", tmp_path, tmp_path / "plots") == []

    def test_missing_data_returns_empty(self, tmp_path):
        """Return no plots when metric JSON files are absent."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        assert mlflow_plots.render_model_plots("binary", model_dir, tmp_path / "plots") == []
