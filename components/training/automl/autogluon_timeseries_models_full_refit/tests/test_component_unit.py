"""Unit tests for the autogluon_timeseries_models_full_refit component."""

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from ..component import autogluon_timeseries_models_full_refit


def _write_notebook_template(tmp_path):
    notebooks_dir = tmp_path / "notebooks_input"
    notebooks_dir.mkdir()
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    'pipeline = "<REPLACE_PIPELINE_NAME>"\n',
                    'run = "<REPLACE_RUN_ID>"\n',
                    'model = "<REPLACE_MODEL_NAME>"\n',
                    "row = <REPLACE_SAMPLE_ROW>\n",
                    'id_col = "<REPLACE_ID_COLUMN>"\n',
                    'ts_col = "<REPLACE_TIMESTAMP_COLUMN>"\n',
                    "covariates = <REPLACE_KNOWN_COVARIATES_NAMES>\n",
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with (notebooks_dir / "timeseries_notebook.ipynb").open("w", encoding="utf-8") as f:
        json.dump(notebook, f)
    notebooks_artifact = mock.MagicMock()
    notebooks_artifact.path = str(notebooks_dir)
    return notebooks_artifact


def _make_artifacts(tmp_path):
    test_dataset = mock.MagicMock()
    test_dataset.path = str(tmp_path / "test.csv")
    Path(test_dataset.path).write_text("item_id,timestamp,target\nA,2024-01-01,1\n", encoding="utf-8")

    model_artifact = mock.MagicMock()
    model_artifact.path = str(tmp_path / "model_output")
    model_artifact.metadata = {}
    Path(model_artifact.path).mkdir(parents=True, exist_ok=True)
    return test_dataset, model_artifact


class _LenBox:
    """Length-only payload so FakeTimeSeriesDataFrame and fake pandas.concat stay stdlib-only."""

    def __init__(self, n: int) -> None:
        """Store a row count used only via ``len()``."""
        self._n = n

    def __len__(self) -> int:
        """Return the stored row count."""
        return self._n


class FakeTimeSeriesDataFrame:
    """Stand-in for TimeSeriesDataFrame without importing pandas (import-guard)."""

    def __init__(self, data=None) -> None:
        """Wrap optional length-bearing ``data`` (e.g. ``_LenBox``)."""
        self._payload = data
        self._len = len(data) if data is not None else 0

    def __len__(self) -> int:
        """Return logical row count for tests."""
        return self._len

    @staticmethod
    def from_path(path, id_column, timestamp_column):
        """Return a stub frame with two logical rows (matches real loader split slices)."""
        _ = (path, id_column, timestamp_column)
        return FakeTimeSeriesDataFrame(_LenBox(2))


def _install_autogluon_mocks(
    *,
    is_ensemble=False,
    load_side_effect=None,
    save_side_effect=None,
    evaluate_side_effect=None,
):
    """Patch sys.modules with lightweight autogluon mocks used inside python_func."""
    fake_pandas = ModuleType("pandas")

    def fake_concat(objs, axis=0):
        _ = axis
        return _LenBox(sum(len(o) for o in objs))

    fake_pandas.concat = fake_concat

    fake_autogluon = ModuleType("autogluon")
    fake_timeseries = ModuleType("autogluon.timeseries")
    fake_metrics = ModuleType("autogluon.timeseries.metrics")
    fake_models = ModuleType("autogluon.timeseries.models")
    fake_ensemble = ModuleType("autogluon.timeseries.models.ensemble")

    class FakeAbstractTimeSeriesEnsembleModel:
        pass

    class FakeNonEnsembleModel:
        pass

    loaded_predictor = mock.MagicMock()
    loaded_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {"epochs": 3}}}
    trainer = mock.MagicMock()
    trainer.get_model_attribute.return_value = (
        FakeAbstractTimeSeriesEnsembleModel if is_ensemble else FakeNonEnsembleModel
    )
    loaded_predictor._trainer = trainer

    refit_predictor = mock.MagicMock()
    refit_predictor.evaluate.return_value = {"MASE": 1.23}
    if evaluate_side_effect is not None:
        refit_predictor.evaluate.side_effect = evaluate_side_effect

    def _predictor_ctor(*args, **kwargs):
        predictor_path = kwargs.get("path")
        if save_side_effect is not None:
            refit_predictor.save.side_effect = save_side_effect
        else:
            # Real predictor.save() writes artifacts under predictor path; mimic that behavior.
            refit_predictor.save.side_effect = lambda: Path(predictor_path).mkdir(parents=True, exist_ok=True)
        return refit_predictor

    predictor_class = mock.MagicMock(side_effect=_predictor_ctor)
    if load_side_effect is not None:
        predictor_class.load.side_effect = load_side_effect
    else:
        predictor_class.load.return_value = loaded_predictor
    predictor_class.return_value = refit_predictor

    fake_timeseries.TimeSeriesDataFrame = FakeTimeSeriesDataFrame
    fake_timeseries.TimeSeriesPredictor = predictor_class
    fake_metrics.AVAILABLE_METRICS = {"MASE": object(), "RMSE": object()}
    fake_ensemble.AbstractTimeSeriesEnsembleModel = FakeAbstractTimeSeriesEnsembleModel

    patcher = mock.patch.dict(
        sys.modules,
        {
            "pandas": fake_pandas,
            "autogluon": fake_autogluon,
            "autogluon.timeseries": fake_timeseries,
            "autogluon.timeseries.metrics": fake_metrics,
            "autogluon.timeseries.models": fake_models,
            "autogluon.timeseries.models.ensemble": fake_ensemble,
        },
    )
    return patcher, predictor_class, loaded_predictor, refit_predictor


class TestTimeseriesModelsFullRefitUnitTests:
    """Unit tests for timeseries full refit behavior."""

    def test_component_function_exists(self):
        """Component exposes a KFP python_func entrypoint."""
        assert callable(autogluon_timeseries_models_full_refit)
        assert hasattr(autogluon_timeseries_models_full_refit, "python_func")

    def test_full_refit_writes_outputs_and_metadata(self, tmp_path):
        """Successful run writes metrics/metadata/notebook and updates artifact metadata."""
        test_dataset, model_artifact = _make_artifacts(tmp_path)
        notebooks = _write_notebook_template(tmp_path)

        patcher, predictor_class, loaded_predictor, refit_predictor = _install_autogluon_mocks()
        with patcher:
            autogluon_timeseries_models_full_refit.python_func(
                model_name="DeepAR",
                test_dataset=test_dataset,
                predictor_path="/tmp/predictor",
                sampling_config={"sampling_method": "first_n_rows"},
                split_config={"test_size": 0.2},
                model_config={
                    "prediction_length": 7,
                    "target": "target",
                    "id_column": "item_id",
                    "timestamp_column": "timestamp",
                    "eval_metric": "MASE",
                },
                pipeline_name="my-pipeline-run-123",
                run_id="run-456",
                models_selection_train_data_path=str(tmp_path / "selection_train.csv"),
                extra_train_data_path=str(tmp_path / "extra_train.csv"),
                sample_rows='[{"item_id":"A","timestamp":"2024-01-01","target":1}]',
                notebooks=notebooks,
                model_artifact=model_artifact,
            )

        predictor_class.load.assert_called_once_with("/tmp/predictor")
        loaded_predictor.fit_summary.assert_called_once()
        refit_predictor.fit.assert_called_once()
        fit_kwargs = refit_predictor.fit.call_args[1]
        assert "hyperparameters" in fit_kwargs
        assert fit_kwargs["hyperparameters"]["DeepAR"] == {"epochs": 3}
        assert len(fit_kwargs["train_data"]) == 4  # selection (2) + extra (2) rows

        model_dir = Path(model_artifact.path) / "DeepAR_FULL"
        assert (model_dir / "predictor" / "predictor_metadata.json").exists()
        assert (model_dir / "metrics" / "metrics.json").exists()
        assert (model_dir / "notebooks" / "automl_predictor_notebook.ipynb").exists()

        metrics = json.loads((model_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics == {"MASE": 1.23}

        notebook_text = (model_dir / "notebooks" / "automl_predictor_notebook.ipynb").read_text(encoding="utf-8")
        assert "<REPLACE_" not in notebook_text
        assert "my-pipeline-run" in notebook_text
        assert "my-pipeline-run-123" not in notebook_text
        assert "DeepAR_FULL" in notebook_text
        assert "run-456" in notebook_text
        assert "item_id" in notebook_text
        assert "timestamp" in notebook_text
        assert "[]" in notebook_text

        # model.json written alongside predictor/, metrics/, notebooks/
        model_json_path = model_dir / "model.json"
        assert model_json_path.exists()
        model_meta = json.loads(model_json_path.read_text(encoding="utf-8"))
        assert model_meta["name"] == "DeepAR_FULL"
        assert "base_model" not in model_meta
        assert model_meta["location"]["model_directory"] == "DeepAR_FULL"
        assert "predictor" in model_meta["location"]
        assert model_meta["location"]["notebook"] == "DeepAR_FULL/notebooks/automl_predictor_notebook.ipynb"
        assert model_meta["location"]["metrics"] == "DeepAR_FULL/metrics"
        assert model_meta["metrics"]["test_data"] == {"MASE": 1.23}

        assert model_artifact.metadata["display_name"] == "DeepAR_FULL"
        assert model_artifact.metadata["context"]["pipeline_info"]["pipeline_name"] == "my-pipeline-run"
        assert model_artifact.metadata["context"]["metrics"]["test_data"] == {"MASE": 1.23}

    def test_model_json_matches_artifact_metadata(self, tmp_path):
        """model.json on disk is consistent with artifact metadata context."""
        test_dataset, model_artifact = _make_artifacts(tmp_path)
        notebooks = _write_notebook_template(tmp_path)

        patcher, _, _, _ = _install_autogluon_mocks()
        with patcher:
            autogluon_timeseries_models_full_refit.python_func(
                model_name="DeepAR",
                test_dataset=test_dataset,
                predictor_path="/tmp/predictor",
                sampling_config={},
                split_config={},
                model_config={
                    "prediction_length": 7,
                    "target": "target",
                    "id_column": "item_id",
                    "timestamp_column": "timestamp",
                    "eval_metric": "MASE",
                },
                pipeline_name="pipe-run-1",
                run_id="run-1",
                models_selection_train_data_path=str(tmp_path / "selection_train.csv"),
                extra_train_data_path=str(tmp_path / "extra_train.csv"),
                sample_rows='[{"target":1}]',
                notebooks=notebooks,
                model_artifact=model_artifact,
            )

        model_dir = Path(model_artifact.path) / "DeepAR_FULL"
        model_json = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        context = model_artifact.metadata["context"]

        # model.json location matches artifact metadata context location
        assert model_json["location"] == context["location"]
        # model.json metrics matches artifact metadata context metrics
        assert model_json["metrics"]["test_data"] == context["metrics"]["test_data"]
        # model.json name matches artifact display_name
        assert model_json["name"] == model_artifact.metadata["display_name"]

    def test_full_refit_uses_ensemble_hyperparameters_for_ensemble_model(self, tmp_path):
        """Ensemble models use ensemble_hyperparameters key during fit."""
        test_dataset, model_artifact = _make_artifacts(tmp_path)
        notebooks = _write_notebook_template(tmp_path)
        patcher, _, _, refit_predictor = _install_autogluon_mocks(is_ensemble=True)

        with patcher:
            autogluon_timeseries_models_full_refit.python_func(
                model_name="DeepAR",
                test_dataset=test_dataset,
                predictor_path="/tmp/predictor",
                sampling_config={},
                split_config={},
                model_config={
                    "prediction_length": 7,
                    "target": "target",
                    "id_column": "item_id",
                    "timestamp_column": "timestamp",
                },
                pipeline_name="pipe-1",
                run_id="run-1",
                models_selection_train_data_path=str(tmp_path / "selection_train.csv"),
                extra_train_data_path=str(tmp_path / "extra_train.csv"),
                sample_rows='[{"target":1}]',
                notebooks=notebooks,
                model_artifact=model_artifact,
            )

        fit_kwargs = refit_predictor.fit.call_args[1]
        assert "ensemble_hyperparameters" in fit_kwargs
        assert "hyperparameters" not in fit_kwargs

    def test_full_refit_raises_when_predictor_load_fails(self, tmp_path):
        """Predictor load failures are wrapped with a clear ValueError."""
        test_dataset, model_artifact = _make_artifacts(tmp_path)
        notebooks = _write_notebook_template(tmp_path)
        patcher, _, _, _ = _install_autogluon_mocks(load_side_effect=FileNotFoundError("missing predictor"))

        with patcher:
            with pytest.raises(ValueError, match=r"Could not load predictor from /tmp/predictor: missing predictor"):
                autogluon_timeseries_models_full_refit.python_func(
                    model_name="DeepAR",
                    test_dataset=test_dataset,
                    predictor_path="/tmp/predictor",
                    sampling_config={},
                    split_config={},
                    model_config={
                        "prediction_length": 7,
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                    },
                    pipeline_name="pipe-1",
                    run_id="run-1",
                    models_selection_train_data_path=str(tmp_path / "selection_train.csv"),
                    extra_train_data_path=str(tmp_path / "extra_train.csv"),
                    sample_rows='[{"target":1}]',
                    notebooks=notebooks,
                    model_artifact=model_artifact,
                )

    def test_full_refit_raises_when_predictor_save_fails(self, tmp_path):
        """Save failures are wrapped with a clear ValueError."""
        test_dataset, model_artifact = _make_artifacts(tmp_path)
        notebooks = _write_notebook_template(tmp_path)
        patcher, _, _, _ = _install_autogluon_mocks(save_side_effect=OSError("disk full"))

        with patcher:
            with pytest.raises(ValueError, match=r"Could not save predictor .* disk full"):
                autogluon_timeseries_models_full_refit.python_func(
                    model_name="DeepAR",
                    test_dataset=test_dataset,
                    predictor_path="/tmp/predictor",
                    sampling_config={},
                    split_config={},
                    model_config={
                        "prediction_length": 7,
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                    },
                    pipeline_name="pipe-1",
                    run_id="run-1",
                    models_selection_train_data_path=str(tmp_path / "selection_train.csv"),
                    extra_train_data_path=str(tmp_path / "extra_train.csv"),
                    sample_rows='[{"target":1}]',
                    notebooks=notebooks,
                    model_artifact=model_artifact,
                )

    def test_full_refit_raises_when_evaluation_fails(self, tmp_path):
        """Evaluation failures are wrapped with a clear ValueError."""
        test_dataset, model_artifact = _make_artifacts(tmp_path)
        notebooks = _write_notebook_template(tmp_path)
        patcher, _, _, _ = _install_autogluon_mocks(evaluate_side_effect=RuntimeError("bad eval"))

        with patcher:
            with pytest.raises(ValueError, match=r"Failed to evaluate model: bad eval"):
                autogluon_timeseries_models_full_refit.python_func(
                    model_name="DeepAR",
                    test_dataset=test_dataset,
                    predictor_path="/tmp/predictor",
                    sampling_config={},
                    split_config={},
                    model_config={
                        "prediction_length": 7,
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                    },
                    pipeline_name="pipe-1",
                    run_id="run-1",
                    models_selection_train_data_path=str(tmp_path / "selection_train.csv"),
                    extra_train_data_path=str(tmp_path / "extra_train.csv"),
                    sample_rows='[{"target":1}]',
                    notebooks=notebooks,
                    model_artifact=model_artifact,
                )
