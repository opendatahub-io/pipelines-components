"""Unit tests for the autogluon_timeseries_models_training component."""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from ..component import autogluon_timeseries_models_training


@pytest.fixture(autouse=True, scope="module")
def isolated_sys_modules():
    """Patch pandas/autogluon modules for decorator patching stability."""
    with mock.patch.dict(sys.modules, clear=False) as mocked_modules:
        mocked_modules["pandas"] = mock.MagicMock()
        _ag = mock.MagicMock()
        _ag.__path__ = []
        _ag.__spec__ = None
        mocked_modules["autogluon"] = _ag
        _ts = mock.MagicMock()
        _ts.__spec__ = None
        mocked_modules["autogluon.timeseries"] = _ts

        # Mock additional autogluon submodules
        _metrics = mock.MagicMock()
        _metrics.AVAILABLE_METRICS = {"MASE": mock.MagicMock(), "MSE": mock.MagicMock()}
        _metrics.__spec__ = None
        mocked_modules["autogluon.timeseries.metrics"] = _metrics

        _models = mock.MagicMock()
        _models.__spec__ = None
        mocked_modules["autogluon.timeseries.models"] = _models

        _ensemble = mock.MagicMock()
        _ensemble.AbstractTimeSeriesEnsembleModel = type("AbstractTimeSeriesEnsembleModel", (), {})
        _ensemble.__spec__ = None
        mocked_modules["autogluon.timeseries.models.ensemble"] = _ensemble
        yield


@pytest.fixture
def mock_artifacts():
    """Create mock artifacts for full-refit path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock models_artifact
        models_artifact = mock.MagicMock()
        models_artifact.path = str(Path(tmpdir) / "models")
        models_artifact.metadata = {}
        Path(models_artifact.path).mkdir(parents=True, exist_ok=True)

        # Create mock notebooks with a template
        notebooks = mock.MagicMock()
        notebooks.path = str(Path(tmpdir) / "notebooks")
        Path(notebooks.path).mkdir(parents=True, exist_ok=True)
        notebook_template = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "# <REPLACE_RUN_ID>\n",
                        "# <REPLACE_PIPELINE_NAME>\n",
                        "# <REPLACE_MODEL_NAME>\n",
                        "# <REPLACE_SAMPLE_ROW>\n",
                        "# <REPLACE_ID_COLUMN>\n",
                        "# <REPLACE_TIMESTAMP_COLUMN>\n",
                        "# <REPLACE_KNOWN_COVARIATES_NAMES>\n",
                    ],
                }
            ]
        }
        with open(Path(notebooks.path) / "timeseries_notebook.ipynb", "w") as f:
            json.dump(notebook_template, f)

        # Create mock extra_train CSV
        extra_train_path = str(Path(tmpdir) / "extra_train.csv")
        Path(extra_train_path).touch()

        yield models_artifact, notebooks, extra_train_path


def _mock_leaderboard(model_names):
    """Create a leaderboard mock supporting head()['model'].values.tolist()."""

    def _head(n):
        head_mock = mock.MagicMock()
        col = mock.MagicMock()
        col.values.tolist.return_value = model_names[:n]
        head_mock.__getitem__.return_value = col
        return head_mock

    leaderboard = mock.MagicMock()
    leaderboard.head.side_effect = _head
    leaderboard.__len__.return_value = len(model_names)
    leaderboard.iloc = [{"score_test": 0.123}]
    return leaderboard


def _mock_ts_df():
    ts_df = mock.MagicMock()
    ts_df.num_items = 3
    ts_df.__len__.return_value = 30
    return ts_df


class TestTimeseriesModelsTrainingUnitTests:
    """Unit tests for autogluon_timeseries_models_training behavior."""

    def test_component_function_exists(self):
        """Component exposes KFP python_func."""
        assert callable(autogluon_timeseries_models_training)
        assert hasattr(autogluon_timeseries_models_training, "python_func")

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_basic_flow_returns_expected_outputs(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """Happy path returns top models, config, and predictor path with full refit."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        # Mock selection predictor
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR", "TFT", "AutoARIMA"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}, "TFT": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        # Mock refit predictor
        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5, "MSE": 1.0}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor, mock_refit_predictor]

        train_ts, test_ts = _mock_ts_df(), _mock_ts_df()
        extra_ts = _mock_ts_df()
        full_train_ts = _mock_ts_df()
        mock_ts_df_cls.from_data_frame.side_effect = [train_ts, test_ts]
        mock_ts_df_cls.from_path.return_value = extra_ts
        mock_ts_df_cls.return_value = full_train_ts
        mock_concat.return_value = mock.MagicMock()

        train_df, test_df = mock.MagicMock(), mock.MagicMock()
        mock_read_csv.side_effect = [train_df, test_df]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        result = autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=2,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            notebooks=notebooks,
            extra_train_data_path=extra_train_path,
            prediction_length=24,
        )

        mock_read_csv.assert_any_call("/tmp/train.csv")
        mock_read_csv.assert_any_call("/tmp/test.csv")
        mock_ts_df_cls.from_data_frame.assert_any_call(train_df, id_column="item_id", timestamp_column="timestamp")
        mock_ts_df_cls.from_data_frame.assert_any_call(test_df, id_column="item_id", timestamp_column="timestamp")
        mock_ts_df_cls.from_path.assert_called_once_with(
            path=extra_train_path,
            id_column="item_id",
            timestamp_column="timestamp",
        )
        mock_concat.assert_called_once_with([train_ts, extra_ts], axis=0)

        assert result.top_models == ["DeepAR", "TFT"]
        assert result.eval_metric == "MASE"
        assert result.predictor_path == "/tmp/workspace/timeseries_predictor"
        assert result.model_config["prediction_length"] == 24
        assert result.model_config["presets"] == "fast_training"
        assert result.model_config["time_limit"] == 600
        assert result.model_config["known_covariates_names"] == []
        assert result.model_config["num_models_trained"] == 3
        # Verify full refit happened
        assert "model_names" in models_artifact.metadata
        assert "context" in models_artifact.metadata

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_known_covariates_propagated_to_predictor_and_model_config(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """Known covariates are passed to predictor ctor and returned in model_config."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        covariates = ["is_holiday", "promo_flag"]
        result = autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            notebooks=notebooks,
            extra_train_data_path=extra_train_path,
            known_covariates_names=covariates,
        )

        # Check that first predictor call (selection) has known_covariates
        assert mock_predictor_cls.call_args_list[0][1]["known_covariates_names"] == covariates
        assert result.model_config["known_covariates_names"] == covariates
        mock_ts_df_cls.from_path.assert_called_once_with(
            path=extra_train_path,
            id_column="item_id",
            timestamp_column="timestamp",
        )
        mock_concat.assert_called_once()

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_top_n_greater_than_available_models_raises(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """top_n exceeding trained model count raises ValueError."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR", "AutoARIMA"])
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        with pytest.raises(
            ValueError,
            match=r"top_n must be less than or equal to number_of_models_trained \(2\); got 3\.",
        ):
            autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=3,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
            )

    def test_invalid_top_n_zero_raises(self, mock_artifacts):  # noqa: F811
        """top_n must be in range (0, TOP_N_MAX] (see component TOP_N_MAX)."""
        models_artifact, notebooks, extra_train_path = mock_artifacts
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match=r"top_n must be an integer in the range \(0, 7\]; got 0\."):
            autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=0,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
            )

    def test_invalid_top_n_above_max_raises(self, mock_artifacts):  # noqa: F811
        """top_n above TOP_N_MAX is rejected before training."""
        models_artifact, notebooks, extra_train_path = mock_artifacts
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match=r"top_n must be an integer in the range \(0, 7\]; got 8\."):
            autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=8,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
            )

    def test_invalid_prediction_length_raises(self, mock_artifacts):  # noqa: F811
        """prediction_length must be a positive integer."""
        models_artifact, notebooks, extra_train_path = mock_artifacts
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match="prediction_length must be greater than 0"):
            autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=1,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
                prediction_length=0,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_training_failure_is_wrapped(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """Training errors are wrapped in ValueError with component-specific message."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.fit.side_effect = RuntimeError("boom")
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        with pytest.raises(ValueError, match=r"TimeSeriesPredictor training failed: boom"):
            autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_leaderboard_failure_is_wrapped(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """Leaderboard errors are wrapped in ValueError with component-specific message."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.side_effect = RuntimeError("no leaderboard")
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        with pytest.raises(ValueError, match=r"Failed to generate leaderboard: no leaderboard"):
            autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_nan_eval_metric_caught_and_logged(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
        caplog,
    ):
        """When eval metric is NaN/Inf, error is caught, logged, and RuntimeError raised if all models fail."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        # Mock selection predictor
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        # Mock refit predictor with NaN eval metric
        mock_refit_predictor = mock.MagicMock()
        # MASE is NaN - should be caught by error isolation
        mock_refit_predictor.evaluate.return_value = {
            "MASE": float("nan"),
            "MSE": 1.0,
            "MAE": 0.5,
        }

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        # With error isolation, NaN eval metric error is caught and all models fail
        with caplog.at_level("ERROR"):
            with pytest.raises(RuntimeError, match="All models failed refit. No artifacts written."):
                autogluon_timeseries_models_training.python_func(
                    target="sales",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    train_data_path="/tmp/train.csv",
                    test_data=test_data,
                    top_n=1,
                    workspace_path="/tmp/workspace",
                    pipeline_name="ts-pipeline-123",
                    run_id="run-123",
                    models_artifact=models_artifact,
                    notebooks=notebooks,
                    extra_train_data_path=extra_train_path,
                )

        # Verify the specific error was logged
        assert "Eval metric 'MASE' was NaN/Inf for model 'DeepAR_FULL'" in caplog.text
        assert "Refit failed for model 'DeepAR'" in caplog.text

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_partial_refit_failure_succeeds_with_warnings(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
        caplog,
    ):
        """When some models fail refit, component succeeds with partial results and warnings."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        # Mock selection predictor
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR", "TFT", "AutoARIMA"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}, "TFT": {}, "AutoARIMA": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        # Mock refit predictors: first succeeds, second fails, third succeeds
        mock_refit_1 = mock.MagicMock()
        mock_refit_1.evaluate.return_value = {"MASE": 0.5, "MSE": 1.0}

        mock_refit_2 = mock.MagicMock()
        mock_refit_2.fit.side_effect = RuntimeError("OOM during refit")

        mock_refit_3 = mock.MagicMock()
        mock_refit_3.evaluate.return_value = {"MASE": 0.6, "MSE": 1.2}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_1, mock_refit_2, mock_refit_3]
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        with caplog.at_level("WARNING"):
            result = autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=3,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
            )

        # Verify partial success: DeepAR and AutoARIMA succeeded, TFT failed
        assert result.top_models == ["DeepAR", "TFT", "AutoARIMA"]  # Original top models unchanged
        assert "model_names" in models_artifact.metadata
        model_names = json.loads(models_artifact.metadata["model_names"])
        assert model_names == ["DeepAR_FULL", "AutoARIMA_FULL"]  # Only successful models
        assert len(models_artifact.metadata["context"]["models"]) == 2

        # Verify warning was logged
        assert "The following models failed refit: ['TFT']" in caplog.text

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_all_models_fail_refit_raises(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """When all models fail refit, component raises RuntimeError."""
        models_artifact, notebooks, extra_train_path = mock_artifacts

        # Mock selection predictor
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR", "TFT"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}, "TFT": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        # Mock refit predictors: all fail
        mock_refit_1 = mock.MagicMock()
        mock_refit_1.fit.side_effect = RuntimeError("OOM")

        mock_refit_2 = mock.MagicMock()
        mock_refit_2.fit.side_effect = RuntimeError("CUDA error")

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_1, mock_refit_2]
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        with pytest.raises(RuntimeError, match="All models failed refit. No artifacts written."):
            autogluon_timeseries_models_training.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                notebooks=notebooks,
                extra_train_data_path=extra_train_path,
            )
