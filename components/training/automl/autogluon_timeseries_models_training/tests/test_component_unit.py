"""Unit tests for the autogluon_timeseries_models_training component."""

import json
import logging
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
        _metrics.AVAILABLE_METRICS = {
            k: mock.MagicMock()
            for k in (
                "mean_absolute_scaled_error",
                "mean_squared_error",
                "weighted_quantile_loss",
                "root_mean_squared_scaled_error",
                "mean_absolute_error",
                "root_mean_squared_error",
                "sql",
            )
        }
        _metrics.METRIC_ALIASES = {
            "mean_absolute_scaled_error": "MASE",
            "root_mean_squared_error": "RMSE",
            "weighted_quantile_loss": "WQL",
            "mean_squared_error": "MSE",
            "mean_absolute_error": "MAE",
            "root_mean_squared_scaled_error": "RMSSE",
        }
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

        extra_train_path = str(Path(tmpdir) / "extra_train.csv")
        Path(extra_train_path).touch()

        html_artifact = _make_html_artifact(Path(tmpdir))

        yield models_artifact, extra_train_path, html_artifact


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


def _make_component_status_artifact(tmp_path):
    art = mock.MagicMock()
    art.path = str(tmp_path / "component_status_out")
    art.metadata = {}
    return art


def _make_html_artifact(tmp_path):
    art = mock.MagicMock()
    art.path = str(tmp_path / "leaderboard.html")
    art.metadata = {}
    return art


_DEFAULT_COMPONENT_STATUS = _make_component_status_artifact(Path("/tmp"))
_DEFAULT_HTML_ARTIFACT = _make_html_artifact(Path("/tmp"))


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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        # Mock selection predictor
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR", "TFT", "AutoARIMA"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}, "TFT": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        # Mock refit predictor
        mock_refit_predictor = mock.MagicMock()
        # AutoGluon returns uppercase acronym keys from evaluate() regardless of the
        # metric names passed in; the component normalizes these via _acronym_to_snake.
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5, "MSE": 1.0}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor, mock_refit_predictor]

        extra_ts = _mock_ts_df()
        full_train_ts = _mock_ts_df()
        # from_data_frame is called for: train, test, and once per model for build_predict_sample_artifact
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
            extra_train_data_path=extra_train_path,
            prediction_length=24,
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
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
        mock_concat.assert_called_once()
        concat_frames = mock_concat.call_args[0][0]
        assert len(concat_frames) == 2
        assert concat_frames[1] is extra_ts

        assert result.top_models == ["DeepAR", "TFT"]
        assert result.eval_metric == "mean_absolute_scaled_error"
        assert result.predictor_path == "/tmp/workspace/timeseries_predictor"
        assert result.model_config["prediction_length"] == 24
        assert result.model_config["presets"] == "speed"
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
    def test_balanced_preset_fit_args(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """Balanced preset uses medium_quality and 60-minute time limit."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

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
            extra_train_data_path=extra_train_path,
            preset="balanced",
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        fit_call = mock_predictor.fit.call_args
        assert fit_call[1]["presets"] == "medium_quality"
        assert fit_call[1]["time_limit"] == 60 * 60
        assert result.model_config["presets"] == "balanced"
        assert result.model_config["time_limit"] == 60 * 60

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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        # from_data_frame is called for: train, test, and once per model for build_predict_sample_artifact (1 model)
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
            extra_train_data_path=extra_train_path,
            known_covariates_names=covariates,
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR", "AutoARIMA"])
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_invalid_top_n_zero_raises(self, mock_artifacts):  # noqa: F811
        """top_n must be in range (0, TOP_N_MAX] (see component TOP_N_MAX)."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
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
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_invalid_top_n_above_max_raises(self, mock_artifacts):  # noqa: F811
        """top_n above TOP_N_MAX is rejected before training."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
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
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_invalid_prediction_length_raises(self, mock_artifacts):  # noqa: F811
        """prediction_length must be a positive integer."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
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
                extra_train_data_path=extra_train_path,
                prediction_length=0,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_invalid_preset(self, mock_artifacts):
        """Preset must be one of the valid AutoGluon quality tiers."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match="preset must be one of"):
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
                extra_train_data_path=extra_train_path,
                preset="best_quality",
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.fit.side_effect = RuntimeError("boom")
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.side_effect = RuntimeError("no leaderboard")
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

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
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
                    extra_train_data_path=extra_train_path,
                    html_artifact=html_artifact,
                    component_status=_DEFAULT_COMPONENT_STATUS,
                )

        # Verify the specific error was logged
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("NaN" in r.message and "DeepAR_FULL" in r.message for r in error_records)
        assert any("Refit failed" in r.message and "DeepAR" in r.message for r in error_records)

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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

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
        # from_data_frame: train, test, and successful refits (DeepAR, AutoARIMA) in build_predict_sample_artifact
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

        # Verify partial success: DeepAR and AutoARIMA succeeded, TFT failed
        # top_models: Original selection-phase ranking, returned unchanged for traceability
        assert result.top_models == ["DeepAR", "TFT", "AutoARIMA"]
        # model_names: Only models that successfully completed refit and were persisted
        # This is the authoritative list of what's actually in models_artifact.path
        assert "model_names" in models_artifact.metadata
        model_names = json.loads(models_artifact.metadata["model_names"])
        assert model_names == ["DeepAR_FULL", "AutoARIMA_FULL"]  # TFT omitted - refit failed
        assert len(models_artifact.metadata["context"]["models"]) == 2

        # Verify warning was logged
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("TFT" in r.message and "failed refit" in r.message for r in warning_records)

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
        models_artifact, extra_train_path, html_artifact = mock_artifacts

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
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    # ── eval_metric parameter ─────────────────────────────────────────────────

    def test_empty_eval_metric_raises(self, mock_artifacts):  # noqa: F811
        """eval_metric must be a non-empty string."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(TypeError, match="eval_metric must be a non-empty string"):
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
                extra_train_data_path=extra_train_path,
                eval_metric="",
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_unsupported_eval_metric_raises(self, mock_artifacts):  # noqa: F811
        """eval_metric not in METRIC_ALIASES raises ValueError before training."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match="eval_metric must be one of"):
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
                extra_train_data_path=extra_train_path,
                eval_metric="BADMETRIC",
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_sql_metric_not_in_metric_aliases_raises(self, mock_artifacts):  # noqa: F811
        """'sql' is in AVAILABLE_METRICS but not METRIC_ALIASES; passes through normalization and fails validation."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match="eval_metric must be one of"):
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
                extra_train_data_path=extra_train_path,
                eval_metric="sql",
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_legacy_acronym_mase_normalized_to_snake_case(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """MASE (old default) is silently normalized to mean_absolute_scaled_error."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        # AutoGluon returns uppercase acronym keys from evaluate() regardless of the
        # metric names passed in; the component normalizes these via _acronym_to_snake.
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5, "MSE": 1.0}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

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
            extra_train_data_path=extra_train_path,
            eval_metric="MASE",
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        assert result.eval_metric == "mean_absolute_scaled_error"
        assert result.model_config["eval_metric"] == "mean_absolute_scaled_error"
        assert models_artifact.metadata["context"]["model_config"]["eval_metric"] == "mean_absolute_scaled_error"
        for call in mock_predictor_cls.call_args_list:
            assert call.kwargs["eval_metric"] == "mean_absolute_scaled_error"

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_custom_eval_metric_passed_to_predictors_and_returned(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
    ):
        """Custom eval_metric is passed to both TimeSeriesPredictor constructors, stored in model_config, and returned."""  # noqa: E501
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        # AutoGluon returns uppercase acronym keys from evaluate() regardless of the
        # metric names passed in; the component normalizes these via _acronym_to_snake.
        mock_refit_predictor.evaluate.return_value = {"WQL": 0.3, "MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        # from_data_frame: train, test, and once per model in build_predict_sample_artifact (1 model)
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

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
            extra_train_data_path=extra_train_path,
            eval_metric="WQL",
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        for call in mock_predictor_cls.call_args_list:
            assert call.kwargs["eval_metric"] == "weighted_quantile_loss"

        assert result.eval_metric == "weighted_quantile_loss"
        assert result.model_config["eval_metric"] == "weighted_quantile_loss"
        assert models_artifact.metadata["context"]["model_config"]["eval_metric"] == "weighted_quantile_loss"


class TestMetricsJsonSignConvention:
    """Tests for metrics.json sign convention (raw AutoGluon, leaderboard-compatible)."""

    @mock.patch("kfp_components.components.training.automl.shared.back_testing.build_back_testing_json")
    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_metrics_json_preserves_autogluon_evaluate_signs(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_build_back_testing_json,
        mock_artifacts,  # noqa: F811
    ):
        """metrics.json keeps negated error metrics from AutoGluon evaluate() for leaderboard sorting."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
        mock_build_back_testing_json.return_value = {"schema_version": 1}

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        # AutoGluon returns uppercase acronym keys from evaluate() regardless of the
        # metric names passed in; the component normalizes these via _acronym_to_snake.
        mock_refit_predictor.evaluate.return_value = {"MASE": -0.42, "MSE": -1.0}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        # from_data_frame is called for: train, test, and once per model for build_predict_sample_artifact (1 model)
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

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
            extra_train_data_path=extra_train_path,
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        metrics_path = Path(models_artifact.path) / "DeepAR_FULL" / "metrics" / "metrics.json"
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        assert metrics["mean_absolute_scaled_error"] == -0.42
        assert metrics["mean_squared_error"] == -1.0


class TestBackTestingArtifactFailure:
    """Tests for best-effort back_testing.json generation in the component."""

    @mock.patch("kfp_components.components.training.automl.shared.back_testing.build_back_testing_json")
    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_back_testing_failure_is_non_fatal(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_build_back_testing_json,
        mock_artifacts,  # noqa: F811
        caplog,
    ):
        """Component continues when back_testing.json generation fails."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts
        mock_build_back_testing_json.side_effect = RuntimeError("backtest unavailable")

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5, "MSE": 1.0}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        # from_data_frame is called for: train, test, and once per model for build_predict_sample_artifact (1 model)
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
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
                top_n=1,
                workspace_path="/tmp/workspace",
                pipeline_name="ts-pipeline-123",
                run_id="run-123",
                models_artifact=models_artifact,
                extra_train_data_path=extra_train_path,
                html_artifact=html_artifact,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

        metrics_dir = Path(models_artifact.path) / "DeepAR_FULL" / "metrics"
        assert result.top_models == ["DeepAR"]
        assert (metrics_dir / "metrics.json").is_file()
        assert not (metrics_dir / "back_testing.json").exists()
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("back_testing.json" in r.message for r in warning_records)
        assert "Could not generate back_testing.json" in caplog.text


class TestLeaderboardPhase:
    """Tests for the Phase C leaderboard generation inlined into autogluon_timeseries_models_training."""

    def _configure_pd_mock_for_leaderboard(self, best_model: str, metrics: dict):
        """Set up sys.modules["pandas"].DataFrame so Phase C produces a deterministic leaderboard.

        Phase C calls pd.DataFrame(rows).sort_values(...) and then reads iloc[0]["model"].
        Since pandas is module-mocked, we wire the return-value chain explicitly.
        """
        mock_sorted_df = mock.MagicMock()
        mock_sorted_df.iloc = [{"model": best_model}]
        mock_sorted_df.to_json.return_value = json.dumps([{"model": best_model, **metrics}])
        import sys

        sys.modules["pandas"].DataFrame.return_value.sort_values.return_value = mock_sorted_df
        return mock_sorted_df

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_leaderboard_html_written_and_best_model_returned(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
        tmp_path,
    ):
        """Phase C writes the leaderboard HTML file and returns best_model_name in the result."""
        models_artifact, extra_train_path, _ = mock_artifacts
        self._configure_pd_mock_for_leaderboard("DeepAR_FULL", {"MASE": -0.42})

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock
        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": -0.42}
        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df(), _mock_ts_df()]
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

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
            extra_train_data_path=extra_train_path,
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        # HTML file written to disk
        assert Path(html_artifact.path).exists()
        html_text = Path(html_artifact.path).read_text(encoding="utf-8")
        assert "mean_absolute_scaled_error" in html_text
        assert "DeepAR_FULL" in html_text

        # best_model_name is the top-ranked model
        assert result.best_model_name == "DeepAR_FULL"

        # context["best_model_name"] matches the return value
        assert models_artifact.metadata["context"]["best_model_name"] == result.best_model_name

        # html_artifact metadata populated
        assert html_artifact.metadata["display_name"] == "automl_leaderboard"
        data_raw = html_artifact.metadata["data"]
        assert isinstance(data_raw, str), "html_artifact.metadata['data'] must be a JSON string"
        data_parsed = json.loads(data_raw)
        assert isinstance(data_parsed, list)

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_leaderboard_best_model_name_consistent_across_return_and_context(
        self,
        mock_predictor_cls,
        mock_ts_df_cls,
        mock_concat,
        mock_read_csv,
        mock_artifacts,  # noqa: F811
        tmp_path,
    ):
        """best_model_name in the return tuple and in models_artifact context are identical."""
        models_artifact, extra_train_path, _ = mock_artifacts
        self._configure_pd_mock_for_leaderboard("TFT_FULL", {"MASE": -0.30})

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["TFT"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"TFT": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock
        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": -0.30}
        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df(), _mock_ts_df()]
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard2.html")
        html_artifact.metadata = {}

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
            extra_train_data_path=extra_train_path,
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        assert result.best_model_name == "TFT_FULL"
        context = models_artifact.metadata["context"]
        assert "best_model_name" in context
        assert context["best_model_name"] == result.best_model_name


class TestPredictorMetadata:
    """Verify predictor_metadata.json is written for each refit model."""

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_predictor_metadata_json_written(
        self, mock_predictor_cls, mock_ts_df_cls, mock_concat, mock_read_csv, mock_artifacts
    ):
        """predictor/predictor_metadata.json exists and captures selected-model config."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="product_id",
            timestamp_column="date",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            extra_train_data_path=extra_train_path,
            prediction_length=7,
            eval_metric="MASE",
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        metadata_path = Path(models_artifact.path) / "DeepAR_FULL" / "predictor" / "predictor_metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata == {
            "model_name": "DeepAR_FULL",
            "base_model": "DeepAR",
            "prediction_length": 7,
            "eval_metric": "mean_absolute_scaled_error",
            "target": "sales",
            "id_column": "product_id",
            "timestamp_column": "date",
            "known_covariates_names": [],
            "uses_synthetic_id": False,
        }

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_predictor_metadata_includes_known_covariates(
        self, mock_predictor_cls, mock_ts_df_cls, mock_concat, mock_read_csv, mock_artifacts
    ):
        """predictor_metadata.json records the covariate columns the model was trained with."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = _mock_ts_df()
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="product_id",
            timestamp_column="date",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            extra_train_data_path=extra_train_path,
            prediction_length=7,
            known_covariates_names=["promo", "temperature"],
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        metadata_path = Path(models_artifact.path) / "DeepAR_FULL" / "predictor" / "predictor_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["known_covariates_names"] == ["promo", "temperature"]


class TestTimeseriesInferenceBlock:
    """Verify model.json includes inference.input_data_schema and sample_payload."""

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_model_json_includes_inference_block(
        self, mock_predictor_cls, mock_ts_df_cls, mock_concat, mock_read_csv, mock_artifacts
    ):
        """model.json contains inference with instances fields for id, timestamp, target."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]

        full_train_ts = _mock_ts_df()
        full_train_ts.columns = ["sales"]
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = full_train_ts
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="product_id",
            timestamp_column="date",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            extra_train_data_path=extra_train_path,
            prediction_length=7,
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        model_json_path = Path(models_artifact.path) / "DeepAR_FULL" / "model.json"
        assert model_json_path.exists()
        model_json = json.loads(model_json_path.read_text())

        assert "inference" in model_json
        schema = model_json["inference"]["input_data_schema"]
        assert schema["protocol"] == "v1_json"
        assert schema["prediction_length"] == 7
        assert schema["instances"]["required"] is True

        fields = schema["instances"]["fields"]
        assert len(fields) == 3
        assert fields[0] == {"name": "product_id", "datatype": "string", "role": "id", "required": True}
        assert fields[1] == {"name": "date", "datatype": "string", "role": "timestamp", "required": True}
        assert fields[2] == {"name": "sales", "datatype": "number", "role": "target", "required": True}

        assert "known_covariates" not in schema

        payload = model_json["inference"]["sample_payload"]
        assert payload == {"instances": [{"product_id": "<string>", "date": "<string>", "sales": "<number>"}]}
        assert "known_covariates" not in payload

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_inference_block_with_known_covariates(
        self, mock_predictor_cls, mock_ts_df_cls, mock_concat, mock_read_csv, mock_artifacts
    ):
        """Inference block includes known_covariates when model has covariate columns."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]

        full_train_ts = _mock_ts_df()
        full_train_ts.columns = ["sales", "promo", "temperature"]
        promo_col = mock.MagicMock()
        promo_col.dtype = "int64"
        temp_col = mock.MagicMock()
        temp_col.dtype = "float64"
        full_train_ts.__getitem__ = lambda self, key: promo_col if key == "promo" else temp_col
        full_train_ts.__contains__ = lambda self, key: key in ("promo", "temperature", "sales")
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = full_train_ts
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="product_id",
            timestamp_column="date",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            extra_train_data_path=extra_train_path,
            prediction_length=7,
            known_covariates_names=["promo", "temperature"],
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        model_json_path = Path(models_artifact.path) / "DeepAR_FULL" / "model.json"
        model_json = json.loads(model_json_path.read_text())

        schema = model_json["inference"]["input_data_schema"]

        # Covariate columns must also be present in the historical instances.
        instance_fields = schema["instances"]["fields"]
        assert len(instance_fields) == 5
        assert instance_fields[3] == {
            "name": "promo",
            "datatype": "integer",
            "role": "known_covariate",
            "required": True,
        }
        assert instance_fields[4] == {
            "name": "temperature",
            "datatype": "number",
            "role": "known_covariate",
            "required": True,
        }
        assert model_json["inference"]["sample_payload"]["instances"] == [
            {
                "product_id": "<string>",
                "date": "<string>",
                "sales": "<number>",
                "promo": "<integer>",
                "temperature": "<number>",
            }
        ]

        assert "known_covariates" in schema
        cov = schema["known_covariates"]
        assert cov["required"] is True
        cov_fields = cov["fields"]
        assert len(cov_fields) == 4
        assert cov_fields[0] == {"name": "product_id", "datatype": "string", "role": "id", "required": True}
        assert cov_fields[1] == {"name": "date", "datatype": "string", "role": "timestamp", "required": True}
        assert cov_fields[2] == {"name": "promo", "datatype": "integer", "role": "known_covariate", "required": True}
        assert cov_fields[3] == {
            "name": "temperature",
            "datatype": "number",
            "role": "known_covariate",
            "required": True,
        }

        payload = model_json["inference"]["sample_payload"]
        assert "known_covariates" in payload
        assert payload["known_covariates"] == [
            {"product_id": "<string>", "date": "<string>", "promo": "<integer>", "temperature": "<number>"}
        ]

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_inference_block_excludes_synthetic_item_id(
        self, mock_predictor_cls, mock_ts_df_cls, mock_concat, mock_read_csv, mock_artifacts
    ):
        """model.json inference block excludes __synthetic_item_id from schema and payload."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]

        full_train_ts = _mock_ts_df()
        full_train_ts.columns = ["sales"]
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = full_train_ts
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="__synthetic_item_id",
            timestamp_column="date",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            extra_train_data_path=extra_train_path,
            prediction_length=7,
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
            uses_synthetic_id=True,
        )

        model_json_path = Path(models_artifact.path) / "DeepAR_FULL" / "model.json"
        assert model_json_path.exists()
        model_json = json.loads(model_json_path.read_text())

        assert "inference" in model_json
        schema = model_json["inference"]["input_data_schema"]
        fields = schema["instances"]["fields"]
        assert len(fields) == 2
        assert fields[0] == {"name": "date", "datatype": "string", "role": "timestamp", "required": True}
        assert fields[1] == {"name": "sales", "datatype": "number", "role": "target", "required": True}

        field_names = [f["name"] for f in fields]
        assert "__synthetic_item_id" not in field_names

        payload = model_json["inference"]["sample_payload"]
        assert payload == {"instances": [{"date": "<string>", "sales": "<number>"}]}
        assert "__synthetic_item_id" not in payload["instances"][0]

        assert "known_covariates" not in schema
        assert "known_covariates" not in payload

        pred_metadata_path = Path(models_artifact.path) / "DeepAR_FULL" / "predictor" / "predictor_metadata.json"
        pred_metadata = json.loads(pred_metadata_path.read_text())
        assert pred_metadata["uses_synthetic_id"] is True

    @mock.patch("pandas.read_csv")
    @mock.patch("pandas.concat")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_inference_block_skipped_on_metadata_failure(
        self, mock_predictor_cls, mock_ts_df_cls, mock_concat, mock_read_csv, mock_artifacts
    ):
        """model.json omits inference when covariate dtype lookup fails during schema build."""
        models_artifact, extra_train_path, html_artifact = mock_artifacts

        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor.fit_summary.return_value = {"model_hyperparams": {"DeepAR": {}}}
        mock_predictor._trainer.get_model_attribute.return_value = mock.MagicMock

        mock_refit_predictor = mock.MagicMock()
        mock_refit_predictor.evaluate.return_value = {"MASE": 0.5}

        mock_predictor_cls.side_effect = [mock_predictor, mock_refit_predictor]

        full_train_ts = _mock_ts_df()
        full_train_ts.columns = ["sales", "promo"]
        full_train_ts.__getitem__ = mock.MagicMock(side_effect=RuntimeError("dtype lookup failed"))
        mock_ts_df_cls.from_data_frame.return_value = _mock_ts_df()
        mock_ts_df_cls.from_path.return_value = _mock_ts_df()
        mock_ts_df_cls.return_value = full_train_ts
        mock_concat.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        autogluon_timeseries_models_training.python_func(
            target="sales",
            id_column="product_id",
            timestamp_column="date",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            pipeline_name="ts-pipeline-123",
            run_id="run-123",
            models_artifact=models_artifact,
            extra_train_data_path=extra_train_path,
            prediction_length=7,
            known_covariates_names=["promo"],
            html_artifact=html_artifact,
            component_status=_DEFAULT_COMPONENT_STATUS,
        )

        model_json_path = Path(models_artifact.path) / "DeepAR_FULL" / "model.json"
        assert model_json_path.exists()
        model_json = json.loads(model_json_path.read_text())
        assert "inference" not in model_json
        assert model_json["name"] == "DeepAR_FULL"
