"""Unit tests for the autogluon_timeseries_models_selection component."""

import sys
from pathlib import Path
from unittest import mock

import pytest

from ..component import autogluon_timeseries_models_selection


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
        yield


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


class TestTimeseriesModelsSelectionUnitTests:
    """Unit tests for autogluon_timeseries_models_selection behavior."""

    def test_component_function_exists(self):
        """Component exposes KFP python_func."""
        assert callable(autogluon_timeseries_models_selection)
        assert hasattr(autogluon_timeseries_models_selection, "python_func")

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_basic_flow_returns_expected_outputs(self, mock_predictor_cls, mock_ts_df_cls, mock_read_csv):
        """Happy path returns top models, config, and predictor path."""
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR", "TFT", "AutoARIMA"])
        mock_predictor_cls.return_value = mock_predictor

        train_ts, test_ts = _mock_ts_df(), _mock_ts_df()
        mock_ts_df_cls.from_data_frame.side_effect = [train_ts, test_ts]
        train_df, test_df = mock.MagicMock(), mock.MagicMock()
        mock_read_csv.side_effect = [train_df, test_df]

        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        result = autogluon_timeseries_models_selection.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=2,
            workspace_path="/tmp/workspace",
            prediction_length=24,
        )

        mock_read_csv.assert_any_call("/tmp/train.csv")
        mock_read_csv.assert_any_call("/tmp/test.csv")
        mock_ts_df_cls.from_data_frame.assert_any_call(train_df, id_column="item_id", timestamp_column="timestamp")
        mock_ts_df_cls.from_data_frame.assert_any_call(test_df, id_column="item_id", timestamp_column="timestamp")

        mock_predictor_cls.assert_called_once_with(
            prediction_length=24,
            target="sales",
            eval_metric="MASE",
            path=str(Path("/tmp/workspace") / "timeseries_predictor"),
            verbosity=2,
            known_covariates_names=None,
        )
        mock_predictor.fit.assert_called_once_with(
            train_data=train_ts,
            presets="fast_training",
            time_limit=600,
            excluded_model_types=["Chronos", "Toto", "Chronos2"],
        )
        mock_predictor.leaderboard.assert_called_once_with(test_ts)

        assert result.top_models == ["DeepAR", "TFT"]
        assert result.eval_metric_name == "MASE"
        assert result.predictor_path == "/tmp/workspace/timeseries_predictor"
        assert result.model_config["prediction_length"] == 24
        assert result.model_config["presets"] == "fast_training"
        assert result.model_config["time_limit"] == 600
        assert result.model_config["known_covariates_names"] == []
        assert result.model_config["num_models_trained"] == 3

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_known_covariates_propagated_to_predictor_and_model_config(
        self, mock_predictor_cls, mock_ts_df_cls, mock_read_csv
    ):
        """Known covariates are passed to predictor ctor and returned in model_config."""
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.return_value = _mock_leaderboard(["DeepAR"])
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        covariates = ["is_holiday", "promo_flag"]
        result = autogluon_timeseries_models_selection.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train.csv",
            test_data=test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            known_covariates_names=covariates,
        )

        assert mock_predictor_cls.call_args[1]["known_covariates_names"] == covariates
        assert result.model_config["known_covariates_names"] == covariates

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_top_n_greater_than_available_models_raises(self, mock_predictor_cls, mock_ts_df_cls, mock_read_csv):
        """top_n exceeding trained model count raises ValueError."""
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
            autogluon_timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=3,
                workspace_path="/tmp/workspace",
            )

    def test_invalid_top_n_zero_raises(self):
        """top_n must be in range (0, TOP_N_MAX] (see component TOP_N_MAX)."""
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match=r"top_n must be an integer in the range \(0, 7\]; got 0\."):
            autogluon_timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=0,
                workspace_path="/tmp/workspace",
            )

    def test_invalid_top_n_above_max_raises(self):
        """top_n above TOP_N_MAX is rejected before training."""
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match=r"top_n must be an integer in the range \(0, 7\]; got 8\."):
            autogluon_timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=8,
                workspace_path="/tmp/workspace",
            )

    def test_invalid_prediction_length_raises(self):
        """prediction_length must be a positive integer."""
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"
        with pytest.raises(ValueError, match="prediction_length must be greater than 0"):
            autogluon_timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=1,
                workspace_path="/tmp/workspace",
                prediction_length=0,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_training_failure_is_wrapped(self, mock_predictor_cls, mock_ts_df_cls, mock_read_csv):
        """Training errors are wrapped in ValueError with component-specific message."""
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.side_effect = RuntimeError("boom")
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        with pytest.raises(ValueError, match=r"TimeSeriesPredictor training failed: boom"):
            autogluon_timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_leaderboard_failure_is_wrapped(self, mock_predictor_cls, mock_ts_df_cls, mock_read_csv):
        """Leaderboard errors are wrapped in ValueError with component-specific message."""
        mock_predictor = mock.MagicMock()
        mock_predictor.leaderboard.side_effect = RuntimeError("no leaderboard")
        mock_predictor_cls.return_value = mock_predictor
        mock_ts_df_cls.from_data_frame.side_effect = [_mock_ts_df(), _mock_ts_df()]
        mock_read_csv.side_effect = [mock.MagicMock(), mock.MagicMock()]
        test_data = mock.MagicMock()
        test_data.path = "/tmp/test.csv"

        with pytest.raises(ValueError, match=r"Failed to generate leaderboard: no leaderboard"):
            autogluon_timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train.csv",
                test_data=test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
            )
