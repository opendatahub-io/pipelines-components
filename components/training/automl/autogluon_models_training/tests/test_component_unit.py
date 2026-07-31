"""Unit tests for the autogluon_models_training component."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True, scope="module")
def isolated_sys_modules():
    """Patch autogluon and sklearn in sys.modules for this module; restored on teardown.

    Note: pandas is NOT mocked - the component needs real pandas for DataFrame/Series
    manipulation in curve generation.
    """
    import numpy as np

    with mock.patch.dict(sys.modules, clear=False) as mocked_modules:
        _ag = mock.MagicMock()
        _ag.__path__ = []
        _ag.__spec__ = None
        mocked_modules["autogluon"] = _ag
        for sub in (
            "autogluon.tabular",
            "autogluon.tabular.configs",
            "autogluon.tabular.configs.hyperparameter_configs",
            "autogluon.core",
            "autogluon.core.metrics",
        ):
            _m = mock.MagicMock()
            _m.__spec__ = None
            if sub in ("autogluon.tabular", "autogluon.tabular.configs", "autogluon.core"):
                _m.__path__ = []
            if sub == "autogluon.tabular.configs.hyperparameter_configs":
                _m.get_hyperparameter_config = mock.MagicMock(
                    return_value={"RF": [{"max_depth": None}], "XT": [{"max_depth": None}]}
                )
            if sub == "autogluon.core.metrics":
                _m.METRICS = {
                    "binary": {
                        k: mock.MagicMock()
                        for k in (
                            "accuracy",
                            "balanced_accuracy",
                            "f1",
                            "f1_macro",
                            "f1_micro",
                            "f1_weighted",
                            "log_loss",
                            "mcc",
                            "roc_auc",
                            "average_precision",
                            "precision",
                            "recall",
                        )
                    },
                    "multiclass": {
                        k: mock.MagicMock()
                        for k in (
                            "accuracy",
                            "balanced_accuracy",
                            "f1_macro",
                            "f1_micro",
                            "f1_weighted",
                            "log_loss",
                            "mcc",
                            "roc_auc_ovo",
                            "roc_auc_ovr",
                        )
                    },
                    "regression": {
                        k: mock.MagicMock()
                        for k in (
                            "r2",
                            "mean_squared_error",
                            "mse",
                            "root_mean_squared_error",
                            "rmse",
                            "mean_absolute_error",
                            "mae",
                            "median_absolute_error",
                            "mape",
                            "smape",
                            "spearmanr",
                            "pearsonr",
                        )
                    },
                }
            mocked_modules[sub] = _m

        def _roc_auc_score_side_effect(y_true, y_score):
            arr = np.asarray(y_true)
            if len(np.unique(arr)) < 2:
                raise ValueError("Only one class present in y_true. ROC AUC score is not defined.")
            return 0.85

        _sklearn_metrics = mock.MagicMock()
        _sklearn_metrics.roc_curve.return_value = (
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.8, 1.0]),
            np.array([1.0, 0.5]),
        )
        _sklearn_metrics.roc_auc_score.side_effect = _roc_auc_score_side_effect
        _sklearn_metrics.precision_recall_curve.return_value = (
            np.array([1.0, 0.8, 0.0]),
            np.array([0.0, 0.5, 1.0]),
            np.array([0.8, 0.5]),
        )
        _sklearn_metrics.average_precision_score.return_value = 0.75
        mocked_modules["sklearn"] = mock.MagicMock()
        mocked_modules["sklearn.metrics"] = _sklearn_metrics
        yield


from ..component import autogluon_models_training  # noqa: E402

PIPELINE_NAME = "test-pipeline-run-123"


def _dataframes_with_real_pandas(build):
    """Build ``pandas.DataFrame`` values while the module autouse fixture mocks ``sys.modules['pandas']``."""
    saved = sys.modules.pop("pandas", None)
    try:
        import importlib

        pd = importlib.import_module("pandas")
        return build(pd)
    finally:
        if saved is not None:
            sys.modules["pandas"] = saved


@contextmanager
def _real_pandas_sys_modules():
    """Use real ``pandas`` in ``sys.modules`` so ``python_func`` cleansing runs on real ``DataFrame`` objects."""
    saved = sys.modules["pandas"]
    sys.modules.pop("pandas")
    import importlib

    sys.modules["pandas"] = importlib.import_module("pandas")
    try:
        yield
    finally:
        sys.modules["pandas"] = saved


def _mock_csv_frame(label_column: str = "target", feature_cols: tuple[str, ...] = ("feature1",)):
    """Minimal ``read_csv`` mock row so cleansing finds ``label_column`` in ``columns``."""
    cols = list(feature_cols)
    if label_column not in cols:
        cols.append(label_column)
    m = mock.MagicMock()
    m.columns = cols
    m.empty = False
    return m


RUN_ID = "run-456"
SAMPLE_ROW = '[{"feature1": 1, "target": 1.1}]'

_MINIMAL_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "abc123",
            "metadata": {},
            "outputs": [],
            "source": [
                'pipeline_name = "<REPLACE_PIPELINE_NAME>"\n',
                'run_id = "<REPLACE_RUN_ID>"\n',
                'model_name = "<REPLACE_MODEL_NAME>"\n',
                "score_data = <REPLACE_SAMPLE_ROW>\n",
            ],
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _mock_leaderboard_top_models(mock_predictor, names: list):
    """Make leaderboard().head(n)['model'].values.tolist() return ``names``."""
    chain = mock_predictor.leaderboard.return_value.head.return_value
    chain.__getitem__.return_value.values.tolist.return_value = names


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


def _base_call_kwargs(workspace_path, models_artifact, test_data, tmp_path=None):
    """Return minimal valid kwargs for autogluon_models_training.python_func."""
    rs = (
        _make_component_status_artifact(tmp_path)
        if tmp_path is not None
        else mock.MagicMock(path="/tmp/rs", metadata={})
    )
    html = (
        _make_html_artifact(tmp_path)
        if tmp_path is not None
        else mock.MagicMock(path="/tmp/leaderboard.html", metadata={})
    )
    return dict(
        label_column="target",
        task_type="regression",
        top_n=2,
        train_data_path="/tmp/train.csv",
        test_data=test_data,
        workspace_path=workspace_path,
        pipeline_name=PIPELINE_NAME,
        run_id=RUN_ID,
        sample_row=SAMPLE_ROW,
        models_artifact=models_artifact,
        html_artifact=html,
        component_status=rs,
        extra_train_data_path="/tmp/extra.csv",
    )


def _run_training_for_evaluation_mode(mock_predictor_class, mock_read_csv, tmp_path, **extra_kwargs):
    """Drive one successful regression run; ``extra_kwargs`` carries evaluation_mode when set.

    Returns the models and HTML artifacts so callers can assert on the recorded metadata.
    """
    mock_predictor = mock.MagicMock()
    mock_predictor_clone = mock.MagicMock()
    mock_predictor_class.return_value.fit.return_value = mock_predictor
    mock_predictor.clone.return_value = mock_predictor_clone
    mock_predictor.problem_type = "regression"
    mock_predictor.label = "target"
    mock_predictor.eval_metric = "r2"
    _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
    mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
    mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
    mock_predictor_clone.predict.return_value = mock.MagicMock()
    mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

    workspace_path = str(tmp_path / "ws")
    Path(workspace_path).mkdir()
    models_output_dir = str(tmp_path / "out")
    Path(models_output_dir).mkdir()
    models_artifact = mock.MagicMock()
    models_artifact.path = models_output_dir
    models_artifact.metadata = {}
    html_artifact = _make_html_artifact(tmp_path)

    autogluon_models_training.python_func(
        label_column="target",
        task_type="regression",
        top_n=1,
        train_data_path="/tmp/train.csv",
        test_data=mock.MagicMock(path="/tmp/test.csv"),
        workspace_path=workspace_path,
        pipeline_name=PIPELINE_NAME,
        run_id=RUN_ID,
        sample_row=SAMPLE_ROW,
        models_artifact=models_artifact,
        html_artifact=html_artifact,
        component_status=_make_component_status_artifact(tmp_path),
        **extra_kwargs,
    )
    return models_artifact, html_artifact


_NOTEBOOK_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "shared" / "notebook_templates"
_DEFAULT_COMPONENT_STATUS = _make_component_status_artifact(Path("/tmp"))
_DEFAULT_HTML_ARTIFACT = _make_html_artifact(Path("/tmp"))


class TestAutogluonModelsTrainingUnitTests:
    """Unit tests for the autogluon_models_training component."""

    def test_regression_notebook_template_excludes_curves(self):
        """Regression template must not reference curves.json (classification-only artifact)."""
        text = (_NOTEBOOK_TEMPLATES_DIR / "regression_notebook.ipynb").read_text(encoding="utf-8")
        assert "curves.json" not in text
        assert "roc-and-precision-recall-curves" not in text
        assert "confusion_matrix.json" not in text

    def test_classification_notebook_template_includes_curves(self):
        """Classification template documents and loads curves.json for binary/multiclass."""
        text = (_NOTEBOOK_TEMPLATES_DIR / "classification_notebook.ipynb").read_text(encoding="utf-8")
        assert "curves.json" in text
        assert "roc-and-precision-recall-curves" in text
        assert 'curves[\\"task_type\\"] == \\"binary\\"' in text
        assert "confusion_matrix.json" in text

    def test_component_imports_correctly(self):
        """Component is callable and has the expected KFP attributes."""
        assert callable(autogluon_models_training)
        assert hasattr(autogluon_models_training, "python_func")
        assert hasattr(autogluon_models_training, "component_spec")

    # ── Happy path ─────────────────────────────────────────────────────────────

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_regression_happy_path(self, mock_predictor_class, mock_read_csv, tmp_path):
        """Full regression flow: fit, select top 2, refit_full batch, per-model artifacts."""
        top_models = ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"]
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, top_models)
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"feature1": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()

        mock_train_df, mock_test_df, mock_extra_df = _mock_csv_frame(), _mock_csv_frame(), _mock_csv_frame()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df, mock_extra_df]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}
        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test.csv"

        result = autogluon_models_training.python_func(
            **_base_call_kwargs(workspace_path, mock_models_artifact, mock_test_data, tmp_path),
            sampling_config={"sample": True},
            split_config={"split": 0.8},
        )

        # Return value
        assert result.eval_metric == "r2"
        assert isinstance(result.best_model_name, str)
        assert result.best_model_name in ("LightGBM_BAG_L1_FULL", "NeuralNetFastAI_BAG_L1_FULL")

        # TabularPredictor constructed and fitted with correct params
        mock_predictor_class.assert_called_once_with(
            problem_type="regression",
            label="target",
            eval_metric="r2",
            path=Path(workspace_path) / "autogluon_predictor",
            verbosity=2,
        )
        fit_call = mock_predictor_class.return_value.fit.call_args
        assert fit_call[1]["train_data"] is mock_train_df
        assert fit_call[1]["presets"] == "good_quality"
        assert fit_call[1]["time_limit"] == 45 * 60
        assert fit_call[1]["refit_full"] is False
        assert fit_call[1]["set_best_to_refit_full"] is False
        assert fit_call[1]["save_bag_folds"] is True
        assert "hyperparameters" not in fit_call[1]
        assert fit_call[1]["excluded_model_types"] == ["CAT"]

        # read_csv: train, test, extra
        assert mock_read_csv.call_count == 3
        assert mock_read_csv.call_args_list[0][0][0] == "/tmp/train.csv"
        assert mock_read_csv.call_args_list[1][0][0] == "/tmp/test.csv"
        assert mock_read_csv.call_args_list[2][0][0] == "/tmp/extra.csv"

        # leaderboard called with test df
        mock_predictor.leaderboard.assert_called_once_with(mock_test_df)

        # clone called ONCE (not per model), with PVC work path
        mock_predictor.clone.assert_called_once()
        work_path = Path(workspace_path) / "refit_work"
        assert mock_predictor.clone.call_args[1]["path"] == work_path
        assert mock_predictor.clone.call_args[1]["return_clone"] is True
        assert mock_predictor.clone.call_args[1]["dirs_exist_ok"] is True

        # refit_full called ONCE with full list (batch, not per-model)
        mock_predictor_clone.refit_full.assert_called_once_with(model=top_models, train_data_extra=mock_extra_df)

        # predict called per model with explicit model= arg
        assert mock_predictor_clone.predict.call_count == 2
        mock_predictor_clone.predict.assert_any_call(mock_test_df, model="LightGBM_BAG_L1_FULL")
        mock_predictor_clone.predict.assert_any_call(mock_test_df, model="NeuralNetFastAI_BAG_L1_FULL")

        # evaluate_predictions called per model (not evaluate())
        assert mock_predictor_clone.evaluate_predictions.call_count == 2

        # feature_importance called per model with model= and subsample_size=2000
        assert mock_predictor_clone.feature_importance.call_count == 2
        mock_predictor_clone.feature_importance.assert_any_call(
            mock_test_df, model="LightGBM_BAG_L1_FULL", subsample_size=2000
        )
        mock_predictor_clone.feature_importance.assert_any_call(
            mock_test_df, model="NeuralNetFastAI_BAG_L1_FULL", subsample_size=2000
        )

        # set_model_best called per model before clone_for_deployment
        assert mock_predictor_clone.set_model_best.call_count == 2
        mock_predictor_clone.set_model_best.assert_any_call(model="LightGBM_BAG_L1_FULL", save_trainer=True)
        mock_predictor_clone.set_model_best.assert_any_call(model="NeuralNetFastAI_BAG_L1_FULL", save_trainer=True)
        assert mock_predictor_clone.clone_for_deployment.call_count == 2

        # metadata["model_names"] serialized as JSON string
        assert json.loads(mock_models_artifact.metadata["model_names"]) == [
            "LightGBM_BAG_L1_FULL",
            "NeuralNetFastAI_BAG_L1_FULL",
        ]

        # Artifacts written on disk for each model
        for model_name_full in ("LightGBM_BAG_L1_FULL", "NeuralNetFastAI_BAG_L1_FULL"):
            model_dir = Path(models_output_dir) / model_name_full
            metrics_dir = model_dir / "metrics"
            assert (metrics_dir / "metrics.json").exists()
            assert (metrics_dir / "feature_importance.json").exists()
            assert not (metrics_dir / "confusion_matrix.json").exists()  # regression: no CM
            assert not (metrics_dir / "curves.json").exists()
            # model.json written alongside metrics/, predictor/, notebooks/
            model_json_path = model_dir / "model.json"
            assert model_json_path.exists()
            model_meta = json.loads(model_json_path.read_text())
            assert model_meta["name"] == model_name_full
            assert model_meta["location"]["model_directory"] == model_name_full
            assert "predictor" in model_meta["location"]
            assert "notebook" in model_meta["location"]
            assert "metrics" in model_meta["location"]
            assert "test_data" in model_meta["metrics"]
            nb_path = Path(models_output_dir) / model_name_full / "notebooks" / "automl_predictor_notebook.ipynb"
            assert nb_path.exists()
            nb_text = nb_path.read_text()
            for placeholder in (
                "<REPLACE_PIPELINE_NAME>",
                "<REPLACE_RUN_ID>",
                "<REPLACE_MODEL_NAME>",
                "<REPLACE_SAMPLE_ROW>",
            ):  # noqa: E501
                assert placeholder not in nb_text
            # pipeline name trimmed (last segment "-123" removed), raw name absent
            assert "test-pipeline-run" in nb_text
            assert PIPELINE_NAME not in nb_text
            # label column stripped from sample row
            assert "feature1" in nb_text
            assert "'target'" not in nb_text

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_speed_preset_fit_args(self, mock_predictor_class, mock_read_csv, tmp_path):
        """Speed preset uses a 45-minute time limit and good_quality AutoGluon preset."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()

        mock_train_df, mock_test_df = _mock_csv_frame(), _mock_csv_frame()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            html_artifact=_make_html_artifact(tmp_path),
            preset="speed",
            component_status=_make_component_status_artifact(tmp_path),
        )

        fit_call = mock_predictor_class.return_value.fit.call_args
        assert fit_call[1]["presets"] == "good_quality"  # AG internal name
        assert fit_call[1]["time_limit"] == 45 * 60
        assert fit_call[1]["refit_full"] is False
        assert fit_call[1]["set_best_to_refit_full"] is False
        assert fit_call[1]["save_bag_folds"] is True
        assert "hyperparameters" not in fit_call[1]
        assert fit_call[1]["excluded_model_types"] == ["CAT"]

        context = mock_models_artifact.metadata["context"]
        assert context["model_config"]["preset"] == "speed"
        assert context["model_config"]["time_limit"] == 45 * 60

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_balanced_preset_fit_args(self, mock_predictor_class, mock_read_csv, tmp_path):
        """Balanced preset uses 90-minute time limit and high_quality AutoGluon preset."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()

        mock_train_df, mock_test_df = _mock_csv_frame(), _mock_csv_frame()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            html_artifact=_make_html_artifact(tmp_path),
            preset="balanced",
            component_status=_make_component_status_artifact(tmp_path),
        )

        fit_call = mock_predictor_class.return_value.fit.call_args
        assert fit_call[1]["presets"] == "high_quality"
        assert fit_call[1]["time_limit"] == 90 * 60
        assert fit_call[1]["excluded_model_types"] == ["CAT"]

        context = mock_models_artifact.metadata["context"]
        assert context["model_config"]["preset"] == "balanced"
        assert context["model_config"]["time_limit"] == 90 * 60

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_without_extra_train_data(self, mock_predictor_class, mock_read_csv, tmp_path):
        """Empty extra_train_data_path passes train_data_extra=None to refit_full."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()

        mock_train_df, mock_test_df = _mock_csv_frame(), _mock_csv_frame()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            html_artifact=_make_html_artifact(tmp_path),
            extra_train_data_path="",
            component_status=_make_component_status_artifact(tmp_path),
        )

        # refit_full gets None for extra data
        mock_predictor_clone.refit_full.assert_called_once_with(model=["LightGBM_BAG_L1"], train_data_extra=None)
        # read_csv called only twice (train + test, no extra)
        assert mock_read_csv.call_count == 2

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_binary_explicit_positive_class_passed_to_tabular_predictor(
        self,
        mock_predictor_class,
        mock_read_csv,
        tmp_path,
    ):
        """Explicit positive_class is forwarded to TabularPredictor for binary tasks."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "binary"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "accuracy"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.positive_class = 0
        mock_predictor_clone.evaluate_predictions.return_value = {
            "accuracy": 0.9,
            "confusion_matrix": mock.MagicMock(to_dict=lambda: {"0": {"0": 1}}),
            "classification_report": {},
        }

        mock_test_df = _mock_csv_frame()
        saved = sys.modules.pop("pandas", None)
        try:
            import importlib

            pd = importlib.import_module("pandas")
            y_true = pd.Series([0, 0, 1, 1])
            y_proba = pd.DataFrame({0: [0.9, 0.8, 0.2, 0.1], 1: [0.1, 0.2, 0.8, 0.9]})
        finally:
            if saved is not None:
                sys.modules["pandas"] = saved
        mock_test_df.__getitem__ = lambda self, key: y_true if key == "target" else mock.MagicMock()
        mock_predictor_clone.predict_proba.return_value = y_proba

        mock_read_csv.side_effect = [_mock_csv_frame(), mock_test_df, _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        call_kwargs = _base_call_kwargs(
            workspace_path, mock_models_artifact, mock.MagicMock(path="/tmp/test.csv"), tmp_path
        )
        call_kwargs["task_type"] = "binary"
        call_kwargs["positive_class"] = "0"
        autogluon_models_training.python_func(**call_kwargs)

        mock_predictor_class.assert_called_once_with(
            problem_type="binary",
            label="target",
            eval_metric="accuracy",
            path=Path(workspace_path) / "autogluon_predictor",
            verbosity=2,
            positive_class=0,
        )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_regression_positive_class_is_ignored(
        self,
        mock_predictor_class,
        mock_read_csv,
        caplog,
        tmp_path,
    ):
        """positive_class on regression runs is not passed to TabularPredictor."""
        import logging

        caplog.set_level(logging.WARNING)
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()

        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            **_base_call_kwargs(workspace_path, mock_models_artifact, mock.MagicMock(path="/tmp/test.csv"), tmp_path),
            positive_class="yes",
        )

        mock_predictor_class.assert_called_once_with(
            problem_type="regression",
            label="target",
            eval_metric="r2",
            path=Path(workspace_path) / "autogluon_predictor",
            verbosity=2,
        )
        assert "ignored when task_type='regression'" in caplog.text

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_binary_classification_writes_confusion_matrix_and_curves(
        self,
        mock_predictor_class,
        mock_read_csv,
        tmp_path,
    ):
        """Binary classification uses predict_proba + detailed_report evaluate_predictions."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "binary"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "accuracy"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.positive_class = 1

        def _dataframes_with_real_pandas_for_curves():
            saved = sys.modules.pop("pandas", None)
            try:
                import importlib

                pd = importlib.import_module("pandas")
                y_true = pd.Series([0, 0, 1, 1])
                y_proba = pd.DataFrame({0: [0.9, 0.8, 0.2, 0.1], 1: [0.1, 0.2, 0.8, 0.9]})
                return y_true, y_proba
            finally:
                if saved is not None:
                    sys.modules["pandas"] = saved

        mock_test_df = _mock_csv_frame()
        y_true, y_proba = _dataframes_with_real_pandas_for_curves()
        mock_test_df.__getitem__ = lambda self, key: y_true if key == "target" else mock.MagicMock()
        mock_predictor_clone.predict_proba.return_value = y_proba

        confusion_matrix_dict = {"0": {"0": 5, "1": 0}, "1": {"0": 0, "1": 3}}
        mock_predictor_clone.evaluate_predictions.return_value = {
            "accuracy": 0.95,
            "roc_auc": 0.99,
            "confusion_matrix": mock.MagicMock(to_dict=lambda: confusion_matrix_dict),
            "classification_report": {"0": {"precision": 1.0}},
        }

        mock_read_csv.side_effect = [_mock_csv_frame(), mock_test_df]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="binary",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        mock_predictor_clone.predict.assert_not_called()
        mock_predictor_clone.predict_proba.assert_called_once()
        proba_call = mock_predictor_clone.predict_proba.call_args
        assert proba_call[1]["model"] == "LightGBM_BAG_L1_FULL"

        mock_predictor_clone.evaluate_predictions.assert_called_once()
        eval_call = mock_predictor_clone.evaluate_predictions.call_args[1]
        assert eval_call["y_pred"] is y_proba
        assert eval_call["detailed_report"] is True

        metrics_path = Path(models_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics" / "metrics.json"
        metrics_payload = json.loads(metrics_path.read_text())
        assert metrics_payload == {"accuracy": 0.95, "roc_auc": 0.99}
        assert "confusion_matrix" not in metrics_payload

        cm_path = Path(models_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics" / "confusion_matrix.json"
        assert cm_path.exists()
        assert json.loads(cm_path.read_text()) == confusion_matrix_dict

        curves_path = Path(models_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics" / "curves.json"
        assert curves_path.exists()
        curves_payload = json.loads(curves_path.read_text())
        assert curves_payload["task_type"] == "binary"
        assert "roc_curve" in curves_payload
        assert "precision_recall_curve" in curves_payload

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_multiclass_writes_curves_json(
        self,
        mock_predictor_class,
        mock_read_csv,
        tmp_path,
    ):
        """Multiclass classification writes curves.json with multiclass task_type."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "multiclass"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "accuracy"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})

        def _multiclass_proba():
            saved = sys.modules.pop("pandas", None)
            try:
                import importlib

                pd = importlib.import_module("pandas")
                return pd.DataFrame(
                    {
                        0: [0.8, 0.1, 0.1],
                        1: [0.1, 0.8, 0.1],
                        2: [0.1, 0.1, 0.8],
                    }
                )
            finally:
                if saved is not None:
                    sys.modules["pandas"] = saved

        mock_test_df = _mock_csv_frame()
        y_true = _dataframes_with_real_pandas(lambda pd: pd.Series([0, 1, 2]))
        mock_test_df.__getitem__ = lambda self, key: y_true if key == "target" else mock.MagicMock()
        y_proba = _multiclass_proba()
        mock_predictor_clone.predict_proba.return_value = y_proba
        mock_predictor_clone.evaluate_predictions.return_value = {
            "accuracy": 0.8,
            "confusion_matrix": mock.MagicMock(to_dict=lambda: {"0": {"0": 1}}),
        }

        mock_read_csv.side_effect = [_mock_csv_frame(), mock_test_df]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="multiclass",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        mock_predictor_clone.predict.assert_not_called()
        eval_call = mock_predictor_clone.evaluate_predictions.call_args[1]
        assert eval_call["y_pred"] is y_proba
        assert eval_call["detailed_report"] is True

        curves_path = Path(models_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics" / "curves.json"
        assert curves_path.exists()
        curves_payload = json.loads(curves_path.read_text())
        assert curves_payload["task_type"] == "multiclass"
        assert curves_payload["strategy"] == "ovr"

        cm_path = Path(models_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics" / "confusion_matrix.json"
        assert cm_path.exists()

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_multiclass_curves_skips_class_with_no_test_examples(
        self,
        mock_predictor_class,
        mock_read_csv,
        tmp_path,
    ):
        """OvR curves skip classes absent from test labels but still write curves for others."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "multiclass"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "accuracy"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})

        y_true, y_proba = _dataframes_with_real_pandas(
            lambda pd: (
                pd.Series([1, 1, 2, 2]),
                pd.DataFrame(
                    {
                        0: [0.1, 0.1, 0.1, 0.1],
                        1: [0.7, 0.8, 0.1, 0.2],
                        2: [0.2, 0.1, 0.8, 0.7],
                    }
                ),
            )
        )
        mock_test_df = _mock_csv_frame()
        mock_test_df.__getitem__ = lambda self, key: y_true if key == "target" else mock.MagicMock()
        mock_predictor_clone.predict_proba.return_value = y_proba
        mock_predictor_clone.evaluate_predictions.return_value = {
            "accuracy": 0.75,
            "confusion_matrix": mock.MagicMock(to_dict=lambda: {"1": {"1": 2}}),
        }
        mock_read_csv.side_effect = [_mock_csv_frame(), mock_test_df]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="multiclass",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        curves_path = Path(models_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics" / "curves.json"
        assert curves_path.exists()
        curves_payload = json.loads(curves_path.read_text())
        assert curves_payload["skipped_classes"] == ["0"]
        assert set(curves_payload["roc_curve"]["per_class"]) == {"1", "2"}
        assert set(curves_payload["precision_recall_curve"]["per_class"]) == {"1", "2"}

    # ── Call order and structural invariants ───────────────────────────────────

    @mock.patch("shutil.rmtree")
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_operations_called_in_correct_order(self, mock_predictor_class, mock_read_csv, mock_rmtree, tmp_path):
        """Verify call order for a single model: fit → clone → refit_full → Phase A (predict → evaluate → fi) → Phase B (set_model_best → clone_for_deployment) → rmtree.

        Phase A (metrics + notebook) runs via ThreadPoolExecutor across models, but within
        a single model's _process_model the calls are always sequential: predict first, then
        evaluate_predictions, then feature_importance.  Phase B always follows Phase A.
        """  # noqa: E501
        call_order = []
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()

        mock_predictor_class.return_value.fit.side_effect = lambda **kw: (call_order.append("fit"), mock_predictor)[1]
        mock_predictor.clone.side_effect = lambda **kw: (call_order.append("clone"), mock_predictor_clone)[1]
        mock_predictor_clone.refit_full.side_effect = lambda **kw: call_order.append("refit_full")
        mock_predictor_clone.predict.side_effect = lambda df, model: (
            call_order.append("predict"),
            mock.MagicMock(),
        )[1]
        mock_predictor_clone.evaluate_predictions.side_effect = lambda **kw: (
            call_order.append("evaluate_predictions"),
            {"r2": 0.9},
        )[1]
        mock_predictor_clone.feature_importance.side_effect = lambda df, model, subsample_size: (
            call_order.append("feature_importance"),
            mock.MagicMock(to_dict=lambda: {"f": 0.1}),
        )[1]
        mock_predictor_clone.set_model_best.side_effect = lambda **kw: call_order.append("set_model_best")
        mock_predictor_clone.clone_for_deployment.side_effect = lambda **kw: call_order.append("clone_for_deployment")
        mock_rmtree.side_effect = lambda path, **kw: call_order.append("rmtree")

        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        # Global ordering invariants (single model, so no inter-model concurrency to worry about)
        assert call_order[0] == "fit"
        assert call_order[1] == "clone"
        assert call_order[2] == "refit_full"
        # Phase A: within _process_model calls are sequential
        assert call_order[3] == "predict"
        assert call_order[4] == "evaluate_predictions"
        assert call_order[5] == "feature_importance"
        # Phase B: always after all Phase A work completes
        assert call_order[6] == "set_model_best"
        assert call_order[7] == "clone_for_deployment"
        assert call_order[-1] == "rmtree"

    @mock.patch("shutil.rmtree")
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_work_path_is_on_pvc_and_cleaned_up(self, mock_predictor_class, mock_read_csv, mock_rmtree, tmp_path):
        """Clone work path is inside workspace_path (PVC), not inside models_artifact (S3)."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        expected_work_path = Path(workspace_path) / "refit_work"
        clone_path = mock_predictor.clone.call_args[1]["path"]
        # Must be inside workspace (PVC), not inside models_artifact path (S3)
        assert clone_path == expected_work_path
        assert not str(clone_path).startswith(models_output_dir)
        # Work dir cleaned up after all models are saved
        mock_rmtree.assert_called_once_with(expected_work_path, ignore_errors=True)

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_refit_full_called_once_with_all_top_models(self, mock_predictor_class, mock_read_csv, tmp_path):
        """refit_full is called exactly once with the full list of top models (batch, not per-model)."""
        top_models = ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1", "XGBoost_BAG_L1"]
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, top_models)
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=3,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        mock_predictor_clone.refit_full.assert_called_once_with(model=top_models, train_data_extra=None)
        # clone also called exactly once (not per model)
        mock_predictor.clone.assert_called_once()

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_context_models_metadata(self, mock_predictor_class, mock_read_csv, tmp_path):
        """context['models'] contains one entry per model with correct name, location, and metrics."""
        top_models = ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"]
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, top_models)
        # Use distinct prediction objects per model so evaluate_predictions can return
        # the correct metrics regardless of concurrent execution order.
        lgbm_preds = mock.MagicMock(name="lgbm_preds")
        cat_preds = mock.MagicMock(name="cat_preds")
        _metrics_by_pred = {
            id(lgbm_preds): {"r2": 0.9, "root_mean_squared_error": 0.31},
            id(cat_preds): {"r2": 0.85, "root_mean_squared_error": 0.42},
        }
        mock_predictor_clone.predict.side_effect = lambda df, model: lgbm_preds if "LightGBM" in model else cat_preds
        mock_predictor_clone.evaluate_predictions.side_effect = lambda y_true, y_pred: _metrics_by_pred[id(y_pred)]
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=2,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        context = mock_models_artifact.metadata["context"]
        models = context["models"]

        assert len(models) == 2

        lgbm = models[0]
        assert lgbm["name"] == "LightGBM_BAG_L1_FULL"
        assert lgbm["location"]["model_directory"] == "LightGBM_BAG_L1_FULL"
        assert lgbm["location"]["predictor"] == str(Path("LightGBM_BAG_L1_FULL") / "predictor")
        assert lgbm["location"]["notebook"] == str(
            Path("LightGBM_BAG_L1_FULL") / "notebooks" / "automl_predictor_notebook.ipynb"
        )
        assert lgbm["location"]["metrics"] == str(Path("LightGBM_BAG_L1_FULL") / "metrics")
        assert lgbm["metrics"]["test_data"] == {"r2": 0.9, "root_mean_squared_error": 0.31}

        cat = models[1]
        assert cat["name"] == "NeuralNetFastAI_BAG_L1_FULL"
        assert cat["location"]["model_directory"] == "NeuralNetFastAI_BAG_L1_FULL"
        assert cat["location"]["predictor"] == str(Path("NeuralNetFastAI_BAG_L1_FULL") / "predictor")
        assert cat["location"]["notebook"] == str(
            Path("NeuralNetFastAI_BAG_L1_FULL") / "notebooks" / "automl_predictor_notebook.ipynb"
        )
        assert cat["location"]["metrics"] == str(Path("NeuralNetFastAI_BAG_L1_FULL") / "metrics")
        assert cat["metrics"]["test_data"] == {"r2": 0.85, "root_mean_squared_error": 0.42}

        # Shared context fields still present alongside models
        assert context["task_type"] == "regression"
        assert context["label_column"] == "target"
        assert "model_config" in context
        assert "data_config" in context

        # model.json on disk matches the corresponding entry in context["models"]
        for model_entry in models:
            model_json_path = Path(models_output_dir) / model_entry["name"] / "model.json"
            assert model_json_path.exists()
            on_disk = json.loads(model_json_path.read_text())
            assert on_disk["name"] == model_entry["name"]
            assert on_disk["location"] == model_entry["location"]
            assert on_disk["metrics"] == model_entry["metrics"]

    # ── Propagated errors ──────────────────────────────────────────────────────

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_raises_on_invalid_problem_type(self, mock_predictor_class, mock_read_csv, tmp_path):
        """ValueError raised when AutoGluon resolves problem_type to an unsupported value."""
        mock_predictor = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock.MagicMock()
        mock_predictor.problem_type = "quantile"  # unsupported in notebook dispatch
        mock_predictor.label = "target"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = str(tmp_path / "out")
        mock_models_artifact.metadata = {}
        Path(mock_models_artifact.path).mkdir()

        with pytest.raises(ValueError, match="Invalid problem type: quantile"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path=workspace_path,
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=mock_models_artifact,
                html_artifact=_make_html_artifact(tmp_path),
                component_status=_make_component_status_artifact(tmp_path),
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_raises_on_refit_failure(self, mock_predictor_class, mock_read_csv, tmp_path):
        """ValueError from refit_full propagates to the caller."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor_clone.refit_full.side_effect = ValueError("model not found")
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = str(tmp_path / "out")
        mock_models_artifact.metadata = {}
        Path(mock_models_artifact.path).mkdir()

        with pytest.raises(ValueError, match="model not found"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path=workspace_path,
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=mock_models_artifact,
                html_artifact=_make_html_artifact(tmp_path),
                component_status=_make_component_status_artifact(tmp_path),
            )

    # ── Input validation ───────────────────────────────────────────────────────

    def _minimal_artifact(self):
        """Return a minimal mock models artifact path/metadata."""
        a = mock.MagicMock()
        a.path = "/tmp/out"
        a.metadata = {}
        return a

    def test_rejects_empty_label_column(self):
        """Reject blank ``label_column``."""
        with pytest.raises(TypeError, match="label_column must be a non-empty string"):
            autogluon_models_training.python_func(
                label_column="  ",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_invalid_task_type(self):
        """Reject unknown ``task_type``."""
        with pytest.raises(ValueError, match="task_type must be one of"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="unsupported",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_empty_train_data_path(self):
        """Reject empty ``train_data_path``."""
        with pytest.raises(TypeError, match="train_data_path must be a non-empty string"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_empty_workspace_path(self):
        """Reject empty ``workspace_path``."""
        with pytest.raises(TypeError, match="workspace_path must be a non-empty string"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_top_n_zero(self):
        """Reject ``top_n`` of zero."""
        with pytest.raises(ValueError, match="top_n must be an integer in the range"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=0,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_top_n_exceeds_max(self):
        """Reject ``top_n`` above the allowed maximum."""
        with pytest.raises(ValueError, match="top_n must be an integer in the range"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=11,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_empty_pipeline_name(self):
        """Reject empty ``pipeline_name``."""
        with pytest.raises(TypeError, match="pipeline_name must be a non-empty string"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name="",
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_empty_run_id(self):
        """Reject blank ``run_id``."""
        with pytest.raises(TypeError, match="run_id must be a non-empty string"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id="  ",
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_invalid_sample_row_json(self):
        """Reject ``sample_row`` that is not valid JSON."""
        with pytest.raises(TypeError, match="sample_row must be valid JSON array"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row="not valid json{{{",
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_sample_row_not_list(self):
        """Reject ``sample_row`` JSON that is not a list."""
        with pytest.raises(ValueError, match="sample_row must be a JSON array"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row='{"key": "value"}',
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_invalid_sampling_config_type(self):
        """Reject non-dict ``sampling_config``."""
        with pytest.raises(TypeError, match="sampling_config must be a dictionary or None"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                sampling_config="invalid",
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_invalid_split_config_type(self):
        """Reject non-dict ``split_config``."""
        with pytest.raises(TypeError, match="split_config must be a dictionary or None"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                split_config=[],
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_invalid_preset(self):
        """Reject unknown preset value."""
        with pytest.raises(ValueError, match="preset must be one of"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                preset="best_quality",
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_whitespace_eval_metric(self):
        """Whitespace-only eval_metric must raise TypeError, not be forwarded to AutoGluon."""
        with pytest.raises(TypeError, match="eval_metric must be a non-empty string"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                eval_metric="   ",
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_invalid_eval_metric_for_task_type_raises(self):
        """eval_metric unknown for the given task_type raises ValueError before training."""
        with pytest.raises(ValueError, match="is not valid for task_type="):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                eval_metric="accuracy",  # valid for binary/multiclass, not regression,
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    def test_rejects_invalid_evaluation_mode(self):
        """Reject unknown evaluation_mode value."""
        with pytest.raises(ValueError, match="evaluation_mode must be one of"):
            autogluon_models_training.python_func(
                label_column="target",
                task_type="regression",
                top_n=1,
                train_data_path="/tmp/train.csv",
                test_data=mock.MagicMock(path="/tmp/test.csv"),
                workspace_path="/tmp/ws",
                pipeline_name=PIPELINE_NAME,
                run_id=RUN_ID,
                sample_row=SAMPLE_ROW,
                models_artifact=self._minimal_artifact(),
                html_artifact=_DEFAULT_HTML_ARTIFACT,
                evaluation_mode="invalid-mode",
                component_status=_DEFAULT_COMPONENT_STATUS,
            )

    # ── eval_metric parameter ─────────────────────────────────────────────────

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_eval_metric_explicit_forwarded_to_predictor(self, mock_predictor_class, mock_read_csv, tmp_path):
        """Explicit eval_metric is passed through to TabularPredictor constructor and returned."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        result = autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            html_artifact=_make_html_artifact(tmp_path),
            eval_metric="r2",
            component_status=_make_component_status_artifact(tmp_path),
        )

        mock_predictor_class.assert_called_once_with(
            problem_type="regression",
            label="target",
            eval_metric="r2",
            path=Path(workspace_path) / "autogluon_predictor",
            verbosity=2,
        )
        assert result.eval_metric == "r2"

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_eval_metric_none_regression_resolves_to_r2(self, mock_predictor_class, mock_read_csv, tmp_path):
        """eval_metric=None with regression resolves to 'r2'."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        result = autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            html_artifact=_make_html_artifact(tmp_path),
            eval_metric=None,
            component_status=_make_component_status_artifact(tmp_path),
        )

        ctor_kwargs = mock_predictor_class.call_args[1]
        assert ctor_kwargs["eval_metric"] == "r2"
        assert result.eval_metric == "r2"

    @mock.patch("autogluon.core.metrics.confusion_matrix")
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_eval_metric_none_binary_resolves_to_accuracy(
        self, mock_predictor_class, mock_read_csv, mock_confusion_matrix, tmp_path
    ):
        """eval_metric=None with binary resolves to 'accuracy'."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "binary"
        mock_predictor.label = "target"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"accuracy": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()
        mock_confusion_matrix.return_value = mock.MagicMock(to_dict=lambda: {"0": {"0": 5}})
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        result = autogluon_models_training.python_func(
            label_column="target",
            task_type="binary",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        assert mock_predictor_class.call_args[1]["eval_metric"] == "accuracy"
        assert result.eval_metric == "accuracy"

    @mock.patch("autogluon.core.metrics.confusion_matrix")
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_eval_metric_none_multiclass_resolves_to_accuracy(
        self, mock_predictor_class, mock_read_csv, mock_confusion_matrix, tmp_path
    ):
        """eval_metric=None with multiclass resolves to 'accuracy'."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "multiclass"
        mock_predictor.label = "target"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"accuracy": 0.88}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()
        mock_confusion_matrix.return_value = mock.MagicMock(to_dict=lambda: {"0": {"0": 5}})
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        result = autogluon_models_training.python_func(
            label_column="target",
            task_type="multiclass",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            component_status=_make_component_status_artifact(tmp_path),
            html_artifact=_make_html_artifact(tmp_path),
        )

        assert mock_predictor_class.call_args[1]["eval_metric"] == "accuracy"
        assert result.eval_metric == "accuracy"

    # ── Leaderboard phase ──────────────────────────────────────────────────────

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_leaderboard_html_written_and_best_model_returned(self, mock_predictor_class, mock_read_csv, tmp_path):
        """After Phase A/B the leaderboard HTML is written and best_model_name is returned."""
        top_models = ["LightGBM_BAG_L1", "NeuralNetFastAI_BAG_L1"]
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, top_models)
        mock_predictor_clone.evaluate_predictions.side_effect = [
            {"r2": 0.9, "root_mean_squared_error": -0.31},
            {"r2": 0.8, "root_mean_squared_error": -0.42},
        ]
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()

        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.uri = "s3://bucket/run123/models"
        mock_models_artifact.metadata = {}
        html_artifact = _make_html_artifact(tmp_path)

        result = autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=2,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            html_artifact=html_artifact,
            component_status=_make_component_status_artifact(tmp_path),
        )

        # HTML file was written
        assert Path(html_artifact.path).exists()
        html_text = Path(html_artifact.path).read_text(encoding="utf-8")
        assert "r2" in html_text
        assert "LightGBM_BAG_L1_FULL" in html_text

        # best_model_name is the top-ranked model (highest r2)
        assert result.best_model_name == "LightGBM_BAG_L1_FULL"

        # context["best_model_name"] matches the return value
        assert mock_models_artifact.metadata["context"]["best_model_name"] == result.best_model_name

        # metadata["data"] must be a JSON string with string keys (MLMD Struct requirement)
        data_raw = html_artifact.metadata["data"]
        assert isinstance(data_raw, str), "html_artifact.metadata['data'] must be a JSON string"
        data_parsed = json.loads(data_raw)
        assert isinstance(data_parsed, list), "parsed data must be a list of records"
        assert all(isinstance(k, str) for record in data_parsed for k in record), "all record keys must be strings"

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_leaderboard_best_model_name_in_context(self, mock_predictor_class, mock_read_csv, tmp_path):
        """best_model_name is stored in models_artifact context metadata."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.95}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()
        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.uri = "s3://bucket/run/models"
        mock_models_artifact.metadata = {}

        result = autogluon_models_training.python_func(
            label_column="target",
            task_type="regression",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            html_artifact=_make_html_artifact(tmp_path),
            component_status=_make_component_status_artifact(tmp_path),
        )

        assert result.best_model_name == "LightGBM_BAG_L1_FULL"
        context = mock_models_artifact.metadata["context"]
        assert "best_model_name" in context
        assert context["best_model_name"] == "LightGBM_BAG_L1_FULL"

    # ── evaluation_mode parameter ──────────────────────────────────────────────

    @pytest.mark.parametrize(
        ("extra_kwargs", "expected"),
        [
            ({"evaluation_mode": "user-provided"}, "user-provided"),
            ({"evaluation_mode": "auto-split"}, "auto-split"),
            pytest.param({}, "auto-split", id="omitted-defaults-to-auto-split"),
        ],
    )
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_evaluation_mode_stored_in_metadata(
        self, mock_predictor_class, mock_read_csv, tmp_path, extra_kwargs, expected
    ):
        """Each accepted evaluation_mode (and the default) is stored in both metadata locations."""
        models_artifact, html_artifact = _run_training_for_evaluation_mode(
            mock_predictor_class, mock_read_csv, tmp_path, **extra_kwargs
        )

        assert models_artifact.metadata["context"]["evaluation_mode"] == expected
        assert html_artifact.metadata["evaluation_mode"] == expected

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_evaluation_mode_sits_beside_data_config(self, mock_predictor_class, mock_read_csv, tmp_path):
        """evaluation_mode is top-level context; sampling/split configs stay inside data_config."""
        models_artifact, _html_artifact = _run_training_for_evaluation_mode(
            mock_predictor_class,
            mock_read_csv,
            tmp_path,
            sampling_config={"sample": True},
            split_config={"test_size": 0.2},
            evaluation_mode="user-provided",
        )

        context = models_artifact.metadata["context"]
        assert context["evaluation_mode"] == "user-provided"
        assert context["data_config"]["sampling_config"] == {"sample": True}
        assert context["data_config"]["split_config"] == {"test_size": 0.2}
        assert "evaluation_mode" not in context["data_config"]


class TestComponentStatusOutput:
    """Verify the component writes meaningful component_status.json content."""

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_regression_writes_component_status_json(self, mock_predictor_class, mock_read_csv, tmp_path):
        """Happy path persists component_id and completed stages to component_status.json."""
        from kfp_components.components.training.automl.shared.component_status import (
            COMPONENT_STATUS_FILENAME,
            load_component_status,
        )

        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "regression"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "r2"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictor_clone.evaluate_predictions.return_value = {"r2": 0.9}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"feature1": 0.1})
        mock_predictor_clone.predict.return_value = mock.MagicMock()

        mock_read_csv.side_effect = [_mock_csv_frame(), _mock_csv_frame(), _mock_csv_frame()]

        workspace_path = str(tmp_path / "ws")
        Path(workspace_path).mkdir()
        models_output_dir = str(tmp_path / "out")
        Path(models_output_dir).mkdir()
        mock_models_artifact = mock.MagicMock()
        mock_models_artifact.path = models_output_dir
        mock_models_artifact.metadata = {}

        status_artifact = _make_component_status_artifact(tmp_path)
        Path(status_artifact.path).mkdir(parents=True, exist_ok=True)

        call_kwargs = _base_call_kwargs(
            workspace_path, mock_models_artifact, mock.MagicMock(path="/tmp/test.csv"), tmp_path
        )
        call_kwargs["component_status"] = status_artifact
        autogluon_models_training.python_func(**call_kwargs)

        status_path = Path(status_artifact.path) / COMPONENT_STATUS_FILENAME
        assert status_path.is_file()
        data = load_component_status(status_artifact.path)
        assert data["component_id"] == "autogluon_models_training"
        assert data["stages"]
        assert data["stages"][-1]["status"] == "completed"
