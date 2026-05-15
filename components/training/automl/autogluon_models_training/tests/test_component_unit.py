"""Unit tests for the autogluon_models_training component."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True, scope="module")
def isolated_sys_modules():
    """Patch pandas/autogluon in sys.modules for this module; restored on teardown."""
    with mock.patch.dict(sys.modules, clear=False) as mocked_modules:
        mocked_modules["pandas"] = mock.MagicMock()
        _ag = mock.MagicMock()
        _ag.__path__ = []
        _ag.__spec__ = None
        mocked_modules["autogluon"] = _ag
        for sub in ("autogluon.tabular", "autogluon.core", "autogluon.core.metrics"):
            _m = mock.MagicMock()
            _m.__spec__ = None
            if sub == "autogluon.core":
                _m.__path__ = []
            mocked_modules[sub] = _m
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


@pytest.fixture()
def mock_notebooks(tmp_path):
    """Temp directory with minimal regression and classification notebook templates."""
    notebooks_dir = tmp_path / "notebooks_input"
    notebooks_dir.mkdir()
    for name in ("regression_notebook.ipynb", "classification_notebook.ipynb"):
        with open(notebooks_dir / name, "w") as f:
            json.dump(_MINIMAL_NOTEBOOK, f)
    artifact = mock.MagicMock()
    artifact.path = str(notebooks_dir)
    return artifact


def _mock_leaderboard_top_models(mock_predictor, names: list):
    """Make leaderboard().head(n)['model'].values.tolist() return ``names``."""
    chain = mock_predictor.leaderboard.return_value.head.return_value
    chain.__getitem__.return_value.values.tolist.return_value = names


def _base_call_kwargs(workspace_path, models_artifact, test_data, notebooks):
    """Return minimal valid kwargs for autogluon_models_training.python_func."""
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
        notebooks=notebooks,
        extra_train_data_path="/tmp/extra.csv",
    )


class TestAutogluonModelsTrainingUnitTests:
    """Unit tests for the autogluon_models_training component."""

    def test_component_imports_correctly(self):
        """Component is callable and has the expected KFP attributes."""
        assert callable(autogluon_models_training)
        assert hasattr(autogluon_models_training, "python_func")
        assert hasattr(autogluon_models_training, "component_spec")

    # ── Happy path ─────────────────────────────────────────────────────────────

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_regression_happy_path(self, mock_predictor_class, mock_read_csv, mock_notebooks, tmp_path):
        """Full regression flow: fit, select top 2, refit_full batch, per-model artifacts."""
        top_models = ["LightGBM_BAG_L1", "CatBoost_BAG_L1"]
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
            **_base_call_kwargs(workspace_path, mock_models_artifact, mock_test_data, mock_notebooks),
            sampling_config={"sample": True},
            split_config={"split": 0.8},
        )

        # Return value
        assert result.eval_metric == "r2"

        # TabularPredictor constructed and fitted with correct params
        mock_predictor_class.assert_called_once_with(
            problem_type="regression",
            label="target",
            eval_metric="r2",
            path=Path(workspace_path) / "autogluon_predictor",
            verbosity=2,
        )
        mock_predictor_class.return_value.fit.assert_called_once_with(
            train_data=mock_train_df,
            num_stack_levels=1,
            num_bag_folds=4,
            use_bag_holdout=True,
            holdout_frac=0.2,
            time_limit=1800,
            presets="medium_quality",
        )

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
        mock_predictor_clone.predict.assert_any_call(mock_test_df, model="CatBoost_BAG_L1_FULL")

        # evaluate_predictions called per model (not evaluate())
        assert mock_predictor_clone.evaluate_predictions.call_count == 2

        # feature_importance called per model with model= and subsample_size=2000
        assert mock_predictor_clone.feature_importance.call_count == 2
        mock_predictor_clone.feature_importance.assert_any_call(
            mock_test_df, model="LightGBM_BAG_L1_FULL", subsample_size=2000
        )
        mock_predictor_clone.feature_importance.assert_any_call(
            mock_test_df, model="CatBoost_BAG_L1_FULL", subsample_size=2000
        )

        # set_model_best called per model before clone_for_deployment
        assert mock_predictor_clone.set_model_best.call_count == 2
        mock_predictor_clone.set_model_best.assert_any_call(model="LightGBM_BAG_L1_FULL", save_trainer=True)
        mock_predictor_clone.set_model_best.assert_any_call(model="CatBoost_BAG_L1_FULL", save_trainer=True)
        assert mock_predictor_clone.clone_for_deployment.call_count == 2

        # metadata["model_names"] serialized as JSON string
        assert json.loads(mock_models_artifact.metadata["model_names"]) == [
            "LightGBM_BAG_L1_FULL",
            "CatBoost_BAG_L1_FULL",
        ]

        # Artifacts written on disk for each model
        for model_name_full in ("LightGBM_BAG_L1_FULL", "CatBoost_BAG_L1_FULL"):
            model_dir = Path(models_output_dir) / model_name_full
            metrics_dir = model_dir / "metrics"
            assert (metrics_dir / "metrics.json").exists()
            assert (metrics_dir / "feature_importance.json").exists()
            assert not (metrics_dir / "confusion_matrix.json").exists()  # regression: no CM
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
    def test_without_extra_train_data(self, mock_predictor_class, mock_read_csv, mock_notebooks, tmp_path):
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
            notebooks=mock_notebooks,
            extra_train_data_path="",
        )

        # refit_full gets None for extra data
        mock_predictor_clone.refit_full.assert_called_once_with(model=["LightGBM_BAG_L1"], train_data_extra=None)
        # read_csv called only twice (train + test, no extra)
        assert mock_read_csv.call_count == 2

    @mock.patch("autogluon.core.metrics.confusion_matrix")
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_binary_classification_writes_confusion_matrix(
        self, mock_predictor_class, mock_read_csv, mock_confusion_matrix, mock_notebooks, tmp_path
    ):
        """Binary classification writes confusion_matrix.json and uses classification notebook."""
        mock_predictor = mock.MagicMock()
        mock_predictor_clone = mock.MagicMock()
        mock_predictor_class.return_value.fit.return_value = mock_predictor
        mock_predictor.clone.return_value = mock_predictor_clone
        mock_predictor.problem_type = "binary"
        mock_predictor.label = "target"
        mock_predictor.eval_metric = "accuracy"
        _mock_leaderboard_top_models(mock_predictor, ["LightGBM_BAG_L1"])
        mock_predictions = mock.MagicMock()
        mock_predictor_clone.predict.return_value = mock_predictions
        mock_predictor_clone.evaluate_predictions.return_value = {"accuracy": 0.95}
        mock_predictor_clone.feature_importance.return_value = mock.MagicMock(to_dict=lambda: {"f": 0.1})

        confusion_matrix_dict = {"0": {"0": 5, "1": 0}, "1": {"0": 0, "1": 3}}
        mock_confusion_matrix.return_value = mock.MagicMock(to_dict=lambda: confusion_matrix_dict)

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
            task_type="binary",
            top_n=1,
            train_data_path="/tmp/train.csv",
            test_data=mock.MagicMock(path="/tmp/test.csv"),
            workspace_path=workspace_path,
            pipeline_name=PIPELINE_NAME,
            run_id=RUN_ID,
            sample_row=SAMPLE_ROW,
            models_artifact=mock_models_artifact,
            notebooks=mock_notebooks,
        )

        # confusion_matrix called with cached predictions (not a second predict() call)
        mock_confusion_matrix.assert_called_once()
        cm_call = mock_confusion_matrix.call_args[1]
        assert cm_call["prediction"] is mock_predictions
        assert cm_call["output_format"] == "pandas_dataframe"

        # confusion_matrix.json written
        cm_path = Path(models_output_dir) / "LightGBM_BAG_L1_FULL" / "metrics" / "confusion_matrix.json"
        assert cm_path.exists()
        assert json.loads(cm_path.read_text()) == confusion_matrix_dict

    # ── Call order and structural invariants ───────────────────────────────────

    @mock.patch("shutil.rmtree")
    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_operations_called_in_correct_order(
        self, mock_predictor_class, mock_read_csv, mock_rmtree, mock_notebooks, tmp_path
    ):
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
            notebooks=mock_notebooks,
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
    def test_work_path_is_on_pvc_and_cleaned_up(
        self, mock_predictor_class, mock_read_csv, mock_rmtree, mock_notebooks, tmp_path
    ):
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
            notebooks=mock_notebooks,
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
    def test_refit_full_called_once_with_all_top_models(
        self, mock_predictor_class, mock_read_csv, mock_notebooks, tmp_path
    ):
        """refit_full is called exactly once with the full list of top models (batch, not per-model)."""
        top_models = ["LightGBM_BAG_L1", "CatBoost_BAG_L1", "NeuralNetFastAI_BAG_L1"]
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
            notebooks=mock_notebooks,
        )

        mock_predictor_clone.refit_full.assert_called_once_with(model=top_models, train_data_extra=None)
        # clone also called exactly once (not per model)
        mock_predictor.clone.assert_called_once()

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_context_models_metadata(self, mock_predictor_class, mock_read_csv, mock_notebooks, tmp_path):
        """context['models'] contains one entry per model with correct name, location, and metrics."""
        top_models = ["LightGBM_BAG_L1", "CatBoost_BAG_L1"]
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
            notebooks=mock_notebooks,
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
        assert cat["name"] == "CatBoost_BAG_L1_FULL"
        assert cat["location"]["model_directory"] == "CatBoost_BAG_L1_FULL"
        assert cat["location"]["predictor"] == str(Path("CatBoost_BAG_L1_FULL") / "predictor")
        assert cat["location"]["notebook"] == str(
            Path("CatBoost_BAG_L1_FULL") / "notebooks" / "automl_predictor_notebook.ipynb"
        )
        assert cat["location"]["metrics"] == str(Path("CatBoost_BAG_L1_FULL") / "metrics")
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
    def test_raises_on_invalid_problem_type(self, mock_predictor_class, mock_read_csv, mock_notebooks, tmp_path):
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
                notebooks=mock_notebooks,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.tabular.TabularPredictor")
    def test_raises_on_refit_failure(self, mock_predictor_class, mock_read_csv, mock_notebooks, tmp_path):
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
                notebooks=mock_notebooks,
            )

    # ── Input validation ───────────────────────────────────────────────────────

    def _minimal_artifact(self):
        """Return a minimal mock models artifact path/metadata."""
        a = mock.MagicMock()
        a.path = "/tmp/out"
        a.metadata = {}
        return a

    def test_rejects_empty_label_column(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_invalid_task_type(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_empty_train_data_path(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_empty_workspace_path(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_top_n_zero(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_top_n_exceeds_max(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_empty_pipeline_name(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_empty_run_id(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_invalid_sample_row_json(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_sample_row_not_list(self, mock_notebooks):
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
                notebooks=mock_notebooks,
            )

    def test_rejects_invalid_sampling_config_type(self, mock_notebooks):
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
                notebooks=mock_notebooks,
                sampling_config="invalid",
            )

    def test_rejects_invalid_split_config_type(self, mock_notebooks):
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
                notebooks=mock_notebooks,
                split_config=[],
            )
