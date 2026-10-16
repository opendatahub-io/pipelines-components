"""Tests for the Tabular AutoGluon live per-model MLflow progress callback."""

import copy
import pickle
import sys
import types
from contextlib import contextmanager
from unittest import mock

from kfp_components.components.training.automl.shared.mlflow_callbacks import (
    CANDIDATE_FIT_TIME_METRIC,
    CANDIDATE_VAL_SCORE_METRIC,
    CANDIDATES_TRAINED_METRIC,
    MLFLOW_PARENT_RUN_ID_TAG,
    build_mlflow_progress_callback,
)


@contextmanager
def _fake_autogluon():
    """Inject a minimal ``autogluon.core.callbacks.AbstractCallback`` so the callback builds.

    AutoGluon is not installed in the unit-test environment; the real callback requires the
    ``AbstractCallback`` base class, so we stub the module tree for the duration of a test.
    """

    class AbstractCallback:
        def __init__(self) -> None:  # noqa: D401 - trivial base
            pass

    pkg = types.ModuleType("autogluon")
    core = types.ModuleType("autogluon.core")
    callbacks = types.ModuleType("autogluon.core.callbacks")
    callbacks.AbstractCallback = AbstractCallback
    modules = {
        "autogluon": pkg,
        "autogluon.core": core,
        "autogluon.core.callbacks": callbacks,
    }
    with mock.patch.dict(sys.modules, modules):
        yield


def _mlflow_with_client(run_ids: list[str] | None = None) -> tuple[mock.MagicMock, mock.MagicMock]:
    """Return (mock_mlflow, mock_client) where create_run yields the given run ids in order."""
    mock_mlflow = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_mlflow.MlflowClient.return_value = mock_client
    # RunTag(key, value) -> a simple identifiable object for tag assertions.
    mock_mlflow.entities.RunTag.side_effect = lambda key, value: ("tag", key, value)

    ids = iter(run_ids or ["child-1", "child-2", "child-3"])

    def _create_run(**kwargs):
        run = mock.MagicMock()
        run.info.run_id = next(ids)
        return run

    mock_client.create_run.side_effect = _create_run
    return mock_mlflow, mock_client


def _trainer_with_stats(stats: dict[str, dict[str, float]]) -> mock.MagicMock:
    """Build a trainer whose ``get_model_attribute`` returns per-model val_score/fit_time."""
    trainer = mock.MagicMock()

    def _get(model: str, attribute: str):
        return stats[model][attribute]

    trainer.get_model_attribute.side_effect = _get
    return trainer


class TestBuildMlflowProgressCallback:
    """Tests for ``build_mlflow_progress_callback``."""

    def test_returns_none_when_mlflow_missing(self):
        """No callback when the MLflow module is unavailable."""
        assert build_mlflow_progress_callback(None, run_id="parent-run") is None

    def test_returns_none_when_run_id_empty(self):
        """No callback without a parent run id to nest children under."""
        assert build_mlflow_progress_callback(mock.MagicMock(), run_id="") is None

    def test_returns_none_when_autogluon_callback_api_unavailable(self):
        """No callback when the runtime AutoGluon lacks the callback API."""
        # AutoGluon is not installed in the test env, so the import fails and we get None.
        assert build_mlflow_progress_callback(mock.MagicMock(), run_id="parent-run") is None

    def test_creates_live_child_run_per_candidate(self):
        """Each newly trained candidate becomes a nested child run with its scores."""
        mock_mlflow, mock_client = _mlflow_with_client(["child-1"])
        registry: dict[str, str] = {}

        with _fake_autogluon():
            callback = build_mlflow_progress_callback(
                mock_mlflow,
                run_id="parent-run",
                experiment_id="7",
                kfp_run_id="kfp-123",
                registry=registry,
            )

        assert callback is not None
        trainer = _trainer_with_stats({"LightGBM": {"val_score": 0.9, "fit_time": 1.5}})
        result = callback._after_model_fit(trainer, model_names=["LightGBM"])

        assert result is False
        # One nested child run, named after the model, in the parent's experiment.
        _, create_kwargs = mock_client.create_run.call_args
        assert create_kwargs["experiment_id"] == "7"
        assert create_kwargs["run_name"] == "LightGBM"
        # tags must be a plain dict (MlflowClient.create_run builds RunTag objects itself).
        assert create_kwargs["tags"][MLFLOW_PARENT_RUN_ID_TAG] == "parent-run"
        assert create_kwargs["tags"]["model_name"] == "LightGBM"
        # Validation score and fit time recorded on the child run, which is then finalized.
        mock_client.log_metric.assert_any_call("child-1", CANDIDATE_VAL_SCORE_METRIC, 0.9)
        mock_client.log_metric.assert_any_call("child-1", CANDIDATE_FIT_TIME_METRIC, 1.5)
        mock_client.set_terminated.assert_called_once_with("child-1", status="FINISHED")
        # Live progress is also surfaced on the parent, and the registry is shared out.
        mock_client.log_metric.assert_any_call("parent-run", CANDIDATES_TRAINED_METRIC, 1.0, step=1)
        assert registry == {"LightGBM": "child-1"}

    def test_partial_child_run_is_terminated_failed_and_not_registered(self):
        """If metric logging fails after run creation, the run is FAILED, not left RUNNING."""
        mock_mlflow, mock_client = _mlflow_with_client(["child-1"])
        mock_client.log_metric.side_effect = RuntimeError("transient mlflow error")
        registry: dict[str, str] = {}

        with _fake_autogluon():
            callback = build_mlflow_progress_callback(
                mock_mlflow, run_id="parent-run", experiment_id="1", registry=registry
            )

        trainer = _trainer_with_stats({"LightGBM": {"val_score": 0.9, "fit_time": 1.5}})
        # Must not raise even though log_metric fails.
        result = callback._after_model_fit(trainer, model_names=["LightGBM"])

        assert result is False
        # Run was created then terminated as FAILED (never left RUNNING).
        mock_client.set_terminated.assert_any_call("child-1", status="FAILED")
        # A partial run must not be registered for later reopen/reuse.
        assert registry == {}

    def test_increments_and_dedupes_seen_models(self):
        """Distinct models each get a run; a repeat model is not re-created."""
        mock_mlflow, mock_client = _mlflow_with_client(["child-1", "child-2"])
        registry: dict[str, str] = {}

        with _fake_autogluon():
            callback = build_mlflow_progress_callback(
                mock_mlflow, run_id="parent-run", experiment_id="1", registry=registry
            )

        trainer = _trainer_with_stats(
            {
                "LightGBM": {"val_score": 0.9, "fit_time": 1.5},
                "CatBoost": {"val_score": 0.92, "fit_time": 2.0},
            }
        )
        callback._after_model_fit(trainer, model_names=["LightGBM"])
        # LightGBM is already seen, so only CatBoost gets a new run.
        callback._after_model_fit(trainer, model_names=["LightGBM", "CatBoost"])

        assert mock_client.create_run.call_count == 2
        assert registry == {"LightGBM": "child-1", "CatBoost": "child-2"}
        mock_client.log_metric.assert_any_call("child-2", CANDIDATE_VAL_SCORE_METRIC, 0.92)

    def test_deepcopy_and_pickle_produce_inert_copy(self):
        """AutoGluon deep-copies fit kwargs during dynamic stacking; the callback must survive it.

        The mlflow module and client cannot be pickled, so a copy drops them and becomes
        inert (creates no runs), while the original keeps creating live runs.
        """
        mock_mlflow, mock_client = _mlflow_with_client(["child-1"])

        with _fake_autogluon():
            callback = build_mlflow_progress_callback(mock_mlflow, run_id="parent-run", experiment_id="1")

        # Deep copy (the exact operation AutoGluon._dynamic_stacking performs) must not raise.
        copied = copy.deepcopy(callback)
        # And a full pickle round-trip (subprocess dynamic stacking) must also work.
        reloaded = pickle.loads(pickle.dumps(callback))

        trainer = _trainer_with_stats({"LightGBM": {"val_score": 0.9, "fit_time": 1.5}})
        # The inert copies create nothing (no client) but never raise.
        assert copied._after_model_fit(trainer, model_names=["LightGBM"]) is False
        assert reloaded._after_model_fit(trainer, model_names=["LightGBM"]) is False
        mock_client.create_run.assert_not_called()

        # The original still creates a live run normally.
        callback._after_model_fit(trainer, model_names=["LightGBM"])
        mock_client.create_run.assert_called_once()

    def test_never_raises_when_mlflow_client_fails(self):
        """A failing MLflow client is swallowed; the callback still returns False."""
        mock_mlflow, mock_client = _mlflow_with_client()
        mock_client.create_run.side_effect = RuntimeError("boom")

        with _fake_autogluon():
            callback = build_mlflow_progress_callback(mock_mlflow, run_id="parent-run", experiment_id="1")

        trainer = _trainer_with_stats({"LightGBM": {"val_score": 0.9, "fit_time": 1.5}})
        assert callback._after_model_fit(trainer, model_names=["LightGBM"]) is False
