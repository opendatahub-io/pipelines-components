"""AutoGluon callbacks that stream live per-model runs to MLflow.

Callbacks are a Tabular-only, developer-facing AutoGluon feature (added in 1.2.0). They
fire during ``TabularPredictor.fit()`` after each *candidate* model is trained on the
internal validation split -- before leaderboard ranking, top-N selection, or the full
refit. This callback turns each of those events into a **live MLflow child run**: as soon
as a model finishes training, a nested run appears under the parent with its validation
score, so the experiment fills in model-by-model while ``fit()`` is still running.

The end-of-run refit loop (in
:class:`~kfp_components.components.training.automl.shared.mlflow_tracking.MlflowExperimentLogger`)
then *enriches* -- rather than duplicates -- these runs: it reuses the child run created
here (matched by model name via a shared ``registry``) and adds the authoritative test
metrics and artifacts. Models that were never selected for refit keep their live
validation-only run, so the search history stays visible.

``TimeSeriesPredictor`` has no equivalent callback system, so this module is used only by
the tabular training component.

Everything here is best-effort: a missing callback API, an unreadable trainer attribute,
or an MLflow error is swallowed so training is never affected.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Metrics logged on each live per-model child run during fit().
CANDIDATE_VAL_SCORE_METRIC = "val_score"
CANDIDATE_FIT_TIME_METRIC = "fit_time"
# Running count logged on the parent run so overall progress is visible live.
CANDIDATES_TRAINED_METRIC = "candidates_trained"

# Kept in sync with mlflow_tracking to avoid a circular import at module load; the values
# are stable tag/run-type strings, not code.
MLFLOW_PARENT_RUN_ID_TAG = "mlflow.parentRunId"
RUN_TYPE_TAG = "run_type"
RUN_TYPE_MODEL = "model"

# Resolve the AutoGluon callback base at import time so the callback class below can be
# defined at MODULE level (see _MlflowProgressCallback for why that matters). AutoGluon is
# present in the training image; when it is absent (e.g. the unit-test env) we fall back to
# ``object`` so the class is still importable/picklable -- build_mlflow_progress_callback
# separately gates on the real API before ever attaching a callback to fit().
try:
    from autogluon.core.callbacks import AbstractCallback as _AbstractCallback
except Exception:  # pragma: no cover - exercised only where autogluon is installed
    _AbstractCallback = object


def _parse_model_name(model_name: str) -> tuple[str, int]:
    """Best-effort (model_type, stack_level) split; lazy import avoids a circular import."""
    try:
        from kfp_components.components.training.automl.shared.mlflow_tracking import parse_model_name

        return parse_model_name(model_name)
    except Exception:  # pragma: no cover - defensive; parse_model_name never raises in practice
        model_type = model_name.split("_")[0] if "_" in model_name else model_name
        return model_type, 1


class _MlflowProgressCallback(_AbstractCallback):
    """Create a live MLflow child run for each candidate model as it finishes training.

    Defined at module level (not nested in ``build_mlflow_progress_callback``) so it is
    picklable by qualified name: AutoGluon deep-copies -- and, when dynamic stacking runs
    its detection fit in a subprocess, pickles -- ``fit()`` kwargs, which include this
    callback. A nested/local class cannot be pickled.
    """

    def __init__(
        self,
        mlflow_mod: Any,
        parent_run_id: str,
        *,
        experiment_id: str = "",
        eval_metric: str = "",
        kfp_run_id: str = "",
        registry: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._mlflow = mlflow_mod
        self._parent_run_id = parent_run_id
        self._experiment_id = experiment_id
        self._eval_metric = eval_metric
        self._kfp_run_id = kfp_run_id
        self._seen: set[str] = set()
        # Shared with MlflowExperimentLogger so the refit loop enriches (rather than
        # duplicates) these runs. Maps candidate model name -> live child run id.
        self._registry: dict[str, str] = registry if registry is not None else {}
        try:
            self._client = mlflow_mod.MlflowClient()
        except Exception:
            logger.exception("Could not create MlflowClient; live progress logging disabled.")
            self._client = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop the un-copyable MLflow module/client so the callback survives copy/pickle.

        AutoGluon deep-copies ``fit()`` kwargs (including ``callbacks``) for the dynamic
        stacking *detection* fit, and neither the ``mlflow`` module nor an ``MlflowClient``
        can be pickled. Nulling them makes those throwaway copies inert (run creation
        no-ops when ``_client is None``), so only the original callback -- used for the real
        fit -- creates live runs; the detection fits stay silent.
        """
        state = self.__dict__.copy()
        state["_mlflow"] = None
        state["_client"] = None
        return state

    def _model_stats(self, trainer: Any, model_name: str) -> tuple[float | None, float | None]:
        """Best-effort read of (val_score, fit_time) for a model from the trainer."""
        score = None
        fit_time = None
        getter = getattr(trainer, "get_model_attribute", None)
        if callable(getter):
            try:
                score = getter(model=model_name, attribute="val_score")
            except Exception:
                logger.debug("No val_score for %s.", model_name, exc_info=True)
            try:
                fit_time = getter(model=model_name, attribute="fit_time")
            except Exception:
                logger.debug("No fit_time for %s.", model_name, exc_info=True)
        return score, fit_time

    def _create_child_run(self, model_name: str, score: float | None, fit_time: float | None) -> None:
        """Create and finalize a nested child run for one freshly trained model."""
        if self._client is None:
            return
        run_id = None
        succeeded = False
        try:
            model_type, stack_level = _parse_model_name(model_name)
            tags = {
                MLFLOW_PARENT_RUN_ID_TAG: self._parent_run_id,
                RUN_TYPE_TAG: RUN_TYPE_MODEL,
                "model_name": model_name,
                "model_type": model_type,
                "stack_level": str(stack_level),
            }
            if self._kfp_run_id:
                tags["kfp_run_id"] = self._kfp_run_id
            # MlflowClient.create_run takes tags as a plain dict (it builds RunTag objects
            # internally); passing a list of RunTag raises AttributeError.
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                run_name=model_name,
                tags=tags,
            )
            run_id = run.info.run_id
            if score is not None:
                self._client.log_metric(run_id, CANDIDATE_VAL_SCORE_METRIC, float(score))
            if fit_time is not None:
                self._client.log_metric(run_id, CANDIDATE_FIT_TIME_METRIC, float(fit_time))
            succeeded = True
        except Exception:
            logger.warning("Failed to create live MLflow child run for %s; continuing.", model_name, exc_info=True)
        finally:
            # Always terminate a created run so it never leaks as RUNNING. Mark it FINISHED
            # only when metric logging succeeded; a partial run is FAILED. Register the run
            # for later reopen (refit loop adds test metrics/artifacts) only on success.
            if run_id is not None:
                try:
                    self._client.set_terminated(run_id, status="FINISHED" if succeeded else "FAILED")
                except Exception:
                    logger.debug("Failed to terminate MLflow child run %s.", run_id, exc_info=True)
                if succeeded:
                    self._registry[model_name] = run_id

    def _log_parent_progress(self, count: int) -> None:
        if self._client is None:
            return
        try:
            self._client.log_metric(self._parent_run_id, CANDIDATES_TRAINED_METRIC, float(count), step=count)
        except Exception:
            logger.debug("Failed to log %s.", CANDIDATES_TRAINED_METRIC, exc_info=True)

    # AutoGluon dispatches the public hook to this underscore-prefixed override.
    def _after_model_fit(
        self,
        trainer: Any,
        model_names: list[str] | None = None,
        stack_name: str = "core",
        level: int = 1,
        **kwargs: Any,
    ) -> bool:
        try:
            for model_name in model_names or []:
                if model_name in self._seen:
                    continue
                self._seen.add(model_name)
                score, fit_time = self._model_stats(trainer, model_name)
                self._create_child_run(model_name, score, fit_time)
                self._log_parent_progress(len(self._seen))
        except Exception:
            logger.exception("MLflow progress callback failed for %s; continuing.", model_names)
        # Never early-stop the trainer from this callback.
        return False


def build_mlflow_progress_callback(
    mlflow: Any,
    *,
    run_id: str,
    experiment_id: str = "",
    eval_metric: str = "",
    kfp_run_id: str = "",
    registry: dict[str, str] | None = None,
) -> Any | None:
    """Build a live per-model AutoGluon callback, or ``None`` when unavailable.

    Returns ``None`` (so the caller simply omits ``callbacks=``) when MLflow is disabled,
    no parent run id is known, or the runtime AutoGluon lacks the callback API. Otherwise
    returns an ``AbstractCallback`` subclass instance that creates one nested child run per
    candidate model -- named after the model, tagged under ``run_id`` -- as it finishes
    training, recording its validation score and fit time. When ``registry`` is provided it
    is populated with ``model_name -> child_run_id`` so the refit loop can enrich the same
    runs instead of creating duplicates.
    """
    if mlflow is None or not run_id:
        return None
    try:
        from autogluon.core.callbacks import AbstractCallback  # noqa: F401 - runtime availability gate
    except Exception:
        logger.info("AutoGluon callback API unavailable; skipping live MLflow progress logging.")
        return None

    try:
        return _MlflowProgressCallback(
            mlflow,
            run_id,
            experiment_id=experiment_id,
            eval_metric=eval_metric,
            kfp_run_id=kfp_run_id,
            registry=registry,
        )
    except Exception:
        logger.exception("Could not construct MLflow progress callback; skipping live progress.")
        return None
