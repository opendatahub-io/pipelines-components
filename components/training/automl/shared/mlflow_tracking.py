"""MLflow helpers for AutoML pipelines (RHOAI 3.5+ opt-in via env vars).

Embedded via ``embedded_artifact_path`` on AutoML components; imported with bare
module names inside component bodies (same pattern as ``leaderboard_utils``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

TRACKING_ARTIFACT_FILENAME = "mlflow_tracking.json"


def mlflow_enabled() -> bool:
    """Return True when MLflow tracking URI is configured."""
    return bool(os.getenv("MLFLOW_TRACKING_URI", "").strip())


def build_mlflow_run_url(
    tracking_uri: str | None,
    experiment_id: str | None,
    run_id: str | None,
    workspace: str | None,
) -> str | None:
    """Build a deep-link URL to the parent run in the MLflow UI."""
    if not tracking_uri or not experiment_id or not run_id:
        return None
    base = tracking_uri.rstrip("/")
    url = f"{base}/#/experiments/{experiment_id}/runs/{run_id}"
    if workspace:
        url += f"?workspace={quote(workspace, safe='')}"
    return url


def build_tracking_artifact_payload(
    *,
    kfp_run_id: str,
    pipeline_name: str,
) -> dict[str, Any]:
    """Build the JSON payload for the KFP ``mlflow_tracking_artifact`` output."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    enabled = bool(tracking_uri)
    payload: dict[str, Any] = {
        "tracking_enabled": enabled,
        "kfp_run_id": kfp_run_id,
        "pipeline_name": pipeline_name,
    }
    if enabled:
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
        mlflow_run_id = os.getenv("MLFLOW_RUN_ID")
        workspace = os.getenv("MLFLOW_WORKSPACE")
        payload.update(
            {
                "mlflow_tracking_uri": tracking_uri,
                "mlflow_experiment_id": experiment_id,
                "mlflow_run_id": mlflow_run_id,
                "mlflow_workspace": workspace,
                "mlflow_run_url": build_mlflow_run_url(tracking_uri, experiment_id, mlflow_run_id, workspace),
            }
        )
    return payload


def write_tracking_artifact(artifact_path: str, payload: dict[str, Any]) -> None:
    """Write ``mlflow_tracking.json`` under the KFP artifact directory."""
    out_dir = Path(artifact_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / TRACKING_ARTIFACT_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_model_display_name(display_name: str) -> tuple[str, int]:
    """Infer ``model_type`` and ``stack_level`` from an AutoGluon model name."""
    name = display_name.removesuffix("_FULL") if display_name.endswith("_FULL") else display_name
    stack_level = 1
    if "_L" in name:
        suffix = name.rsplit("_L", 1)[-1]
        try:
            stack_level = int(suffix)
        except ValueError:
            stack_level = 1
    model_type = name.split("_")[0] if "_" in name else name
    return model_type, stack_level


def _score_for_model(metrics: dict[str, Any], eval_metric: str) -> float | None:
    if eval_metric not in metrics:
        return None
    try:
        return float(metrics[eval_metric])
    except (TypeError, ValueError):
        return None


def log_tabular_training_to_mlflow(
    *,
    models_artifact_path: str,
    models_artifact_uri: str,
    model_names: list[str],
    eval_results_by_model: dict[str, dict[str, Any]],
    eval_metric: str,
    task_type: str,
    pipeline_name: str,
    run_id: str,
    model_config: dict[str, Any],
    top_n: int,
    label_column: str,
) -> None:
    """Resume the KFP parent run and log parent + nested child runs for refitted models."""
    if not mlflow_enabled():
        return

    parent_run_id = os.getenv("MLFLOW_RUN_ID")
    if not parent_run_id:
        logger.info("MLFLOW_TRACKING_URI is set but MLFLOW_RUN_ID is missing; skipping MLflow logging.")
        return

    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    try:
        import autogluon

        ag_version = getattr(autogluon, "__version__", "unknown")
    except Exception:
        ag_version = "unknown"

    with mlflow.start_run(run_id=parent_run_id):
        mlflow.set_tags(
            {
                "pipeline_name": pipeline_name,
                "kfp_run_id": run_id,
                "task_type": task_type,
            }
        )
        mlflow.log_params(
            {
                "eval_metric": eval_metric,
                "preset": model_config.get("preset", ""),
                "top_n": top_n,
                "label_column": label_column,
                "task_type": task_type,
                "autogluon_version": ag_version,
            }
        )

        scores: list[float] = []
        best_name = ""
        best_score_val = float("-inf")
        for name in model_names:
            score = _score_for_model(eval_results_by_model.get(name, {}), eval_metric)
            if score is not None:
                scores.append(score)
                if score > best_score_val:
                    best_score_val = score
                    best_name = name

        if scores:
            mlflow.log_metric("best_score", max(scores))
            mlflow.log_metric("worst_score", min(scores))
            mlflow.log_metric("mean_score", sum(scores) / len(scores))
            if best_name:
                mlflow.log_param("best_model_name", best_name)

        mlflow.log_metric("num_models_trained", len(model_names))

        base_uri = models_artifact_uri.rstrip("/")
        for display_name in model_names:
            metrics = eval_results_by_model.get(display_name, {})
            model_type, stack_level = parse_model_display_name(display_name)
            model_uri = f"{base_uri}/{display_name}"

            with mlflow.start_run(run_name=display_name, nested=True):
                mlflow.log_params(
                    {
                        "model_type": model_type,
                        "stack_level": stack_level,
                        "metrics_path": f"{models_artifact_path}/{display_name}/metrics",
                        "predictor_path": f"{model_uri}/predictor",
                        "notebook_path": f"{model_uri}/notebooks/automl_predictor_notebook.ipynb",
                    }
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, float(value))
