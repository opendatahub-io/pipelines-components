"""Workspace run status for AutoML tabular pipelines.

Writes ``{workspace}/.automl/run_status.json`` so sequential pipeline tasks on a
shared PVC can append per-component stages.

Pipeline manifests live under ``shared/run_status_templates/pipelines/`` (JSON,
one file per ``@dsl.pipeline`` ``name``). Components load them via
``dsl.EmbeddedInput`` (same pattern as ``notebook_templates/``).
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUN_STATUS_REL_PATH = ".automl/run_status.json"
RUN_STATUS_ARTIFACT_FILENAME = "run_status.json"
RUN_STATUS_ARTIFACT_DISPLAY_NAME = "automl_run_status"
TEMPLATES_DIR_NAME = "run_status_templates"
PIPELINES_SUBDIR = "pipelines"
DOCUMENT_PIPELINE_ID_FIELD = "run_status_pipeline_id"

PIPELINE_TABULAR_TRAINING = "autogluon-tabular-training-pipeline"

COMPONENT_DATA_LOADER = "automl_data_loader"
COMPONENT_MODELS_TRAINING = "autogluon_models_training"
COMPONENT_LEADERBOARD = "leaderboard_evaluation"

_DEFAULT_INITIAL_DOCUMENT: dict[str, Any] = {
    "run_status_rel_path": RUN_STATUS_REL_PATH,
    "components": {},
}


def resolve_templates_dir(templates_root: str | None) -> Path:
    """Return ``run_status_templates`` under the embedded shared root or beside this module."""
    if templates_root:
        return Path(templates_root) / TEMPLATES_DIR_NAME
    return Path(__file__).resolve().parent / TEMPLATES_DIR_NAME


def load_pipeline_run_status_manifest(
    pipeline_id: str,
    *,
    templates_root: str | None = None,
) -> dict[str, Any]:
    """Load ``pipelines/<pipeline_id>.json`` from the embedded shared artifact."""
    path = resolve_templates_dir(templates_root) / PIPELINES_SUBDIR / f"{pipeline_id}.json"
    if not path.is_file():
        logger.warning(
            "AUTOML_RUN_STATUS no pipeline manifest for pipeline_id=%s (expected %s)",
            pipeline_id,
            path,
        )
        return {"pipeline_id": pipeline_id, "components": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pipeline_component_ids(pipeline_id: str, *, templates_root: str | None = None) -> list[str]:
    """Ordered component ids from the pipeline manifest."""
    manifest = load_pipeline_run_status_manifest(pipeline_id, templates_root=templates_root)
    components = sorted(manifest.get("components", []), key=lambda c: c.get("order", 0))
    return [c["id"] for c in components if c.get("id")]


def load_component_stage_catalog(
    component_name: str,
    *,
    pipeline_id: str | None = None,
    workspace_path: str | None = None,
    templates_root: str | None = None,
) -> dict[str, Any]:
    """Return one component entry (with ``stages``) from the pipeline manifest."""
    if pipeline_id is None and workspace_path:
        pipeline_id = resolve_run_status_pipeline_id(workspace_path)
    if not pipeline_id:
        return {"id": component_name, "stages": []}
    manifest = load_pipeline_run_status_manifest(pipeline_id, templates_root=templates_root)
    for component in manifest.get("components", []):
        if component.get("id") == component_name:
            return component
    logger.warning(
        "AUTOML_RUN_STATUS component=%s not in pipeline manifest pipeline_id=%s",
        component_name,
        pipeline_id,
    )
    return {"id": component_name, "stages": []}


def expected_stage_ids(
    component_name: str,
    *,
    pipeline_id: str | None = None,
    workspace_path: str | None = None,
    templates_root: str | None = None,
) -> list[str]:
    """Ordered stage ids for a component from the pipeline manifest."""
    catalog = load_component_stage_catalog(
        component_name,
        pipeline_id=pipeline_id,
        workspace_path=workspace_path,
        templates_root=templates_root,
    )
    return [stage["id"] for stage in catalog.get("stages", [])]


def resolve_run_status_pipeline_id(workspace_path: str) -> str | None:
    """Read the static pipeline manifest id stored at run init."""
    document = load_run_status(workspace_path)
    pipeline_id = document.get(DOCUMENT_PIPELINE_ID_FIELD)
    return pipeline_id if isinstance(pipeline_id, str) and pipeline_id else None


def run_status_file_path(workspace_path: str) -> Path:
    """Absolute path to the run status JSON file under the pipeline workspace."""
    return Path(workspace_path) / RUN_STATUS_REL_PATH


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_run_status(workspace_path: str) -> dict[str, Any]:
    """Load run status from the workspace, or return an empty document."""
    path = run_status_file_path(workspace_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_run_status(workspace_path: str, document: dict[str, Any]) -> None:
    """Persist run status JSON to the workspace."""
    path = run_status_file_path(workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document["updated_at"] = _utc_now_iso()
    with path.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)


def _initial_document_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if "initial_document" in manifest:
        return copy.deepcopy(manifest["initial_document"])
    return copy.deepcopy(_DEFAULT_INITIAL_DOCUMENT)


def _log_pipeline_flow(pipeline_id: str, *, templates_root: str | None = None) -> None:
    component_ids = pipeline_component_ids(pipeline_id, templates_root=templates_root)
    if component_ids:
        logger.info(
            "AUTOML_RUN_STATUS pipeline_id=%s component_flow=%s",
            pipeline_id,
            " -> ".join(component_ids),
        )


def init_run_status(
    workspace_path: str,
    *,
    kfp_run_id: str,
    pipeline_name: str,
    run_status_pipeline_id: str,
    templates_root: str | None = None,
) -> None:
    """Create or reset the run status document for a new pipeline run."""
    manifest = load_pipeline_run_status_manifest(run_status_pipeline_id, templates_root=templates_root)
    document = _initial_document_from_manifest(manifest)
    document["kfp_run_id"] = kfp_run_id
    document["pipeline_name"] = pipeline_name
    document[DOCUMENT_PIPELINE_ID_FIELD] = run_status_pipeline_id
    document["run_status_rel_path"] = RUN_STATUS_REL_PATH
    save_run_status(workspace_path, document)
    _log_pipeline_flow(run_status_pipeline_id, templates_root=templates_root)


def _log_expected_stages(
    component_name: str,
    *,
    pipeline_id: str | None,
    templates_root: str | None,
) -> None:
    if not pipeline_id:
        return
    stage_ids = expected_stage_ids(component_name, pipeline_id=pipeline_id, templates_root=templates_root)
    if stage_ids:
        logger.info(
            "AUTOML_RUN_STATUS pipeline_id=%s component=%s expected_stages=%s",
            pipeline_id,
            component_name,
            ",".join(stage_ids),
        )


def validate_component_stages(
    document: dict[str, Any],
    component_name: str,
    *,
    templates_root: str | None = None,
) -> None:
    """Log warnings when recorded stages diverge from the pipeline manifest (non-fatal)."""
    pipeline_id = document.get(DOCUMENT_PIPELINE_ID_FIELD)
    if not isinstance(pipeline_id, str) or not pipeline_id:
        return
    expected = set(
        expected_stage_ids(component_name, pipeline_id=pipeline_id, templates_root=templates_root)
    )
    if not expected:
        return
    entry = document.get("components", {}).get(component_name, {})
    recorded = {stage.get("id") for stage in entry.get("stages", []) if stage.get("id")}
    missing = expected - recorded
    unknown = recorded - expected
    if missing:
        logger.warning(
            "AUTOML_RUN_STATUS pipeline_id=%s component=%s missing manifest stages: %s",
            pipeline_id,
            component_name,
            sorted(missing),
        )
    if unknown:
        logger.warning(
            "AUTOML_RUN_STATUS pipeline_id=%s component=%s stages not in manifest: %s",
            pipeline_id,
            component_name,
            sorted(unknown),
        )


def begin_component(
    workspace_path: str,
    component_name: str,
    *,
    templates_root: str | None = None,
) -> None:
    """Mark a pipeline component as running."""
    pipeline_id = resolve_run_status_pipeline_id(workspace_path)
    _log_expected_stages(component_name, pipeline_id=pipeline_id, templates_root=templates_root)
    document = load_run_status(workspace_path)
    components = document.setdefault("components", {})
    entry = components.setdefault(component_name, {})
    entry["state"] = "running"
    entry.setdefault("stages", [])
    save_run_status(workspace_path, document)


def record_stage(
    workspace_path: str,
    component_name: str,
    stage_id: str,
    status: str,
    **details: Any,
) -> None:
    """Append a stage entry for a component (e.g. ``read_and_sample``, ``completed``)."""
    document = load_run_status(workspace_path)
    components = document.setdefault("components", {})
    entry = components.setdefault(component_name, {"state": "running", "stages": []})
    stage: dict[str, Any] = {
        "id": stage_id,
        "status": status,
        "timestamp": _utc_now_iso(),
    }
    stage.update(details)
    entry.setdefault("stages", []).append(stage)
    save_run_status(workspace_path, document)
    logger.info("AUTOML_RUN_STATUS component=%s stage=%s status=%s", component_name, stage_id, status)


def complete_component(workspace_path: str, component_name: str, *, state: str = "completed") -> None:
    """Mark a component finished (``completed`` or ``failed``)."""
    document = load_run_status(workspace_path)
    components = document.setdefault("components", {})
    entry = components.setdefault(component_name, {"stages": []})
    entry["state"] = state
    save_run_status(workspace_path, document)
    logger.info("AUTOML_RUN_STATUS component=%s state=%s", component_name, state)


def publish_run_status_artifact(
    artifact_path: str,
    workspace_path: str,
    *,
    component_name: str | None = None,
    validate: bool = True,
    templates_root: str | None = None,
) -> dict[str, Any]:
    """Copy the workspace run status JSON into a KFP ``Output[Artifact]`` directory."""
    document = load_run_status(workspace_path)
    if validate and component_name:
        validate_component_stages(document, component_name, templates_root=templates_root)
    dest_dir = Path(artifact_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / RUN_STATUS_ARTIFACT_FILENAME
    with dest_file.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    return document


class RunStatusRecorder:
    """Per-component helper; pass ``templates_root`` from ``EmbeddedInput.path`` (shared root)."""

    def __init__(
        self,
        workspace_path: str,
        component_name: str,
        *,
        templates_root: str | None = None,
    ) -> None:
        self.workspace_path = workspace_path
        self.component_name = component_name
        self.templates_root = templates_root

    @staticmethod
    def init_pipeline_run(
        workspace_path: str,
        *,
        kfp_run_id: str,
        pipeline_name: str,
        run_status_pipeline_id: str,
        templates_root: str | None = None,
    ) -> None:
        """Initialize run status for the first task in a pipeline (data loader)."""
        init_run_status(
            workspace_path,
            kfp_run_id=kfp_run_id,
            pipeline_name=pipeline_name,
            run_status_pipeline_id=run_status_pipeline_id,
            templates_root=templates_root,
        )

    def begin(self) -> None:
        begin_component(
            self.workspace_path,
            self.component_name,
            templates_root=self.templates_root,
        )

    def record(self, stage_id: str, status: str, **details: Any) -> None:
        record_stage(self.workspace_path, self.component_name, stage_id, status, **details)

    def complete(self, *, state: str = "completed") -> None:
        complete_component(self.workspace_path, self.component_name, state=state)

    def publish_artifact(self, artifact_path: str, *, validate: bool = True) -> dict[str, Any]:
        return publish_run_status_artifact(
            artifact_path,
            self.workspace_path,
            component_name=self.component_name,
            validate=validate,
            templates_root=self.templates_root,
        )
