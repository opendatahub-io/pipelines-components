"""AutoML pipeline catalog and legacy workspace run status helpers.

**Dashboard progress (tabular/time series pipelines):** use
``component_stage_map`` (static plan) and per-component ``component_status``
artifacts via ``ComponentStatusTracker`` in ``component_status.py``.

**Legacy workspace API:** ``init_run_status``, ``RunStatusRecorder``, and
``publish_run_status_artifact`` write ``{workspace}/.automl/run_status.json`` for
PVC-based flows. Tabular and time series training pipelines no longer use this path.

Pipeline manifests live under ``shared/run_status_templates/pipelines/`` (JSON,
one file per ``@dsl.pipeline`` ``name``). Components load them from the
``kfp_components`` package (preinstalled on the AutoML runtime image).
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
PIPELINE_TIMESERIES_TRAINING = "autogluon-timeseries-training-pipeline"

COMPONENT_DATA_LOADER = "automl_data_loader"
COMPONENT_MODELS_TRAINING = "autogluon_models_training"
COMPONENT_TIMESERIES_DATA_LOADER = "timeseries_data_loader"
COMPONENT_TIMESERIES_MODELS_TRAINING = "autogluon_timeseries_models_training"
COMPONENT_LEADERBOARD = "leaderboard_evaluation"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_DEFAULT_INITIAL_DOCUMENT: dict[str, Any] = {
    "components": [],
}


def shared_automl_dir() -> Path:
    """Directory of the installed ``kfp_components...automl.shared`` package."""
    return Path(__file__).resolve().parent


def resolve_templates_dir(templates_root: str | None = None) -> Path:
    """Return ``run_status_templates`` from the package or an explicit root (tests)."""
    if templates_root:
        return Path(templates_root) / TEMPLATES_DIR_NAME
    return shared_automl_dir() / TEMPLATES_DIR_NAME


def load_pipeline_run_status_manifest(
    pipeline_id: str,
    *,
    templates_root: str | None = None,
) -> dict[str, Any]:
    """Load ``pipelines/<pipeline_id>.json`` from the shared package.

    Args:
        pipeline_id: Pipeline identifier (e.g., "autogluon-tabular-training-pipeline").
                     Must be a simple identifier without path separators.
        templates_root: Optional directory containing ``run_status_templates`` (tests only).
    """
    # Defense-in-depth: prevent path traversal even though pipeline_id is typically hardcoded
    if not pipeline_id.strip() or "/" in pipeline_id or "\\" in pipeline_id or ".." in pipeline_id:
        raise ValueError(f"Invalid pipeline_id '{pipeline_id}': must be a simple identifier without path separators")

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


def _manifest_component_definitions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered component definitions from a pipeline manifest."""
    return [component for component in manifest.get("components", []) if component.get("id")]


def pipeline_component_ids(pipeline_id: str, *, templates_root: str | None = None) -> list[str]:
    """Ordered component ids from the pipeline manifest."""
    manifest = load_pipeline_run_status_manifest(pipeline_id, templates_root=templates_root)
    return [component["id"] for component in _manifest_component_definitions(manifest)]


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
    for component in _manifest_component_definitions(manifest):
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


def _manifest_stage_steps(
    component_name: str,
    stage_id: str,
    *,
    pipeline_id: str | None,
    templates_root: str | None,
) -> list[str] | None:
    """Ordered step ids from the manifest for a stage, or ``None`` if undefined."""
    if not pipeline_id:
        return None
    for stage in load_component_stage_catalog(
        component_name,
        pipeline_id=pipeline_id,
        templates_root=templates_root,
    ).get("stages", []):
        if stage.get("id") == stage_id:
            raw = stage.get("steps")
            if isinstance(raw, list) and raw:
                return [str(step) for step in raw]
    return None


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


def _components_list(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ``components`` as an ordered list (supports legacy map-shaped documents)."""
    raw = document.get("components")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        items: list[tuple[int, str, dict[str, Any]]] = []
        for component_id, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            merged = dict(entry)
            merged.setdefault("id", component_id)
            sort_key = int(merged.pop("order")) if isinstance(merged.get("order"), int) else 10**9
            items.append((sort_key, component_id, merged))
        items.sort(key=lambda item: (item[0], item[1]))
        return [entry for _, _, entry in items]
    return []


def _set_components_list(document: dict[str, Any], components: list[dict[str, Any]]) -> None:
    document["components"] = components


def _find_component_index(components: list[dict[str, Any]], component_name: str) -> int | None:
    for index, entry in enumerate(components):
        if entry.get("id") == component_name:
            return index
    return None


def _get_component_entry(document: dict[str, Any], component_name: str) -> dict[str, Any] | None:
    components = _components_list(document)
    index = _find_component_index(components, component_name)
    return components[index] if index is not None else None


def _get_or_create_component_entry(
    document: dict[str, Any],
    component_name: str,
    *,
    default_state: str = STATUS_PENDING,
) -> dict[str, Any]:
    components = _components_list(document)
    index = _find_component_index(components, component_name)
    if index is None:
        entry: dict[str, Any] = {"id": component_name, "state": default_state, "stages": []}
        components.append(entry)
    else:
        entry = components[index]
        entry.setdefault("id", component_name)
        entry.setdefault("stages", [])
    _set_components_list(document, components)
    return entry


def _initial_document_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if "initial_document" in manifest:
        return copy.deepcopy(manifest["initial_document"])
    return copy.deepcopy(_DEFAULT_INITIAL_DOCUMENT)


def _catalog_fields_from_definition(defn: dict[str, Any], *, include_steps: bool = False) -> dict[str, Any]:
    """Copy static catalog fields from a manifest component or stage definition."""
    fields: dict[str, Any] = {}
    description = defn.get("description")
    if isinstance(description, str) and description:
        fields["description"] = description
    if include_steps:
        raw_steps = defn.get("steps")
        if isinstance(raw_steps, list) and raw_steps:
            fields["steps"] = [str(step) for step in raw_steps]
    return fields


def _pending_stage_entry(stage_def: dict[str, Any]) -> dict[str, Any]:
    stage_id = stage_def.get("id")
    if not stage_id:
        raise ValueError("manifest stage entry requires an id")
    entry: dict[str, Any] = {"id": stage_id, "status": STATUS_PENDING}
    entry.update(_catalog_fields_from_definition(stage_def, include_steps=True))
    return entry


def _pending_component_entry(comp_def: dict[str, Any]) -> dict[str, Any]:
    component_id = comp_def.get("id")
    if not component_id:
        raise ValueError("manifest component entry requires an id")
    stages = [_pending_stage_entry(stage_def) for stage_def in comp_def.get("stages", []) if stage_def.get("id")]
    entry: dict[str, Any] = {"id": component_id, "state": STATUS_PENDING, "stages": stages}
    entry.update(_catalog_fields_from_definition(comp_def))
    return entry


def _build_pipeline_plan_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """All pipeline components and stages from the manifest, initially ``pending``."""
    return [_pending_component_entry(comp_def) for comp_def in _manifest_component_definitions(manifest)]


def _merge_component_stages(entry: dict[str, Any], comp_def: dict[str, Any]) -> None:
    """Add manifest stages missing from ``entry`` as ``pending``; preserve unknown stage ids."""
    existing_by_id = {stage["id"]: stage for stage in entry.get("stages", []) if stage.get("id")}
    manifest_ids = {stage_def.get("id") for stage_def in comp_def.get("stages", []) if stage_def.get("id")}
    merged: list[dict[str, Any]] = []
    for stage_def in comp_def.get("stages", []):
        stage_id = stage_def.get("id")
        if not stage_id:
            continue
        merged.append(existing_by_id.get(stage_id) or _pending_stage_entry(stage_def))
    for stage_id, stage in existing_by_id.items():
        if stage_id not in manifest_ids:
            merged.append(stage)
    entry["stages"] = merged
    entry.setdefault("state", STATUS_PENDING)


def ensure_pipeline_plan(
    workspace_path: str,
    *,
    templates_root: str | None = None,
) -> None:
    """Ensure the document lists every manifest component and stage (unrun items stay ``pending``)."""
    document = load_run_status(workspace_path)
    if not document:
        return
    pipeline_id = document.get(DOCUMENT_PIPELINE_ID_FIELD)
    if not isinstance(pipeline_id, str) or not pipeline_id:
        return
    manifest = load_pipeline_run_status_manifest(pipeline_id, templates_root=templates_root)
    existing = {entry["id"]: entry for entry in _components_list(document) if isinstance(entry.get("id"), str)}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for comp_def in _manifest_component_definitions(manifest):
        component_id = comp_def["id"]
        seen.add(component_id)
        if component_id in existing:
            entry = existing[component_id]
            _merge_component_stages(entry, comp_def)
        else:
            entry = _pending_component_entry(comp_def)
        merged.append(entry)
    for component_id, entry in existing.items():
        if component_id not in seen:
            merged.append(entry)
    _set_components_list(document, merged)
    save_run_status(workspace_path, document)


def _stage_index(stages: list[dict[str, Any]], stage_id: str) -> int | None:
    for index, stage in enumerate(stages):
        if stage.get("id") == stage_id:
            return index
    return None


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
    document["components"] = _build_pipeline_plan_from_manifest(manifest)
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
    expected = set(expected_stage_ids(component_name, pipeline_id=pipeline_id, templates_root=templates_root))
    if not expected:
        return
    entry = _get_component_entry(document, component_name) or {}
    stages_by_id = {stage.get("id"): stage for stage in entry.get("stages", []) if stage.get("id")}
    missing = expected - set(stages_by_id)
    unknown = set(stages_by_id) - expected
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
    if entry.get("state") == STATUS_COMPLETED:
        still_pending = [
            stage_id for stage_id in expected if stages_by_id.get(stage_id, {}).get("status") == STATUS_PENDING
        ]
        if still_pending:
            logger.warning(
                "AUTOML_RUN_STATUS pipeline_id=%s component=%s completed with pending stages: %s",
                pipeline_id,
                component_name,
                sorted(still_pending),
            )


def begin_component(
    workspace_path: str,
    component_name: str,
    *,
    templates_root: str | None = None,
) -> None:
    """Mark a pipeline component as running."""
    ensure_pipeline_plan(workspace_path, templates_root=templates_root)
    pipeline_id = resolve_run_status_pipeline_id(workspace_path)
    _log_expected_stages(component_name, pipeline_id=pipeline_id, templates_root=templates_root)
    document = load_run_status(workspace_path)
    entry = _get_or_create_component_entry(document, component_name, default_state=STATUS_PENDING)
    entry["state"] = STATUS_RUNNING
    save_run_status(workspace_path, document)


def record_stage(
    workspace_path: str,
    component_name: str,
    stage_id: str,
    status: str,
    *,
    templates_root: str | None = None,
    **details: Any,
) -> None:
    """Record or update a stage for a component.

    When ``status`` is ``completed``, ``stages[].steps`` from the pipeline manifest are
    copied onto the stage object automatically (if defined).
    """
    ensure_pipeline_plan(workspace_path, templates_root=templates_root)
    document = load_run_status(workspace_path)
    pipeline_id = document.get(DOCUMENT_PIPELINE_ID_FIELD)
    entry = _get_or_create_component_entry(document, component_name, default_state=STATUS_RUNNING)
    stages = entry.setdefault("stages", [])
    index = _stage_index(stages, stage_id)
    prior_stage = stages[index] if index is not None else {}
    stage: dict[str, Any] = {
        **prior_stage,
        "id": stage_id,
        "status": status,
        "timestamp": _utc_now_iso(),
    }
    stage.update(details)
    if status == STATUS_COMPLETED and "steps" not in stage:
        manifest_steps = _manifest_stage_steps(
            component_name,
            stage_id,
            pipeline_id=pipeline_id if isinstance(pipeline_id, str) else None,
            templates_root=templates_root,
        )
        if manifest_steps:
            stage["steps"] = manifest_steps
    if index is None:
        stages.append(stage)
    else:
        stages[index] = stage
    save_run_status(workspace_path, document)
    log_msg = "AUTOML_RUN_STATUS component=%s stage=%s status=%s"
    log_args: list[Any] = [component_name, stage_id, status]
    if "steps" in stage:
        log_msg += " steps=%s"
        log_args.append(",".join(stage["steps"]))
    logger.info(log_msg, *log_args)


def complete_component(
    workspace_path: str,
    component_name: str,
    *,
    state: str = STATUS_COMPLETED,
    templates_root: str | None = None,
) -> None:
    """Mark a component finished (``completed`` or ``failed``)."""
    ensure_pipeline_plan(workspace_path, templates_root=templates_root)
    document = load_run_status(workspace_path)
    entry = _get_or_create_component_entry(document, component_name)
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
    """Per-component helper for workspace run status updates."""

    def __init__(  # noqa: D107
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

    def begin(self) -> None:  # noqa: D102
        begin_component(
            self.workspace_path,
            self.component_name,
            templates_root=self.templates_root,
        )

    def record(self, stage_id: str, status: str, **details: Any) -> None:  # noqa: D102
        record_stage(
            self.workspace_path,
            self.component_name,
            stage_id,
            status,
            templates_root=self.templates_root,
            **details,
        )

    def complete(self, *, state: str = STATUS_COMPLETED) -> None:  # noqa: D102
        complete_component(
            self.workspace_path,
            self.component_name,
            state=state,
            templates_root=self.templates_root,
        )

    def publish_artifact(self, artifact_path: str, *, validate: bool = True) -> dict[str, Any]:  # noqa: D102
        return publish_run_status_artifact(
            artifact_path,
            self.workspace_path,
            component_name=self.component_name,
            validate=validate,
            templates_root=self.templates_root,
        )
