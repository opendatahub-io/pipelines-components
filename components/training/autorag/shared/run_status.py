"""AutoRAG pipeline stage-map templates, notebook templates, and manifest loading.

Pipeline manifests live under ``run_status_templates/pipelines/`` (JSON, one file per
``@dsl.pipeline`` ``name``). Notebook templates live under ``notebook_templates/``.
AutoML components load these from ``kfp_components`` on the runtime image; AutoRAG
embeds them at compile time (see ``component_stage_map_publisher`` and
``rag_templates_optimization``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMPLATES_DIR_NAME = "run_status_templates"
PIPELINES_SUBDIR = "pipelines"

PIPELINE_DOCUMENTS_RAG_OPTIMIZATION = "documents-rag-optimization-pipeline"


def shared_autorag_dir() -> Path:
    """Directory of the installed ``kfp_components...autorag.shared`` package."""
    return Path(__file__).resolve().parent


def resolve_templates_dir(templates_root: str | None = None) -> Path:
    """Return ``run_status_templates`` from the package or an explicit root (tests)."""
    if templates_root:
        return Path(templates_root) / TEMPLATES_DIR_NAME
    return shared_autorag_dir() / TEMPLATES_DIR_NAME


def load_pipeline_run_status_manifest(
    pipeline_id: str,
    *,
    templates_root: str | None = None,
) -> dict[str, Any]:
    """Load ``pipelines/<pipeline_id>.json`` from the shared package.

    Args:
        pipeline_id: Pipeline identifier (e.g., "documents-rag-optimization-pipeline").
                     Must be a simple identifier without path separators.
        templates_root: Optional directory containing ``run_status_templates`` (tests only).
    """
    # Defense-in-depth: prevent path traversal even though pipeline_id is typically hardcoded
    if not pipeline_id.strip() or "/" in pipeline_id or "\\" in pipeline_id or ".." in pipeline_id:
        raise ValueError(f"Invalid pipeline_id '{pipeline_id}': must be a simple identifier without path separators")

    path = resolve_templates_dir(templates_root) / PIPELINES_SUBDIR / f"{pipeline_id}.json"
    if not path.is_file():
        logger.warning(
            "AUTORAG_RUN_STATUS no pipeline manifest for pipeline_id=%s (expected %s)",
            pipeline_id,
            path,
        )
        return {"pipeline_id": pipeline_id, "components": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pipeline_component_ids(pipeline_id: str, *, templates_root: str | None = None) -> list[str]:
    """Ordered component ids from the pipeline manifest."""
    manifest = load_pipeline_run_status_manifest(pipeline_id, templates_root=templates_root)
    return [component["id"] for component in manifest.get("components", []) if component.get("id")]
