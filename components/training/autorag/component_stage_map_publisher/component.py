"""Component stage map publisher for AutoRAG pipelines.

Publishes the static component-to-stage map as a KFP artifact at pipeline start so
dashboards know the expected structure before components run.
"""

from pathlib import Path

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,
    embedded_artifact_path=str(_AUTORAG_SHARED / "run_status_templates"),
    install_kfp_package=False,
)
def publish_component_stage_map(
    pipeline_id: str,
    run_id: str,
    component_stage_map: dsl.Output[dsl.Artifact],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
) -> None:
    """Publish the component-to-stage map for dashboard consumption.

    Reads the static JSON template from the embedded artifact
    (``run_status_templates/pipelines/``) and publishes it as a KFP artifact.
    Dashboards use this map to show expected components, stages, and steps before
    pipeline execution begins.

    Args:
        pipeline_id: Pipeline identifier matching the template filename
            (e.g. ``documents-rag-optimization-pipeline``).
        run_id: KFP run ID for tracking (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``).
        component_stage_map: Output artifact containing the component-to-stage map.
        embedded_artifact: Embedded ``autorag.shared`` package with pipeline templates.

    Raises:
        FileNotFoundError: If the template for ``pipeline_id`` is missing or empty.
        ValueError: If ``pipeline_id`` or ``run_id`` is empty.
    """
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    if not isinstance(pipeline_id, str) or not pipeline_id.strip():
        raise ValueError("pipeline_id must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if "/" in pipeline_id or "\\" in pipeline_id or ".." in pipeline_id:
        raise ValueError(f"Invalid pipeline_id '{pipeline_id}': must be a simple identifier without path separators")

    templates_root = Path(embedded_artifact.path) if embedded_artifact is not None else None
    manifest_path = templates_root / "pipelines" / f"{pipeline_id}.json" if templates_root is not None else None
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Component stage map not found or empty for pipeline_id='{pipeline_id}'. "
            f"Expected embedded template at pipelines/{pipeline_id}.json."
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        stage_map = json.load(f)
    stage_map.pop("initial_document", None)
    if not stage_map.get("components"):
        raise FileNotFoundError(
            f"Component stage map not found or empty for pipeline_id='{pipeline_id}'. "
            "Ensure run_status_templates/pipelines/<pipeline_id>.json is packaged in the embedded artifact."
        )

    stage_map["kfp_run_id"] = run_id
    stage_map["published_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    output_path = Path(component_stage_map.path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "component_stage_map.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(stage_map, f, indent=2)

    component_stage_map.metadata["display_name"] = "Component Stage Map"
    component_stage_map.metadata["pipeline_id"] = pipeline_id
    component_stage_map.metadata["component_count"] = len(stage_map.get("components", []))

    component_count = len(stage_map.get("components", []))
    stage_count = sum(len(c.get("stages", [])) for c in stage_map.get("components", []))
    print(f"Published component stage map for pipeline_id='{pipeline_id}':")
    print(f"  - Components: {component_count}")
    print(f"  - Total stages: {stage_count}")
    print(f"  - Published to: {output_file}")
