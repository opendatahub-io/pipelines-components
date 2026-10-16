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
    pipeline execution begins. An ``mlflow`` block is added from the platform-injected
    ``KFP_MLFLOW_CONFIG``, so the dashboard can deep-link the run into the MLflow UI
    (``tracking_enabled: false`` when the platform integration is off).

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
    import os
    from datetime import UTC, datetime
    from pathlib import Path

    def _build_mlflow_stage_map_block() -> dict:
        """Build the ``mlflow`` block from the platform-injected ``KFP_MLFLOW_CONFIG`` blob.

        Inlined so the publisher keeps embedding only ``run_status_templates`` (KFP allows
        one embedded artifact path). Mirrors ``build_mlflow_stage_map_block`` in
        ``shared/mlflow_tracking.py``.
        """
        raw = os.getenv("KFP_MLFLOW_CONFIG", "").strip()
        if not raw:
            return {"tracking_enabled": False}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"tracking_enabled": False}
        if not isinstance(data, dict):
            return {"tracking_enabled": False}

        tracking_uri = str(data.get("endpoint", "")).strip()
        if not tracking_uri:
            return {"tracking_enabled": False}

        # Same gate as resolve_mlflow_config(): never advertise tracking when the optimizer
        # would refuse to send a ServiceAccount token over cleartext HTTP.
        auth_type = str(data.get("authType", "")).strip()
        if auth_type == "kubernetes" and not tracking_uri.lower().startswith("https://"):
            return {"tracking_enabled": False}

        block: dict = {"tracking_enabled": True, "tracking_uri": tracking_uri}
        experiment_id = str(data.get("experimentId", "")).strip()
        if experiment_id:
            block["experiment_id"] = experiment_id
        parent_run_id = str(data.get("parentRunId", "")).strip()
        if parent_run_id:
            block["run_id"] = parent_run_id
        workspace = str(data.get("workspace", "")).strip() if data.get("workspacesEnabled") else ""
        if workspace:
            block["workspace"] = workspace
        if experiment_id and parent_run_id:
            base = tracking_uri.rstrip("/")
            block["run_url"] = f"{base}/#/experiments/{experiment_id}/runs/{parent_run_id}"
        return block

    if not isinstance(pipeline_id, str) or not pipeline_id.strip():
        raise ValueError("pipeline_id must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if "/" in pipeline_id or "\\" in pipeline_id:
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
    if not stage_map.get("components"):
        raise FileNotFoundError(
            f"Component stage map not found or empty for pipeline_id='{pipeline_id}'. "
            "Ensure run_status_templates/pipelines/<pipeline_id>.json is packaged in the embedded artifact."
        )

    stage_map["kfp_run_id"] = run_id
    stage_map["published_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    stage_map["mlflow"] = _build_mlflow_stage_map_block()

    output_path = Path(component_stage_map.path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "component_stage_map.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(stage_map, f, indent=2)

    component_count = len(stage_map.get("components", []))
    component_stage_map.metadata["display_name"] = "Component Stage Map"
    component_stage_map.metadata["pipeline_id"] = pipeline_id
    component_stage_map.metadata["component_count"] = component_count
    mlflow_block = stage_map["mlflow"]
    component_stage_map.metadata["mlflow_tracking_enabled"] = str(mlflow_block.get("tracking_enabled", False))
    stage_count = sum(len(c.get("stages", [])) for c in stage_map.get("components", []))
    print(f"Published component stage map for pipeline_id='{pipeline_id}':")
    print(f"  - Components: {component_count}")
    print(f"  - Total stages: {stage_count}")
    print(f"  - Published to: {output_file}")
    print(f"  - MLflow tracking enabled: {mlflow_block.get('tracking_enabled', False)}")
