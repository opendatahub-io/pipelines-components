"""Component stage map publisher for AutoML pipelines.

Publishes the static component-to-stage-to-step map as a KFP artifact at pipeline start so
dashboards know the expected structure before components run.

Dashboard contract
~~~~~~~~~~~~~~~~~~
The stage map (``component_stage_map.json``) defines the *expected* structure;
each component then emits a ``component_status.json`` artifact tracking *actual*
progress against that map.

Join keys:
- ``component_status.component_id`` == stage-map ``components[].id``
- ``component_status.stages[].id`` == stage-map ``components[].stages[].id``
- ``component_status.stages[].status.step`` (when present) must be one of
  stage-map ``components[].stages[].steps[]``

The canonical JSON Schema for ``component_status.json`` lives in this component
directory at ``component_status.schema.json``.
"""

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,
    install_kfp_package=False,
)
def publish_component_stage_map(
    pipeline_id: str,
    run_id: str,
    component_stage_map: dsl.Output[dsl.Artifact],
) -> None:
    """Publish the component-to-stage-to-step map for dashboard consumption.

    Reads the static JSON template from the package (``run_status_templates/pipelines/``)
    and publishes it as a KFP artifact. Dashboards use this map to show expected
    components, stages, and steps before pipeline execution begins.

    Args:
        pipeline_id: Pipeline identifier matching the template filename
            (e.g. ``autogluon-tabular-training-pipeline``).
        run_id: KFP run ID for tracking (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``).
        component_stage_map: Output artifact containing the component-to-stage-to-step map.

    Raises:
        FileNotFoundError: If the template for ``pipeline_id`` is missing or empty.
        ValueError: If ``pipeline_id`` or ``run_id`` is empty.

    Example:
        map_task = publish_component_stage_map(
            pipeline_id="autogluon-tabular-training-pipeline",
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        )
    """
    import json
    import os
    from datetime import UTC, datetime
    from pathlib import Path

    from kfp_components.components.training.automl.shared.run_status import (
        load_pipeline_run_status_manifest,
    )

    def _build_mlflow_stage_map_block() -> dict:
        """Build the ``mlflow`` block from the platform-injected ``KFP_MLFLOW_CONFIG`` blob.

        Inlined so the publisher does not depend on ``shared/mlflow_tracking.py`` being
        installed in the runtime image (mirrors ``build_mlflow_stage_map_block`` there).
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

    stage_map = load_pipeline_run_status_manifest(pipeline_id)
    # Legacy templates may include an empty initial_document shell; not part of the stage map.
    stage_map.pop("initial_document", None)
    if not stage_map.get("components"):
        raise FileNotFoundError(
            f"Component stage map not found or empty for pipeline_id='{pipeline_id}'. "
            "Ensure run_status_templates/pipelines/<pipeline_id>.json is packaged in the AutoML image."
        )

    stage_map["kfp_run_id"] = run_id
    stage_map["published_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    stage_map["mlflow"] = _build_mlflow_stage_map_block()

    output_path = Path(component_stage_map.path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "component_stage_map.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(stage_map, f, indent=2)

    component_stage_map.metadata["display_name"] = "Component Stage Map"
    component_stage_map.metadata["pipeline_id"] = pipeline_id
    component_stage_map.metadata["component_count"] = len(stage_map.get("components", []))
    mlflow_block = stage_map["mlflow"]
    component_stage_map.metadata["mlflow_tracking_enabled"] = str(mlflow_block.get("tracking_enabled", False))

    component_count = len(stage_map.get("components", []))
    stage_count = sum(len(c.get("stages", [])) for c in stage_map.get("components", []))
    print(f"Published component stage map for pipeline_id='{pipeline_id}':")
    print(f"  - Components: {component_count}")
    print(f"  - Total stages: {stage_count}")
    print(f"  - Published to: {output_file}")
    print(f"  - MLflow tracking enabled: {mlflow_block.get('tracking_enabled', False)}")
