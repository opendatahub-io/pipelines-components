"""Kubeflow Model Registry Component.

Registers a trained model to Kubeflow Model Registry using the official SDK.
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-th06-cpu-torch291-py312:odh-3.4",
    packages_to_install=["model-registry==0.3.9"],
)
def kubeflow_model_registry(
    pvc_mount_path: str,
    model_pvc_name: str = "",
    input_model: dsl.Input[dsl.Model] = None,
    input_metrics: dsl.Input[dsl.Metrics] = None,
    eval_metrics: dsl.Input[dsl.Metrics] = None,
    eval_results: dsl.Input[dsl.Artifact] = None,
    registry_address: str = "",
    registry_port: int = 8080,
    model_name: str = "fine-tuned-model",
    model_version: str = "1.0.0",
    model_format_name: str = "pytorch",
    model_format_version: str = "1.0",
    model_description: str = "",
    author: str = "pipeline",
    shared_log_file: str = "pipeline_log.txt",
    # -------------------------------------------------------------------------
    # PROVENANCE / LINEAGE FIELDS (auto-populated from KFP placeholders)
    # -------------------------------------------------------------------------
    source_pipeline_name: str = "",
    source_pipeline_run_id: str = "",
    source_pipeline_run_name: str = "",
    source_namespace: str = "",
) -> str:
    """Register model to Kubeflow Model Registry with full provenance tracking.

    Uses the upstream model artifact (input_model) produced by training,
    or falls back to PVC path if no artifact is provided.

    Args:
        pvc_mount_path: PVC mount path for workspace storage.
        model_pvc_name: Name of the persistent PVC containing the model.
        input_model: Model artifact from training step.
        input_metrics: Training metrics.
        eval_metrics: Evaluation metrics from lm-eval.
        eval_results: Full evaluation results JSON artifact.
        registry_address: Model Registry server address (hostname or IP).
        registry_port: Model Registry server port (default: 8080).
        model_name: Name for the registered model.
        model_version: Version string for the model (e.g. "1.0.0").
        model_format_name: Model format name (e.g. "pytorch", "onnx").
        model_format_version: Model format version.
        model_description: Optional description for the model.
        author: Author name for the model registration.
        shared_log_file: Filename for shared pipeline log.
        source_pipeline_name: Name of the source KFP pipeline.
        source_pipeline_run_id: Unique ID of the pipeline run.
        source_pipeline_run_name: Display name of the pipeline run.
        source_namespace: Namespace where pipeline runs (auto-detected if empty).
    """
    import os

    from model_registry import ModelRegistry
    from model_registry.exceptions import StoreError

    print("=" * 60)
    print("MODEL REGISTRY COMPONENT")
    print("=" * 60)

    # Derive model URI from the persistent PVC when provided. KFP artifact paths
    # such as /s3/... are temporary task-local paths and are not valid storage
    # locations for a persistent Model Registry entry or KServe deployment.
    resolved_model_name = model_name  # User's parameter takes precedence
    model_uri = ""
    base_model_name = None
    if input_model:
        meta = getattr(input_model, "metadata", {}) or {}
        base_model_name = meta.get("model_name")  # e.g., "Qwen/Qwen2.5-1.5B-Instruct"
        # Only use metadata name if user didn't provide one (kept default)
        if model_name == "fine-tuned-model" and base_model_name:
            resolved_model_name = base_model_name
        if model_pvc_name:
            model_subpath = meta.get("pvc_model_subpath", "")
            if not isinstance(model_subpath, str):
                model_subpath = ""
            model_subpath = model_subpath.strip("/") or "final_model"
            model_uri = f"pvc://{model_pvc_name}/{model_subpath}"
        else:
            model_uri = meta.get("artifact_path") or getattr(input_model, "path", "") or model_uri
            if not model_uri:
                model_uri = f"pvc://{pvc_mount_path}/final_model"
    else:
        model_uri = f"pvc://{model_pvc_name}/final_model" if model_pvc_name else f"pvc://{pvc_mount_path}/final_model"

    print(f"\n  Model Name: {resolved_model_name}")
    if base_model_name:
        print(f"  Base Model: {base_model_name}")
    print(f"  Model Version: {model_version}")
    print(f"  Model URI: {model_uri}")
    print(f"  Registry: {registry_address}:{registry_port}")

    # Register to Model Registry
    model_id = "SKIPPED"
    if registry_address:
        print("\n[Registering to Model Registry...]")

        # Ensure address has scheme for client URL building
        server_addr = registry_address
        if not server_addr.startswith("http://") and not server_addr.startswith("https://"):
            server_addr = f"http://{server_addr}"
        # Create client (HTTP/insecure)
        client = ModelRegistry(
            server_address=server_addr,
            port=registry_port,
            author=author,
            is_secure=False,  # HTTP
        )

        # Collect metrics into metadata if provided
        version_metadata = {}
        eval_summary = {}
        try:
            # -------------------------------------------------------------------------
            # PROVENANCE (logged here, passed as direct params to register_model)
            # -------------------------------------------------------------------------
            # Auto-detect namespace if not provided
            resolved_namespace = source_namespace
            if not resolved_namespace:
                try:
                    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
                        resolved_namespace = f.read().strip()
                        print(f"\n  Auto-detected namespace: {resolved_namespace}")
                except Exception:
                    resolved_namespace = ""

            print("\n  Provenance:")
            print("    - Source Kind: kfp")
            print("    - Source Class: pipelinerun")
            if source_pipeline_name:
                version_metadata["pipeline_name"] = source_pipeline_name
                print(f"    - Pipeline Name: {source_pipeline_name}")
            if source_pipeline_run_id:
                print(f"    - Run ID: {source_pipeline_run_id}")
            if source_pipeline_run_name:
                print(f"    - Run Name: {source_pipeline_run_name} (used for UI link text)")
            if resolved_namespace:
                print(f"    - Namespace: {resolved_namespace}")

            # Add base model info
            if base_model_name:
                version_metadata["base_model"] = base_model_name

            # Add training metrics (hyperparameters)
            if input_metrics and getattr(input_metrics, "metadata", None):
                print("\n  Training Hyperparameters:")
                for k, v in input_metrics.metadata.items():
                    if k not in ("display_name", "store_session_info"):
                        version_metadata[f"training_{k}"] = str(v)
                        print(f"    - {k}: {v}")

            # Add evaluation metrics
            if eval_metrics and getattr(eval_metrics, "metadata", None):
                print("\n  Evaluation Metrics:")
                for k, v in eval_metrics.metadata.items():
                    if k not in ("display_name", "store_session_info"):
                        version_metadata[f"eval_{k}"] = str(v)
                        # Extract primary accuracy metrics for summary
                        if "_acc," in k or "_acc_norm," in k:
                            if "stderr" not in k:
                                eval_summary[k] = v
                                print(f"    - {k}: {v:.4f}" if isinstance(v, float) else f"    - {k}: {v}")

                # Print eval config
                eval_tasks = eval_metrics.metadata.get("eval_tasks", "")
                eval_duration = eval_metrics.metadata.get("eval_duration_seconds", "")
                if eval_tasks:
                    print(f"    - Tasks: {eval_tasks}")
                if eval_duration:
                    print(f"    - Duration: {eval_duration}s")

            # Add summary of best metrics
            if eval_summary:
                # Find the best accuracy metric
                best_metric = max(eval_summary.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                version_metadata["eval_best_metric"] = best_metric[0]
                version_metadata["eval_best_score"] = str(best_metric[1])
                print(
                    f"\n  Best Eval Score: {best_metric[0]} = {best_metric[1]:.4f}"
                    if isinstance(best_metric[1], float)
                    else f"\n  Best: {best_metric}"
                )

            print(f"\n  Total metadata keys: {len(version_metadata)}")
        except Exception as e:
            print(f"  Warning: Could not extract metrics: {e}")
            version_metadata = {}

        try:
            registered_model = client.register_model(
                name=resolved_model_name,
                uri=model_uri,
                version=model_version,
                model_format_name=model_format_name,
                model_format_version=model_format_version,
                author=author,
                owner=author,
                description=model_description or f"Registered via pipeline - {model_version}",
                # Provenance (direct params in SDK 0.3.4+)
                model_source_kind="kfp",
                model_source_class="pipelinerun",
                model_source_name=source_pipeline_run_name or source_pipeline_name or None,
                model_source_id=source_pipeline_run_id or None,
                model_source_group=resolved_namespace or None,
                # Additional metadata (training/eval metrics, run name, etc.)
                metadata=version_metadata or None,
            )
            model_id = registered_model.id
            print(f"  Registered model: {registered_model.name} (ID: {model_id})")
        except StoreError as e:
            msg = str(e)
            if "already exists" in msg.lower():
                print(f"  Model version already exists; skipping create. Details: {msg}")
                model_id = f"{resolved_model_name}:{model_version}"
            else:
                raise

    # Write to shared log
    log_path = os.path.join(pvc_mount_path, shared_log_file)
    with open(log_path, "a") as f:
        f.write(f"Model Registry: {model_name} v{model_version} (ID: {model_id})\n")
    print(f"\n[Log written to {log_path}]")

    print("\n" + "=" * 60)
    print(f"COMPLETE - Model ID: {model_id}")
    print("=" * 60)

    return str(model_id)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        kubeflow_model_registry,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: kubeflow_model_registry_component.yaml")
