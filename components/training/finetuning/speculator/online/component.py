"""Mode-specific fused online Speculator component."""

import os
from typing import Optional

from kfp import dsl

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")


@dsl.component(
    base_image=(
        "quay.io/opendatahub/odh-th-torch-cuda-py312:odh-3.6-ea.1@"
        "sha256:9ad2d72ebe892dffd3554eeb81e2a47c9fcf0d97f89ed261d48f4e8bbc367b4b"
    ),
    packages_to_install=["kfp==2.17.0", "kubernetes", "olot"],
    install_kfp_package=False,
    embedded_artifact_path=_SHARED_DIR,
    task_config_passthroughs=[
        dsl.TaskConfigField.RESOURCES,
        dsl.TaskConfigField.KUBERNETES_TOLERATIONS,
        dsl.TaskConfigField.KUBERNETES_NODE_SELECTOR,
        dsl.TaskConfigField.KUBERNETES_AFFINITY,
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.ENV, apply_to_task=True),
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.KUBERNETES_VOLUMES, apply_to_task=True),
    ],
)
def online_draft_model_training(
    verifier_model: str,
    output_dir: str,
    pvc_path: str,
    dataset_name: str,
    verifier_model_pvc: Optional[str] = None,
    persistent_pvc: str = "",
    persistent_mount_path: str = "/mnt/persistent",
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: Optional[str] = None,
    data_extraction_max_samples: Optional[int] = None,
    data_extraction_regenerate_responses: bool = False,
    data_extraction_concurrency: int = 4,
    training_epochs: int = 3,
    training_lr: float = 1e-4,
    training_draft_vocab_size: Optional[int] = None,
    training_num_layers: int = 1,
    training_ttt_steps: int = 1,
    training_norm_before_residual: bool = True,
    training_norm_before_fc: bool = False,
    training_embed_requires_grad: bool = False,
    training_hidden_states_dtype: str = "bfloat16",
    training_scheduler_type: str = "linear",
    training_scheduler_warmup_steps: Optional[int] = None,
    training_scheduler_total_steps: Optional[int] = None,
    training_scheduler_num_cosine_cycles: float = 0.5,
    training_checkpoint_freq: float = 1.0,
    training_save_best: bool = False,
    training_log_freq: int = 1,
    training_resume_from_checkpoint: bool = False,
    training_from_pretrained: Optional[str] = None,
    training_resource_cpu: str = "4",
    training_resource_gpu: int = 1,
    training_resource_memory: str = "64Gi",
    vllm_resource_gpu: int = 1,
    vllm_resource_memory: str = "96Gi",
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_readiness_timeout_minutes: int = 60,
    enable_progression_tracking: bool = True,
    metrics_port: int = 28080,
    metrics_poll_interval_seconds: int = 30,
    packages_to_install: Optional[str] = None,
    pip_index_urls: Optional[str] = None,
    training_envs: str = "",
    training_runtime: str = "vllm-extract-cuda",
    download_model: bool = False,
    output_model: dsl.Output[dsl.Model] = None,
    output_metrics: dsl.Output[dsl.Metrics] = None,
    kubernetes_config: dsl.TaskConfig = None,
):
    """Extract and train a draft model with a managed vLLM sidecar."""
    from pathlib import Path

    # Pre-strip params consumed by inline logic below; all string inputs are also
    # stripped centrally in run_speculator.
    verifier_model = verifier_model.strip()
    output_dir = output_dir.strip()
    pvc_path = pvc_path.strip()
    dataset_name = dataset_name.strip()
    if persistent_pvc != persistent_pvc.strip():
        raise ValueError(f"persistent_pvc must not contain leading or trailing whitespace (got {persistent_pvc!r}).")
    persistent_mount_path = persistent_mount_path.strip()
    verifier_model_pvc = verifier_model_pvc.strip() if verifier_model_pvc else verifier_model_pvc

    def resolve_local_dataset_path(dataset_uri: str) -> str:
        if not dataset_uri:
            return dataset_uri

        if dataset_uri.startswith("pvc://"):
            claim, separator, dataset_path = dataset_uri.removeprefix("pvc://").partition("/")
            if not claim or not separator or not dataset_path:
                raise ValueError("PVC dataset URIs must use pvc://<claim>/<path>")
            requested_path = Path(dataset_path)
            if claim != persistent_pvc:
                if requested_path.is_absolute() or ".." in requested_path.parts:
                    raise ValueError("PVC dataset paths must remain relative")
                if requested_path.suffix.lower() not in {".json", ".jsonl"}:
                    raise ValueError("dataset files on the shared PVC must end in .json or .jsonl")
                return dataset_uri
        elif "://" in dataset_uri:
            return dataset_uri

        else:
            candidate = Path(dataset_uri)
            if candidate.suffix.lower() not in {".json", ".jsonl"}:
                return dataset_uri
            dataset_path = dataset_uri

        if not persistent_pvc:
            raise ValueError("persistent_pvc is required for local JSON/JSONL dataset paths")

        pvc_root = Path(persistent_mount_path).resolve()
        candidate = Path(dataset_path)
        candidate = candidate if candidate.is_absolute() else pvc_root / candidate
        resolved = candidate.resolve()
        try:
            relative_path = resolved.relative_to(pvc_root)
        except ValueError as exc:
            raise ValueError("local dataset paths must remain within persistent_mount_path") from exc

        if not relative_path.parts:
            raise ValueError("dataset path must be a JSON or JSONL file under persistent_mount_path")
        if relative_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("dataset files on the shared PVC must end in .json or .jsonl")
        if not resolved.is_file():
            raise FileNotFoundError(f"dataset file not found on persistent_pvc: {resolved}")

        return f"pvc://{persistent_pvc}/{relative_path.as_posix()}"

    if verifier_model_pvc and not download_model:
        verifier_model = verifier_model_pvc
    dataset_name = resolve_local_dataset_path(dataset_name)

    from speculator import run_speculator

    run_speculator("online", locals())
