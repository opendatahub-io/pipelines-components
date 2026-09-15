"""Mode-specific Speculator hidden-state extraction component."""

import os
from typing import Optional

from kfp import dsl

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


@dsl.component(
    base_image=(
        "quay.io/opendatahub/odh-th-torch-cuda-py312@"
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
def extract_speculator(
    verifier_model: str,
    output_dir: str,
    pvc_path: str,
    dataset_name: str,
    hidden_states_path: Optional[str] = None,
    vllm_endpoint: Optional[str] = None,
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: Optional[str] = None,
    data_extraction_max_samples: Optional[int] = None,
    data_extraction_regenerate_responses: bool = False,
    data_extraction_concurrency: int = 4,
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
    training_runtime: str = "speculator-model-opt-cuda",
    persistent_pvc: str = "",
    persistent_mount_path: str = "/mnt/persistent",
    download_model: bool = False,
    output_hidden_states: dsl.Output[dsl.Artifact] = None,
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Extract verifier hidden states using managed or external vLLM.

    ``dataset_name`` selects the source dataset. When regeneration is enabled,
    responses are regenerated through the verifier before hidden-state extraction.
    Managed mode uses the in-job vLLM sidecar; providing ``vllm_endpoint`` uses an
    external vLLM service and requires matching shared model and hidden-state paths.
    """
    from speculator.shared.speculator import run_speculator

    return run_speculator("data_only", locals())
