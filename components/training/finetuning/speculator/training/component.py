"""Mode-specific Speculator draft-model training component."""

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
def train_speculator_mode(
    verifier_model: str,
    output_dir: str,
    pvc_path: str,
    hidden_states_path: str,
    verifier_model_pvc: Optional[str] = None,
    training_data_path: Optional[str] = None,
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: Optional[str] = None,
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
    data_extraction_concurrency: int = 4,
    enable_progression_tracking: bool = True,
    metrics_port: int = 28080,
    metrics_poll_interval_seconds: int = 30,
    packages_to_install: Optional[str] = None,
    pip_index_urls: Optional[str] = None,
    training_envs: str = "",
    training_runtime: str = "speculator-model-opt-cuda",
    persistent_pvc: str = "",
    persistent_mount_path: str = "/mnt/persistent",
    output_model: dsl.Output[dsl.Model] = None,
    output_metrics: dsl.Output[dsl.Metrics] = None,
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Train a draft model from hidden states already stored on the PVC."""
    from speculator import run_speculator

    return run_speculator("train_only", locals())
