"""SFT Minimal (Supervised Fine-Tuning) Training Pipeline.

A 4-stage pipeline for standard supervised fine-tuning:
1. Dataset Download
2. SFT Training (instructlab-training backend)
3. Evaluation with lm-eval
4. Model Registry


SFT is the standard approach for adapting pre-trained language models
to new tasks or domains using labeled training data.
"""

import os
import sys

import kfp
import kfp.kubernetes
from kfp import dsl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import (
    kubeflow_model_registry as model_registry,
)
from components.evaluation.lm_eval import universal_llm_evaluator
from components.training.finetuning.sft import train_model

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "50Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "sft-minimal-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="SFT minimal pipeline: standard supervised fine-tuning using instructlab-training",
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size=PVC_SIZE,
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "accessModes": PVC_ACCESS_MODES,
                    "storageClassName": PVC_STORAGE_CLASS,
                }
            ),
        ),
    ),
)
def sft_minimal_pipeline(
    # =========================================================================
    # KEY PARAMETERS (Required/Important) - Sorted by step
    # =========================================================================
    phase_01_dataset_man_data_uri: str,
    phase_01_dataset_man_data_split: float = 0.9,
    phase_02_train_man_train_batch: int = 128,
    phase_02_train_man_epochs: int = 1,
    phase_02_train_man_gpu: int = 1,
    phase_02_train_man_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    phase_02_train_man_tokens: int = 10000,
    phase_02_train_man_workers: int = 4,
    phase_03_eval_man_tasks: list = ["arc_easy"],
    phase_04_registry_man_address: str = "",
    phase_04_registry_man_reg_name: str = "sft-model",
    phase_04_registry_man_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by step
    # =========================================================================
    phase_01_dataset_opt_subset: int = 0,
    phase_02_train_opt_env_vars: str = (
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
        "NCCL_DEBUG=INFO, NCCL_P2P_DISABLE=1, "
        "INSTRUCTLAB_NCCL_TIMEOUT_MS=60000"
    ),
    phase_02_train_opt_learning_rate: float = 5e-6,
    phase_02_train_opt_max_seq_len: int = 8192,
    phase_02_train_opt_fsdp_sharding: str = "FULL_SHARD",
    phase_02_train_opt_use_liger: bool = False,
    phase_02_train_opt_runtime: str = "training-hub",
    phase_04_registry_opt_port: int = 8080,
):
    """SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

    A 4-stage ML pipeline for fine-tuning language models:

    1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
    2) SFT Training - Fine-tunes using instructlab-training backend
    3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
    4) Model Registry - Registers trained model to Kubeflow Model Registry
    Args:
        phase_01_dataset_man_data_uri: [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url)
        phase_01_dataset_man_data_split: Train/eval split (0.9 = 90% train/10% eval, 1.0 = no split, all for training)
        phase_02_train_man_train_batch: Effective batch size (samples per optimizer step). Start with 128
        phase_02_train_man_epochs: Number of training epochs. 1 is often sufficient
        phase_02_train_man_gpu: GPUs per worker. KEEP AT 1 to avoid /dev/shm issues
        phase_02_train_man_model: Base model (HuggingFace ID or path)
        phase_02_train_man_tokens: Max tokens per GPU (memory cap). 10000 for SFT
        phase_02_train_man_workers: Number of training pods. 4 pods × 1 GPU = 4 total GPUs
        phase_03_eval_man_tasks: lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.)
        phase_04_registry_man_address: Model Registry address (empty = skip registration)
        phase_04_registry_man_version: Semantic version (major.minor.patch)
        phase_04_registry_man_reg_name: Model name in registry
        phase_01_dataset_opt_subset: Limit to first N examples (0 = all)
        phase_02_train_opt_env_vars: Env vars (KEY=VAL,...) with NCCL timeout and memory optimization
        phase_02_train_opt_learning_rate: Learning rate (1e-6 to 1e-4). 5e-6 recommended
        phase_02_train_opt_max_seq_len: Max sequence length in tokens
        phase_02_train_opt_fsdp_sharding: FSDP strategy (FULL_SHARD, HYBRID_SHARD, NO_SHARD)
        phase_02_train_opt_use_liger: Enable Liger kernel optimizations
        phase_02_train_opt_runtime: Name of the ClusterTrainingRuntime to use.
        phase_04_registry_opt_port: Model Registry server port.
    """
    # =========================================================================
    # Stage 1: Dataset Download
    # =========================================================================
    dataset_download_task = dataset_download(
        dataset_uri=phase_01_dataset_man_data_uri,
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        train_split_ratio=phase_01_dataset_man_data_split,
        subset_count=phase_01_dataset_opt_subset,
        shared_log_file="pipeline_log.txt",
    )
    dataset_download_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(dataset_download_task, "IfNotPresent")

    kfp.kubernetes.use_secret_as_env(
        dataset_download_task,
        secret_name="s3-secret",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
        optional=True,
    )

    # =========================================================================
    # Stage 2: SFT Training
    # =========================================================================
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset=dataset_download_task.outputs["train_dataset"],
        # Model - SFT specific
        training_base_model=phase_02_train_man_model,
        # Hyperparameters
        training_effective_batch_size=phase_02_train_man_train_batch,
        training_max_tokens_per_gpu=phase_02_train_man_tokens,
        training_max_seq_len=phase_02_train_opt_max_seq_len,
        training_learning_rate=phase_02_train_opt_learning_rate,
        training_seed=42,
        training_num_epochs=phase_02_train_man_epochs,
        # SFT optimizations
        training_use_liger=phase_02_train_opt_use_liger,
        # LR scheduler
        training_lr_scheduler="cosine",
        # Saving (SFT-specific)
        training_checkpoint_at_epoch=True,
        training_save_samples=0,
        training_accelerate_full_state_at_epoch=False,
        training_fsdp_sharding_strategy=phase_02_train_opt_fsdp_sharding,
        # Environment
        training_envs=phase_02_train_opt_env_vars,
        # Resources
        training_resource_cpu_per_worker="4",
        training_resource_gpu_per_worker=phase_02_train_man_gpu,
        training_resource_memory_per_worker="64Gi",
        training_resource_num_procs_per_worker="auto",
        training_resource_num_workers=phase_02_train_man_workers,
        training_runtime=phase_02_train_opt_runtime,
    )
    training_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(training_task, "IfNotPresent")

    kfp.kubernetes.use_secret_as_env(
        task=training_task,
        secret_name="kubernetes-credentials",
        secret_key_to_env={
            "KUBERNETES_SERVER_URL": "KUBERNETES_SERVER_URL",
            "KUBERNETES_AUTH_TOKEN": "KUBERNETES_AUTH_TOKEN",
        },
        optional=False,
    )

    # =========================================================================
    # Stage 3: Evaluation
    # =========================================================================
    eval_task = universal_llm_evaluator(
        model_artifact=training_task.outputs["output_model"],
        eval_dataset=dataset_download_task.outputs["eval_dataset"],
        task_names=phase_03_eval_man_tasks,
        batch_size="auto",
        limit=int(-1),
        log_samples=True,
        verbosity="INFO",
        model_args={},
        gen_kwargs={},
    )
    eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_task, "IfNotPresent")

    kfp.kubernetes.add_node_selector(eval_task, "nvidia.com/gpu.present", "true")
    eval_task.set_accelerator_type("nvidia.com/gpu")
    eval_task.set_accelerator_limit(1)

    # Attach shared Hugging Face token secret to all main tasks
    for _task in [dataset_download_task, training_task, eval_task]:
        kfp.kubernetes.use_secret_as_env(
            task=_task,
            secret_name="hf-token",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
            optional=True,
        )

    # =========================================================================
    # Stage 4: Model Registry
    # =========================================================================
    model_registry_task = model_registry(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        input_model=training_task.outputs["output_model"],
        input_metrics=training_task.outputs["output_metrics"],
        eval_metrics=eval_task.outputs["output_metrics"],
        eval_results=eval_task.outputs["output_results"],
        registry_address=phase_04_registry_man_address,
        registry_port=phase_04_registry_opt_port,
        model_name=phase_04_registry_man_reg_name,
        model_version=phase_04_registry_man_version,
        model_format_name="pytorch",
        model_format_version="2.9",
        model_description="",
        author="pipeline",
        shared_log_file="pipeline_log.txt",
        source_pipeline_name=PIPELINE_NAME,
        source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        source_namespace="",
    )
    model_registry_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(model_registry_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=sft_minimal_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
