"""OSFT (Orthogonal Subspace Fine-Tuning) Training Pipeline.

A 4-stage pipeline for continual learning without catastrophic forgetting:
1. Dataset Download
2. OSFT Training (mini-trainer backend)
3. Evaluation with lm-eval
4. Model Registry

OSFT enables adapting pre-trained or instruction-tuned models to new tasks
while preserving their original capabilities.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import (
    kubeflow_model_registry as model_registry,
)
from components.evaluation.lm_eval import universal_llm_evaluator
from components.training.finetuning.osft import train_model

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "50Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "osft-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="OSFT pipeline: continual learning without catastrophic forgetting using mini-trainer",
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
def osft_pipeline(
    # =========================================================================
    # KEY PARAMETERS (Required/Important) - Sorted by step
    # =========================================================================
    phase_01_dataset_man_data_uri: str,
    phase_01_dataset_man_data_split: float = 0.9,
    phase_02_train_man_train_batch: int = 128,
    phase_02_train_man_train_epochs: int = 1,
    phase_02_train_man_train_gpu: int = 1,
    phase_02_train_man_train_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    phase_02_train_man_train_tokens: int = 64000,
    phase_02_train_man_train_unfreeze: float = 0.25,
    phase_02_train_man_train_workers: int = 1,
    phase_03_eval_man_eval_tasks: list = ["arc_easy"],
    phase_04_registry_man_address: str = "",
    phase_04_registry_man_reg_author: str = "pipeline",
    phase_04_registry_man_reg_name: str = "osft-model",
    phase_04_registry_man_reg_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by step
    # =========================================================================
    phase_01_dataset_opt_subset: int = 0,
    phase_02_train_opt_annotations: str = "",
    phase_02_train_opt_cpu: str = "8",
    phase_02_train_opt_env_vars: str = "",
    phase_02_train_opt_labels: str = "",
    phase_02_train_opt_learning_rate: float = 5e-6,
    phase_02_train_opt_lr_scheduler: str = "cosine",
    phase_02_train_opt_lr_scheduler_kwargs: str = "",
    phase_02_train_opt_lr_warmup: int = 0,
    phase_02_train_opt_max_seq_len: int = 8192,
    phase_02_train_opt_memory: str = "32Gi",
    phase_02_train_opt_num_procs: str = "auto",
    phase_02_train_opt_processed_data: bool = False,
    phase_02_train_opt_save_epoch: bool = False,
    phase_02_train_opt_save_final: bool = True,
    phase_02_train_opt_seed: int = 42,
    phase_02_train_opt_target_patterns: str = "",
    phase_02_train_opt_unmask: bool = False,
    phase_02_train_opt_use_liger: bool = True,
    phase_02_train_opt_runtime: str = "training-hub",
    phase_03_eval_opt_batch: str = "auto",
    phase_03_eval_opt_gen_kwargs: dict = {},
    phase_03_eval_opt_limit: int = -1,
    phase_03_eval_opt_log_samples: bool = True,
    phase_03_eval_opt_model_args: dict = {},
    phase_03_eval_opt_verbosity: str = "INFO",
    phase_04_registry_opt_description: str = "",
    phase_04_registry_opt_format_name: str = "pytorch",
    phase_04_registry_opt_format_version: str = "1.0",
    phase_04_registry_opt_port: int = 8080,
):
    """OSFT Training Pipeline - Continual learning without catastrophic forgetting.

        A 4-stage ML pipeline for fine-tuning language models with OSFT:

        1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
        2) OSFT Training - Fine-tunes using mini-trainer backend (orthogonal subspace)
        3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
        4) Model Registry - Registers trained model to Kubeflow Model Registry

    Args:
            phase_01_dataset_man_data_uri: [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url)
            phase_01_dataset_man_data_split: Train/eval split (0.9 = 90%/10%, 1.0 = no split)
            phase_02_train_man_train_batch: Effective batch size (samples per optimizer step)
            phase_02_train_man_train_epochs: Number of training epochs. OSFT typically needs 1-2
            phase_02_train_man_train_gpu: GPUs per worker. OSFT handles multi-GPU well
            phase_02_train_man_train_model: Base model (HuggingFace ID or path)
            phase_02_train_man_train_tokens: Max tokens per GPU (memory cap). 64000 for OSFT
            phase_02_train_man_train_unfreeze: [OSFT] Fraction to unfreeze (0.1=minimal, 0.25=balanced, 0.5=strong)
            phase_02_train_man_train_workers: Number of training pods. OSFT efficient single-node (1)
            phase_03_eval_man_eval_tasks: lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.)
            phase_04_registry_man_address: Model Registry address (empty = skip registration)
            phase_04_registry_man_reg_author: Author name for the registered model
            phase_04_registry_man_reg_name: Model name in registry
            phase_04_registry_man_reg_version: Semantic version (major.minor.patch)
            phase_01_dataset_opt_subset: Limit to first N examples (0 = all)
            phase_02_train_opt_annotations: K8s annotations (key=val,...)
            phase_02_train_opt_cpu: CPU cores per worker. 8 recommended for OSFT
            phase_02_train_opt_env_vars: Env vars (KEY=VAL,...). OSFT typically doesn't need special vars
            phase_02_train_opt_labels: K8s labels (key=val,...)
            phase_02_train_opt_learning_rate: Learning rate (1e-6 to 1e-4). 5e-6 recommended
            phase_02_train_opt_lr_scheduler: [OSFT] LR schedule (cosine, linear, constant)
            phase_02_train_opt_lr_scheduler_kwargs: [OSFT] Extra scheduler params (key=val,...)
            phase_02_train_opt_lr_warmup: Warmup steps before full LR
            phase_02_train_opt_max_seq_len: Max sequence length in tokens
            phase_02_train_opt_memory: RAM per worker. 32Gi usually sufficient for OSFT
            phase_02_train_opt_num_procs: Processes per worker ('auto' = one per GPU)
            phase_02_train_opt_processed_data: [OSFT] True if dataset already has tokenized input_ids
            phase_02_train_opt_save_epoch: Save checkpoint at each epoch. Usually False for OSFT
            phase_02_train_opt_save_final: [OSFT] Save final checkpoint after all epochs
            phase_02_train_opt_seed: Random seed for reproducibility
            phase_02_train_opt_target_patterns: [OSFT] Module patterns to unfreeze (empty=auto)
            phase_02_train_opt_unmask: [OSFT] Unmask all tokens (False=assistant only)
            phase_02_train_opt_use_liger: [OSFT] Enable Liger kernel optimizations. Recommended
            phase_02_train_opt_runtime: Name of the ClusterTrainingRuntime to use.
            phase_03_eval_opt_batch: Eval batch size ('auto' or integer)
            phase_03_eval_opt_gen_kwargs: Generation params dict (max_tokens, temperature)
            phase_03_eval_opt_limit: Max samples per task (-1 = all)
            phase_03_eval_opt_log_samples: Log individual predictions
            phase_03_eval_opt_model_args: Model init args dict (dtype, gpu_memory_utilization)
            phase_03_eval_opt_verbosity: Logging level (DEBUG, INFO, WARNING, ERROR)
            phase_04_registry_opt_description: Model description
            phase_04_registry_opt_format_name: Model format (pytorch, onnx, tensorflow)
            phase_04_registry_opt_format_version: Model format version
            phase_04_registry_opt_port: Model registry server port
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
    # Stage 2: OSFT Training
    # =========================================================================
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset=dataset_download_task.outputs["train_dataset"],
        # Model - OSFT specific
        training_base_model=phase_02_train_man_train_model,
        training_unfreeze_rank_ratio=phase_02_train_man_train_unfreeze,
        # Hyperparameters
        training_effective_batch_size=phase_02_train_man_train_batch,
        training_max_tokens_per_gpu=phase_02_train_man_train_tokens,
        training_max_seq_len=phase_02_train_opt_max_seq_len,
        training_learning_rate=phase_02_train_opt_learning_rate,
        training_target_patterns=phase_02_train_opt_target_patterns,
        training_seed=phase_02_train_opt_seed,
        training_num_epochs=phase_02_train_man_train_epochs,
        # OSFT-specific optimizations
        training_use_liger=phase_02_train_opt_use_liger,
        training_use_processed_dataset=phase_02_train_opt_processed_data,
        training_unmask_messages=phase_02_train_opt_unmask,
        # Learning rate scheduler (OSFT)
        training_lr_scheduler=phase_02_train_opt_lr_scheduler,
        training_lr_warmup_steps=phase_02_train_opt_lr_warmup,
        training_lr_scheduler_kwargs=phase_02_train_opt_lr_scheduler_kwargs,
        # Saving (OSFT)
        training_checkpoint_at_epoch=phase_02_train_opt_save_epoch,
        training_save_final_checkpoint=phase_02_train_opt_save_final,
        # Environment
        training_envs=phase_02_train_opt_env_vars,
        training_metadata_labels=phase_02_train_opt_labels,
        training_metadata_annotations=phase_02_train_opt_annotations,
        # Resources
        training_resource_cpu_per_worker=phase_02_train_opt_cpu,
        training_resource_gpu_per_worker=phase_02_train_man_train_gpu,
        training_resource_memory_per_worker=phase_02_train_opt_memory,
        training_resource_num_procs_per_worker=phase_02_train_opt_num_procs,
        training_resource_num_workers=phase_02_train_man_train_workers,
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

    kfp.kubernetes.use_secret_as_env(
        task=training_task,
        secret_name="oci-pull-secret-model-download",
        secret_key_to_env={"OCI_PULL_SECRET_MODEL_DOWNLOAD": "OCI_PULL_SECRET_MODEL_DOWNLOAD"},
        optional=True,
    )

    # =========================================================================
    # Stage 3: Evaluation
    # =========================================================================
    eval_task = universal_llm_evaluator(
        model_artifact=training_task.outputs["output_model"],
        eval_dataset=dataset_download_task.outputs["eval_dataset"],
        task_names=phase_03_eval_man_eval_tasks,
        batch_size=phase_03_eval_opt_batch,
        limit=phase_03_eval_opt_limit,
        log_samples=phase_03_eval_opt_log_samples,
        verbosity=phase_03_eval_opt_verbosity,
        model_args=phase_03_eval_opt_model_args,
        gen_kwargs=phase_03_eval_opt_gen_kwargs,
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
        model_version=phase_04_registry_man_reg_version,
        model_format_name=phase_04_registry_opt_format_name,
        model_format_version=phase_04_registry_opt_format_version,
        model_description=phase_04_registry_opt_description,
        author=phase_04_registry_man_reg_author,
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
        pipeline_func=osft_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
