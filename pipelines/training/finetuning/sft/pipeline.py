"""SFT (Supervised Fine-Tuning) Training Pipeline.

A 4-stage pipeline for standard supervised fine-tuning:
1. Dataset Download
2. SFT Training (instructlab-training backend)
3. Evaluation with lm-eval
4. Model Registry

SFT is the standard approach for adapting pre-trained language models
to new tasks or domains using labeled training data.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

# Import reusable components
from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.evaluation.lm_eval import universal_llm_evaluator
from components.training.finetuning.sft import train_model

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "50Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "sft-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="SFT pipeline: standard supervised fine-tuning using instructlab-training",
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
def sft_pipeline(
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
    phase_04_registry_man_author: str = "pipeline",
    phase_04_registry_man_name: str = "sft-model",
    phase_04_registry_man_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by step
    # =========================================================================
    phase_01_dataset_opt_subset: int = 0,
    phase_02_train_opt_annotations: str = "",
    phase_02_train_opt_cpu: str = "4",
    phase_02_train_opt_env_vars: str = (
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000"
    ),
    phase_02_train_opt_labels: str = "",
    phase_02_train_opt_learning_rate: float = 5e-6,
    phase_02_train_opt_lr_warmup: int = 0,
    phase_02_train_opt_lr_scheduler: str = "cosine",
    phase_02_train_opt_max_seq_len: int = 8192,
    phase_02_train_opt_memory: str = "64Gi",
    phase_02_train_opt_num_procs: str = "auto",
    phase_02_train_opt_save_epoch: bool = True,
    phase_02_train_opt_save_full_state: bool = False,
    phase_02_train_opt_fsdp_sharding: str = "FULL_SHARD",
    phase_02_train_opt_save_samples: int = 0,
    phase_02_train_opt_seed: int = 42,
    phase_02_train_opt_use_liger: bool = False,
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
    """SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

    A 4-stage ML pipeline for fine-tuning language models:

    1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
    2) SFT Training - Fine-tunes using instructlab-training backend
    3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
    4) Model Registry - Registers trained model to Kubeflow Model Registry

    Args:
        phase_01_dataset_man_data_uri: Dataset location (hf://, s3://, https://).
        phase_01_dataset_man_data_split: Train/eval split (0.9 = 90% train/10% eval, 1.0 = no split, all for training).
        phase_02_train_man_train_batch: Effective batch size per optimizer step.
        phase_02_train_man_epochs: Number of training epochs.
        phase_02_train_man_gpu: GPUs per worker. Keep at 1 to avoid /dev/shm issues.
        phase_02_train_man_model: Base model (HuggingFace ID or path).
        phase_02_train_man_tokens: Max tokens per GPU (memory cap).
        phase_02_train_man_workers: Number of training pods.
        phase_03_eval_man_tasks: lm-eval tasks (arc_easy, mmlu, gsm8k, etc.).
        phase_04_registry_man_address: Model Registry address (empty = skip).
        phase_04_registry_man_author: Author name for the registered model.
        phase_04_registry_man_name: Model name in registry.
        phase_04_registry_man_version: Semantic version (major.minor.patch).
        phase_01_dataset_opt_subset: Limit dataset to N samples (0 = all).
        phase_02_train_opt_annotations: Pod annotations as key=value,key=value.
        phase_02_train_opt_cpu: CPU cores per worker.
        phase_02_train_opt_env_vars: Environment variables as KEY=VAL,KEY=VAL.
        phase_02_train_opt_fsdp_sharding: FSDP strategy (FULL_SHARD, HYBRID_SHARD).
        phase_02_train_opt_labels: Pod labels as key=value,key=value.
        phase_02_train_opt_learning_rate: Learning rate for training.
        phase_02_train_opt_lr_scheduler: LR scheduler type (cosine, linear).
        phase_02_train_opt_lr_warmup: Learning rate warmup steps.
        phase_02_train_opt_max_seq_len: Maximum sequence length in tokens.
        phase_02_train_opt_memory: Memory per worker (e.g., 64Gi).
        phase_02_train_opt_num_procs: Processes per worker (auto or int).
        phase_02_train_opt_save_epoch: Save checkpoint at each epoch.
        phase_02_train_opt_save_full_state: Save full accelerate state at epoch.
        phase_02_train_opt_save_samples: Number of samples to save (0 = none).
        phase_02_train_opt_seed: Random seed for reproducibility.
        phase_02_train_opt_use_liger: Enable Liger kernel optimizations.
        phase_02_train_opt_runtime: Name of the ClusterTrainingRuntime to use.
        phase_03_eval_opt_batch: Batch size for evaluation (auto or int).
        phase_03_eval_opt_gen_kwargs: Generation kwargs for evaluation.
        phase_03_eval_opt_limit: Limit examples per task (-1 = no limit).
        phase_03_eval_opt_log_samples: Log individual evaluation samples.
        phase_03_eval_opt_model_args: Model initialization arguments.
        phase_03_eval_opt_verbosity: Logging verbosity (DEBUG, INFO, etc.).
        phase_04_registry_opt_description: Model description for registry.
        phase_04_registry_opt_format_name: Model format (pytorch, onnx).
        phase_04_registry_opt_format_version: Model format version.
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
        training_seed=phase_02_train_opt_seed,
        training_num_epochs=phase_02_train_man_epochs,
        # SFT optimizations
        training_use_liger=phase_02_train_opt_use_liger,
        # LR scheduler
        training_lr_scheduler=phase_02_train_opt_lr_scheduler,
        training_lr_warmup_steps=phase_02_train_opt_lr_warmup,
        # Saving (SFT-specific)
        training_checkpoint_at_epoch=phase_02_train_opt_save_epoch,
        training_save_samples=phase_02_train_opt_save_samples,
        training_accelerate_full_state_at_epoch=phase_02_train_opt_save_full_state,
        training_fsdp_sharding_strategy=phase_02_train_opt_fsdp_sharding,
        # Environment
        training_envs=phase_02_train_opt_env_vars,
        training_metadata_labels=phase_02_train_opt_labels,
        training_metadata_annotations=phase_02_train_opt_annotations,
        # Resources
        training_resource_cpu_per_worker=phase_02_train_opt_cpu,
        training_resource_gpu_per_worker=phase_02_train_man_gpu,
        training_resource_memory_per_worker=phase_02_train_opt_memory,
        training_resource_num_procs_per_worker=phase_02_train_opt_num_procs,
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
        task_names=phase_03_eval_man_tasks,
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
    model_registry_task = kubeflow_model_registry(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        input_model=training_task.outputs["output_model"],
        input_metrics=training_task.outputs["output_metrics"],
        eval_metrics=eval_task.outputs["output_metrics"],
        eval_results=eval_task.outputs["output_results"],
        registry_address=phase_04_registry_man_address,
        registry_port=phase_04_registry_opt_port,
        model_name=phase_04_registry_man_name,
        model_version=phase_04_registry_man_version,
        model_format_name=phase_04_registry_opt_format_name,
        model_format_version=phase_04_registry_opt_format_version,
        model_description=phase_04_registry_opt_description,
        author=phase_04_registry_man_author,
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
        pipeline_func=sft_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
