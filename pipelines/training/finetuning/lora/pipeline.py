"""LoRA (Low-Rank Adaptation) Training Pipeline.

A 4-stage pipeline for parameter-efficient fine-tuning:
1. Dataset Download
2. LoRA Training (unsloth backend)
3. Evaluation with lm-eval
4. Model Registry

LoRA enables efficient fine-tuning by training low-rank adapter matrices
instead of full model weights, dramatically reducing compute and memory.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import (
    kubeflow_model_registry as model_registry,
)
from components.evaluation.lm_eval import universal_llm_evaluator
from components.training.finetuning.lora import train_model

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "50Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "lora-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="LoRA pipeline: parameter-efficient fine-tuning using unsloth backend",
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
def lora_pipeline(
    # =========================================================================
    # KEY PARAMETERS (Required/Important) - Sorted by step
    # =========================================================================
    phase_01_dataset_man_data_uri: str,
    phase_01_dataset_man_data_split: float = 0.9,
    phase_02_train_man_train_batch: int = 128,
    phase_02_train_man_train_epochs: int = 2,
    phase_02_train_man_train_gpu: int = 1,
    phase_02_train_man_train_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    phase_02_train_man_train_tokens: int = 32000,
    # TODO: LoRA (unsloth backend) only supports single-node training.
    # Uncomment when unsloth/training_hub add multi-node LoRA support.
    # phase_02_train_man_train_workers: int = 1,
    phase_02_train_man_lora_r: int = 16,
    phase_02_train_man_lora_alpha: int = 32,
    phase_03_eval_man_eval_tasks: list = ["arc_easy"],
    phase_04_registry_man_address: str = "",
    phase_04_registry_man_reg_author: str = "pipeline",
    phase_04_registry_man_reg_name: str = "lora-model",
    phase_04_registry_man_reg_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by step
    # =========================================================================
    phase_01_dataset_opt_subset: int = 0,
    phase_02_train_opt_annotations: str = "",
    phase_02_train_opt_cpu: str = "4",
    phase_02_train_opt_env_vars: str = "",
    phase_02_train_opt_labels: str = "",
    phase_02_train_opt_learning_rate: float = 2e-4,
    phase_02_train_opt_lr_scheduler: str = "cosine",
    phase_02_train_opt_lr_warmup: int = 0,
    phase_02_train_opt_max_seq_len: int = 8192,
    phase_02_train_opt_memory: str = "32Gi",
    phase_02_train_opt_num_procs: str = "auto",
    phase_02_train_opt_save_epoch: bool = False,
    phase_02_train_opt_seed: int = 42,
    phase_02_train_opt_use_liger: bool = True,
    phase_02_train_opt_lora_dropout: float = 0.0,
    phase_02_train_opt_lora_target_modules: str = "",
    phase_02_train_opt_lora_use_rslora: bool = False,
    phase_02_train_opt_lora_use_dora: bool = False,
    phase_02_train_opt_lora_load_in_4bit: bool = True,
    phase_02_train_opt_lora_load_in_8bit: bool = False,
    phase_02_train_opt_lora_sample_packing: bool = False,
    # Batch params
    phase_02_train_opt_micro_batch_size: int = 2,
    phase_02_train_opt_grad_accum_steps: int = 1,
    # Optimization params
    phase_02_train_opt_flash_attention: bool = True,
    phase_02_train_opt_bf16: bool = True,
    phase_02_train_opt_fp16: bool = False,
    phase_02_train_opt_tf32: bool = True,
    # Saving/Logging params
    phase_02_train_opt_save_steps: int = 500,
    phase_02_train_opt_eval_steps: int = 500,
    phase_02_train_opt_logging_steps: int = 10,
    phase_02_train_opt_save_total_limit: int = 3,
    # Logging integration params
    phase_02_train_opt_wandb_project: str = "",
    phase_02_train_opt_wandb_entity: str = "",
    phase_02_train_opt_wandb_run_name: str = "",
    phase_02_train_opt_tensorboard_log_dir: str = "",
    phase_02_train_opt_mlflow_tracking_uri: str = "",
    phase_02_train_opt_mlflow_experiment_name: str = "",
    phase_02_train_opt_mlflow_run_name: str = "",
    # Dataset format params
    phase_02_train_opt_dataset_type: str = "",
    phase_02_train_opt_field_messages: str = "",
    phase_02_train_opt_field_instruction: str = "",
    phase_02_train_opt_field_input: str = "",
    phase_02_train_opt_field_output: str = "",
    # Multi-GPU params
    phase_02_train_opt_enable_model_splitting: bool = False,
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
    """LoRA Training Pipeline - Parameter-efficient fine-tuning.

        A 4-stage ML pipeline for fine-tuning language models with LoRA:

        1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
        2) LoRA Training - Fine-tunes using unsloth backend (low-rank adapters)
        3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
        4) Model Registry - Registers trained model to Kubeflow Model Registry

    Args:
            phase_01_dataset_man_data_uri: [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url)
            phase_01_dataset_man_data_split: Train/eval split (0.9 = 90%/10%, 1.0 = no split)
            phase_02_train_man_train_batch: Effective batch size (samples per optimizer step)
            phase_02_train_man_train_epochs: Number of training epochs. LoRA typically needs 2-3
            phase_02_train_man_train_gpu: GPUs per worker
            phase_02_train_man_train_model: Base model (HuggingFace ID or path)
            phase_02_train_man_train_tokens: Max tokens per GPU (memory cap). 32000 for LoRA
            phase_02_train_man_lora_r: [LoRA] Rank of the low-rank matrices (4, 8, 16, 32, 64)
            phase_02_train_man_lora_alpha: [LoRA] Scaling factor (typically 2x lora_r)
            phase_03_eval_man_eval_tasks: lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.)
            phase_04_registry_man_address: Model Registry address (empty = skip registration)
            phase_04_registry_man_reg_author: Author name for the registered model
            phase_04_registry_man_reg_name: Model name in registry
            phase_04_registry_man_reg_version: Semantic version (major.minor.patch)
            phase_01_dataset_opt_subset: Limit to first N examples (0 = all)
            phase_02_train_opt_annotations: K8s annotations (key=val,...)
            phase_02_train_opt_cpu: CPU cores per worker
            phase_02_train_opt_env_vars: Env vars (KEY=VAL,...)
            phase_02_train_opt_labels: K8s labels (key=val,...)
            phase_02_train_opt_learning_rate: Learning rate. 2e-4 recommended for LoRA
            phase_02_train_opt_lr_scheduler: LR schedule (cosine, linear, constant)
            phase_02_train_opt_lr_warmup: Warmup steps before full LR
            phase_02_train_opt_max_seq_len: Max sequence length in tokens
            phase_02_train_opt_memory: RAM per worker
            phase_02_train_opt_num_procs: Processes per worker ('auto' = one per GPU)
            phase_02_train_opt_save_epoch: Save checkpoint at each epoch
            phase_02_train_opt_seed: Random seed for reproducibility
            phase_02_train_opt_use_liger: Enable Liger kernel optimizations
            phase_02_train_opt_lora_dropout: [LoRA] Dropout rate for LoRA layers
            phase_02_train_opt_lora_target_modules: [LoRA] Modules to apply LoRA (empty=auto-detect)
            phase_02_train_opt_lora_use_rslora: [LoRA] Use Rank-Stabilized LoRA
            phase_02_train_opt_lora_use_dora: [LoRA] Use Weight-Decomposed LoRA (DoRA)
            phase_02_train_opt_lora_load_in_4bit: [QLoRA] Enable 4-bit quantization (cannot use with 8-bit)
            phase_02_train_opt_lora_load_in_8bit: [QLoRA] Enable 8-bit quantization (cannot use with 4-bit)
            phase_02_train_opt_lora_sample_packing: [LoRA] Pack multiple samples for efficiency
            phase_02_train_opt_micro_batch_size: Micro batch size per GPU
            phase_02_train_opt_grad_accum_steps: Gradient accumulation steps
            phase_02_train_opt_flash_attention: Enable flash attention
            phase_02_train_opt_bf16: Use bfloat16 precision
            phase_02_train_opt_fp16: Use float16 precision
            phase_02_train_opt_tf32: Enable TF32 on Ampere+ GPUs
            phase_02_train_opt_save_steps: Save checkpoint every N steps
            phase_02_train_opt_eval_steps: Run evaluation every N steps
            phase_02_train_opt_logging_steps: Log metrics every N steps
            phase_02_train_opt_save_total_limit: Max checkpoints to keep
            phase_02_train_opt_wandb_project: Weights & Biases project name
            phase_02_train_opt_wandb_entity: Weights & Biases entity/team
            phase_02_train_opt_wandb_run_name: Weights & Biases run name
            phase_02_train_opt_tensorboard_log_dir: TensorBoard log directory
            phase_02_train_opt_mlflow_tracking_uri: MLflow tracking server URI
            phase_02_train_opt_mlflow_experiment_name: MLflow experiment name
            phase_02_train_opt_mlflow_run_name: MLflow run name
            phase_02_train_opt_dataset_type: Dataset format type
            phase_02_train_opt_field_messages: Field name for messages in dataset
            phase_02_train_opt_field_instruction: Field name for instruction in dataset
            phase_02_train_opt_field_input: Field name for input in dataset
            phase_02_train_opt_field_output: Field name for output in dataset
            phase_02_train_opt_enable_model_splitting: Enable model splitting across GPUs
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
    # Stage 2: LoRA Training
    # =========================================================================
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset=dataset_download_task.outputs["train_dataset"],
        # Model
        training_base_model=phase_02_train_man_train_model,
        # Hyperparameters
        training_effective_batch_size=phase_02_train_man_train_batch,
        training_max_tokens_per_gpu=phase_02_train_man_train_tokens,
        training_max_seq_len=phase_02_train_opt_max_seq_len,
        training_learning_rate=phase_02_train_opt_learning_rate,
        training_seed=phase_02_train_opt_seed,
        training_num_epochs=phase_02_train_man_train_epochs,
        # LoRA-specific parameters
        training_lora_r=phase_02_train_man_lora_r,
        training_lora_alpha=phase_02_train_man_lora_alpha,
        training_lora_dropout=phase_02_train_opt_lora_dropout,
        training_lora_target_modules=phase_02_train_opt_lora_target_modules,
        training_lora_use_rslora=phase_02_train_opt_lora_use_rslora,
        training_lora_use_dora=phase_02_train_opt_lora_use_dora,
        # QLoRA parameters
        training_lora_load_in_4bit=phase_02_train_opt_lora_load_in_4bit,
        training_lora_load_in_8bit=phase_02_train_opt_lora_load_in_8bit,
        training_lora_sample_packing=phase_02_train_opt_lora_sample_packing,
        # Optimizations
        training_use_liger=phase_02_train_opt_use_liger,
        # Learning rate scheduler
        training_lr_scheduler=phase_02_train_opt_lr_scheduler,
        training_lr_warmup_steps=phase_02_train_opt_lr_warmup,
        # Saving
        training_checkpoint_at_epoch=phase_02_train_opt_save_epoch,
        # Environment
        training_envs=phase_02_train_opt_env_vars,
        training_metadata_labels=phase_02_train_opt_labels,
        training_metadata_annotations=phase_02_train_opt_annotations,
        # Resources
        training_resource_cpu_per_worker=phase_02_train_opt_cpu,
        training_resource_gpu_per_worker=phase_02_train_man_train_gpu,
        training_resource_memory_per_worker=phase_02_train_opt_memory,
        training_resource_num_procs_per_worker=phase_02_train_opt_num_procs,
        # TODO: LoRA (unsloth backend) only supports single-node training.
        # Hardcoded to 1 until unsloth/training_hub add multi-node LoRA support.
        training_resource_num_workers=1,
        # Batch params
        training_micro_batch_size=phase_02_train_opt_micro_batch_size,
        training_gradient_accumulation_steps=phase_02_train_opt_grad_accum_steps,
        # Optimization params
        training_flash_attention=phase_02_train_opt_flash_attention,
        training_bf16=phase_02_train_opt_bf16,
        training_fp16=phase_02_train_opt_fp16,
        training_tf32=phase_02_train_opt_tf32,
        # Saving/Logging params
        training_save_steps=phase_02_train_opt_save_steps,
        training_eval_steps=phase_02_train_opt_eval_steps,
        training_logging_steps=phase_02_train_opt_logging_steps,
        training_save_total_limit=phase_02_train_opt_save_total_limit,
        # Logging integration params
        training_wandb_project=phase_02_train_opt_wandb_project,
        training_wandb_entity=phase_02_train_opt_wandb_entity,
        training_wandb_run_name=phase_02_train_opt_wandb_run_name,
        training_tensorboard_log_dir=phase_02_train_opt_tensorboard_log_dir,
        training_mlflow_tracking_uri=phase_02_train_opt_mlflow_tracking_uri,
        training_mlflow_experiment_name=phase_02_train_opt_mlflow_experiment_name,
        training_mlflow_run_name=phase_02_train_opt_mlflow_run_name,
        # Dataset format params
        training_dataset_type=phase_02_train_opt_dataset_type,
        training_field_messages=phase_02_train_opt_field_messages,
        training_field_instruction=phase_02_train_opt_field_instruction,
        training_field_input=phase_02_train_opt_field_input,
        training_field_output=phase_02_train_opt_field_output,
        # Multi-GPU params
        training_enable_model_splitting=phase_02_train_opt_enable_model_splitting,
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
        pipeline_func=lora_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
