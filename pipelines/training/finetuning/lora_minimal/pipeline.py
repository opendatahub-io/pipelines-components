"""LoRA Minimal (Low-Rank Adaptation) Training Pipeline.

A minimal 4-stage pipeline for parameter-efficient fine-tuning:
1. Dataset Download
2. LoRA Training (unsloth backend)
3. Model Serving + EvalHub Evaluation
4. Model Registry

LoRA enables efficient fine-tuning by training low-rank adapter matrices
instead of full model weights, dramatically reducing compute and memory.
This minimal version provides a streamlined workflow for quick testing
and development.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

# Import reusable components
from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.evaluation.evalhub_eval import evalhub_evaluate
from components.training.finetuning_algorithms.lora import train_model

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "50Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "lora-minimal-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="LoRA Minimal pipeline: parameter-efficient fine-tuning using unsloth backend",
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
def lora_minimal_pipeline(
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
    phase_03_eval_man_evalhub_url: str = "",
    phase_03_eval_man_namespace: str = "",
    phase_04_registry_man_address: str = "",
    phase_04_registry_man_reg_name: str = "lora-model",
    phase_04_registry_man_reg_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by step
    # =========================================================================
    phase_01_dataset_opt_subset: int = 0,
    phase_02_train_opt_learning_rate: float = 2e-4,
    phase_02_train_opt_max_seq_len: int = 8192,
    phase_02_train_opt_use_liger: bool = True,
    phase_02_train_opt_lora_dropout: float = 0.0,
    phase_02_train_opt_lora_target_modules: str = "",
    phase_02_train_opt_lora_load_in_4bit: bool = True,
    phase_02_train_opt_lora_load_in_8bit: bool = False,
    phase_02_train_opt_dataset_type: str = "",
    phase_02_train_opt_field_messages: str = "",
    phase_02_train_opt_field_instruction: str = "",
    phase_02_train_opt_field_input: str = "",
    phase_02_train_opt_field_output: str = "",
    phase_02_train_opt_runtime: str = "training-hub",
    phase_03_eval_opt_evalhub_tenant: str = "",
    phase_03_eval_opt_tokenizer: str = "",
    phase_04_registry_opt_port: int = 8080,
):
    """LoRA Minimal Training Pipeline - Parameter-efficient fine-tuning.

    A minimal 4-stage ML pipeline for fine-tuning language models with LoRA:

    1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
    2) LoRA Training - Fine-tunes using unsloth backend (low-rank adapters)
    3) Model Serving + EvalHub - Serves the trained model and runs EvalHub evaluation
    4) Model Registry - Registers trained model to Kubeflow Model Registry

    Args:
        phase_01_dataset_man_data_uri: [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url)
        phase_01_dataset_man_data_split: Train/eval split (0.9 = 90% train/10% eval, 1.0 = no split, all for training)
        phase_02_train_man_train_batch: Effective batch size (samples per optimizer step)
        phase_02_train_man_train_epochs: Number of training epochs. LoRA typically needs 2-3
        phase_02_train_man_train_gpu: GPUs per worker
        phase_02_train_man_train_model: Base model (HuggingFace ID or path)
        phase_02_train_man_train_tokens: Max tokens per GPU (memory cap). 32000 for LoRA
        phase_02_train_man_lora_r: [LoRA] Rank of the low-rank matrices (4, 8, 16, 32, 64)
        phase_02_train_man_lora_alpha: [LoRA] Scaling factor (typically 2x lora_r)
        phase_03_eval_man_evalhub_url: EvalHub API base URL (empty = use component default)
        phase_03_eval_man_namespace: Kubernetes namespace for model serving
        phase_04_registry_man_address: Model Registry address (empty = skip registration)
        phase_04_registry_man_reg_name: Model name in registry
        phase_04_registry_man_reg_version: Semantic version (major.minor.patch)
        phase_01_dataset_opt_subset: Limit to first N examples (0 = all)
        phase_02_train_opt_learning_rate: Learning rate. 2e-4 recommended for LoRA
        phase_02_train_opt_max_seq_len: Max sequence length in tokens
        phase_02_train_opt_use_liger: Enable Liger kernel optimizations
        phase_02_train_opt_lora_dropout: [LoRA] Dropout rate for LoRA layers
        phase_02_train_opt_lora_target_modules: [LoRA] Modules to apply LoRA (empty=auto-detect)
        phase_02_train_opt_lora_load_in_4bit: [QLoRA] Enable 4-bit quantization (cannot use with 8-bit)
        phase_02_train_opt_lora_load_in_8bit: [QLoRA] Enable 8-bit quantization (cannot use with 4-bit)
        phase_02_train_opt_dataset_type: Dataset format type (empty = chat template auto-detect)
        phase_02_train_opt_field_messages: Field name for messages column (chat-format datasets)
        phase_02_train_opt_field_instruction: Field name for instruction/question column
        phase_02_train_opt_field_input: Field name for input/context column
        phase_02_train_opt_field_output: Field name for output/answer column
        phase_02_train_opt_runtime: Name of the ClusterTrainingRuntime to use.
        phase_03_eval_opt_evalhub_tenant: EvalHub tenant namespace (required by EvalHub server).
        phase_03_eval_opt_tokenizer: HuggingFace tokenizer ID for evaluation (e.g. base model ID).
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
        training_seed=42,
        training_num_epochs=phase_02_train_man_train_epochs,
        # LoRA-specific parameters
        training_lora_r=phase_02_train_man_lora_r,
        training_lora_alpha=phase_02_train_man_lora_alpha,
        training_lora_dropout=phase_02_train_opt_lora_dropout,
        training_lora_target_modules=phase_02_train_opt_lora_target_modules,
        # QLoRA parameters
        training_lora_load_in_4bit=phase_02_train_opt_lora_load_in_4bit,
        training_lora_load_in_8bit=phase_02_train_opt_lora_load_in_8bit,
        # Dataset format
        training_dataset_type=phase_02_train_opt_dataset_type,
        training_field_messages=phase_02_train_opt_field_messages,
        training_field_instruction=phase_02_train_opt_field_instruction,
        training_field_input=phase_02_train_opt_field_input,
        training_field_output=phase_02_train_opt_field_output,
        # Optimizations
        training_use_liger=phase_02_train_opt_use_liger,
        # Learning rate scheduler
        training_lr_scheduler="cosine",
        training_lr_warmup_steps=0,
        # Saving
        training_checkpoint_at_epoch=True,
        # Resources
        training_resource_cpu_per_worker="4",
        training_resource_gpu_per_worker=phase_02_train_man_train_gpu,
        training_resource_memory_per_worker="32Gi",
        training_resource_num_procs_per_worker="auto",
        # TODO: LoRA (unsloth backend) only supports single-node training.
        # Hardcoded to 1 until unsloth/training_hub add multi-node LoRA support.
        training_resource_num_workers=1,
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
        optional=True,
    )

    kfp.kubernetes.use_secret_as_env(
        task=training_task,
        secret_name="oci-pull-secret-model-download",
        secret_key_to_env={"OCI_PULL_SECRET_MODEL_DOWNLOAD": "OCI_PULL_SECRET_MODEL_DOWNLOAD"},
        optional=True,
    )

    # =========================================================================
    # Stage 3: Serve + EvalHub evaluation + teardown
    # =========================================================================
    eval_task = evalhub_evaluate(
        model_id=phase_02_train_man_train_model,
        model_name=phase_04_registry_man_reg_name,
        namespace=phase_03_eval_man_namespace,
        evalhub_url=phase_03_eval_man_evalhub_url,
        tenant=phase_03_eval_opt_evalhub_tenant,
        tokenizer=phase_03_eval_opt_tokenizer,
        gpu_count=phase_02_train_man_train_gpu,
    )
    eval_task.set_caching_options(False)
    eval_task.after(training_task)

    # Attach shared Hugging Face token secret to tasks that need it
    for _task in [dataset_download_task, training_task]:
        kfp.kubernetes.use_secret_as_env(
            task=_task,
            secret_name="hf-token",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
            optional=True,
        )

    kfp.kubernetes.use_secret_as_env(
        task=eval_task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )
    kfp.kubernetes.use_secret_as_env(
        task=eval_task,
        secret_name="evalhub-auth",
        secret_key_to_env={"EVALHUB_TOKEN": "EVALHUB_TOKEN"},
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
        model_name=phase_04_registry_man_reg_name,
        model_version=phase_04_registry_man_reg_version,
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
        pipeline_func=lora_minimal_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
