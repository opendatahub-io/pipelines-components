"""LoRA (Low-Rank Adaptation) Training Pipeline — Eval Hub variant.

A 5-stage pipeline for parameter-efficient fine-tuning:
1. Dataset Download
2. OCI Model Resolution
3. LoRA Training (unsloth backend)
4. Evaluation via Eval Hub (KServe InferenceService for model serving)
5. Model Registry

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
from components.evaluation.evalhub.kserve import evalhub_evaluator_kserve
from components.training.finetuning.lora import train_model
from components.training.finetuning.oci_utils import (
    copy_oci_model_to_pvc,
    is_oci_uri,
    passthrough_uri,
)

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "50Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "lora-pipeline-evalhub"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="LoRA pipeline with Eval Hub evaluation via KServe, results optionally tracked in MLflow",
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
def lora_pipeline_evalhub(
    # =========================================================================
    # KEY PARAMETERS (Required/Important) - Sorted by step
    # =========================================================================
    phase_01_dataset_man_data_uri: str,
    phase_02_train_man_train_batch: int = 128,
    phase_02_train_man_train_epochs: int = 2,
    phase_02_train_man_train_gpu: int = 1,
    phase_02_train_man_train_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    phase_02_train_man_train_tokens: int = 32000,
    phase_02_train_man_lora_r: int = 16,
    phase_02_train_man_lora_alpha: int = 32,
    phase_03_eval_opt_evalhub_url: str = "",
    phase_03_eval_opt_collection: str = "",
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
    phase_02_train_opt_micro_batch_size: int = 2,
    phase_02_train_opt_grad_accum_steps: int = 1,
    phase_02_train_opt_flash_attention: bool = True,
    phase_02_train_opt_bf16: bool = True,
    phase_02_train_opt_fp16: bool = False,
    phase_02_train_opt_tf32: bool = True,
    phase_02_train_opt_save_steps: int = 500,
    phase_02_train_opt_eval_steps: int = 500,
    phase_02_train_opt_logging_steps: int = 10,
    phase_02_train_opt_save_total_limit: int = 3,
    phase_02_train_opt_wandb_project: str = "",
    phase_02_train_opt_wandb_entity: str = "",
    phase_02_train_opt_wandb_run_name: str = "",
    phase_02_train_opt_tensorboard_log_dir: str = "",
    phase_02_train_opt_dataset_type: str = "",
    phase_02_train_opt_field_messages: str = "",
    phase_02_train_opt_field_instruction: str = "",
    phase_02_train_opt_field_input: str = "",
    phase_02_train_opt_field_output: str = "",
    phase_02_train_opt_enable_model_splitting: bool = False,
    phase_02_train_opt_runtime: str = "training-hub",
    phase_03_eval_opt_benchmarks: list = [
        {"id": "leaderboard_ifeval", "provider_id": "lm_evaluation_harness"},
        {"id": "leaderboard_bbh", "provider_id": "lm_evaluation_harness"},
        {"id": "leaderboard_mmlu_pro", "provider_id": "lm_evaluation_harness"},
        {"id": "leaderboard_musr", "provider_id": "lm_evaluation_harness"},
        {"id": "leaderboard_math_hard", "provider_id": "lm_evaluation_harness"},
    ],
    phase_03_eval_opt_mlflow_experiment: str = "",
    phase_03_eval_opt_timeout: int = 7200,
    phase_03_eval_opt_kserve_gpu_count: int = 1,
    phase_03_eval_opt_kserve_cpu: str = "2",
    phase_03_eval_opt_kserve_memory: str = "32Gi",
    phase_04_registry_opt_description: str = "",
    phase_04_registry_opt_format_name: str = "pytorch",
    phase_04_registry_opt_format_version: str = "1.0",
    phase_04_registry_opt_port: int = 8080,
):
    """LoRA Training Pipeline with Eval Hub evaluation (KServe).

    A 5-stage ML pipeline for fine-tuning language models with LoRA:

    1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
    2) OCI Model Resolution - Resolves OCI URIs to PVC paths or passes through HF URIs
    3) LoRA Training - Fine-tunes using unsloth backend (low-rank adapters)
    4) Evaluation - Evaluates via Eval Hub with a KServe InferenceService for
       model serving. Results optionally tracked in MLflow.
    5) Model Registry - Registers trained model to Kubeflow Model Registry

    Prerequisites: Eval Hub and KServe must be installed on the cluster.
    The pipeline ServiceAccount needs RBAC permissions for
    inferenceservices.serving.kserve.io and servingruntimes.serving.kserve.io
    resources (create, delete, get, list, patch). The workspace PVC must use
    ReadWriteMany access mode (NFS-backed) so the KServe predictor pod can
    mount the model. The eval component uses the in-cluster ServiceAccount
    token for K8s API access.

    Known limitations: Some HuggingFace datasets used by benchmarks require
    trust_remote_code=True. The 5 default leaderboard benchmarks (ifeval,
    bbh, mmlu_pro, musr, math_hard) work without it. For other benchmarks,
    a custom provider ConfigMap with HF_DATASETS_TRUST_REMOTE_CODE=1 must
    be configured. The base_model_name parameter is needed for tokenizer
    resolution since the served model is a local checkpoint.

    Args:
        phase_01_dataset_man_data_uri: Dataset location (hf://, s3://, https://).
        phase_01_dataset_opt_subset: Limit to first N examples (0 = all).
        phase_02_train_man_train_batch: Effective batch size (samples per optimizer step).
        phase_02_train_man_train_epochs: Number of training epochs.
        phase_02_train_man_train_gpu: GPUs per worker.
        phase_02_train_man_train_model: Base model (HuggingFace ID or path).
        phase_02_train_man_train_tokens: Max tokens per GPU (memory cap).
        phase_02_train_man_lora_r: Rank of the low-rank matrices (4, 8, 16, 32, 64).
        phase_02_train_man_lora_alpha: Scaling factor (typically 2x lora_r).
        phase_02_train_opt_annotations: K8s annotations (key=val,...).
        phase_02_train_opt_bf16: Use bfloat16 precision.
        phase_02_train_opt_cpu: CPU cores per worker.
        phase_02_train_opt_dataset_type: Dataset format type.
        phase_02_train_opt_enable_model_splitting: Enable model splitting
            across GPUs.
        phase_02_train_opt_env_vars: Env vars (KEY=VAL,...).
        phase_02_train_opt_eval_steps: Run evaluation every N steps.
        phase_02_train_opt_field_input: Field name for input in dataset.
        phase_02_train_opt_field_instruction: Field name for instruction.
        phase_02_train_opt_field_messages: Field name for messages in dataset.
        phase_02_train_opt_field_output: Field name for output in dataset.
        phase_02_train_opt_flash_attention: Enable flash attention.
        phase_02_train_opt_fp16: Use float16 precision.
        phase_02_train_opt_grad_accum_steps: Gradient accumulation steps.
        phase_02_train_opt_labels: K8s labels (key=val,...).
        phase_02_train_opt_learning_rate: Learning rate. 2e-4 for LoRA.
        phase_02_train_opt_logging_steps: Log metrics every N steps.
        phase_02_train_opt_lora_dropout: Dropout rate for LoRA layers.
        phase_02_train_opt_lora_load_in_4bit: Enable 4-bit quantization.
        phase_02_train_opt_lora_load_in_8bit: Enable 8-bit quantization.
        phase_02_train_opt_lora_sample_packing: Pack multiple samples.
        phase_02_train_opt_lora_target_modules: Modules to apply LoRA
            (empty=auto-detect).
        phase_02_train_opt_lora_use_dora: Use Weight-Decomposed LoRA (DoRA).
        phase_02_train_opt_lora_use_rslora: Use Rank-Stabilized LoRA.
        phase_02_train_opt_lr_scheduler: LR schedule (cosine, linear).
        phase_02_train_opt_lr_warmup: Warmup steps before full LR.
        phase_02_train_opt_max_seq_len: Max sequence length in tokens.
        phase_02_train_opt_memory: RAM per worker.
        phase_02_train_opt_micro_batch_size: Micro batch size per GPU.
        phase_02_train_opt_num_procs: Processes per worker (auto = per GPU).
        phase_02_train_opt_runtime: ClusterTrainingRuntime name.
        phase_02_train_opt_save_epoch: Save checkpoint at each epoch.
        phase_02_train_opt_save_steps: Save checkpoint every N steps.
        phase_02_train_opt_save_total_limit: Max checkpoints to keep.
        phase_02_train_opt_seed: Random seed for reproducibility.
        phase_02_train_opt_tensorboard_log_dir: TensorBoard log directory.
        phase_02_train_opt_tf32: Enable TF32 on Ampere+ GPUs.
        phase_02_train_opt_use_liger: Enable Liger kernel optimizations.
        phase_02_train_opt_wandb_entity: Weights & Biases entity/team.
        phase_02_train_opt_wandb_project: Weights & Biases project name.
        phase_02_train_opt_wandb_run_name: Weights & Biases run name.
        phase_03_eval_opt_evalhub_url: Eval Hub API endpoint URL
            (empty = skip evaluation).
        phase_03_eval_opt_collection: Eval Hub collection ID
            (overrides benchmarks list). Available: "leaderboard-v2",
            "safety-and-fairness-v1", "toxicity-and-ethical-principles".
        phase_03_eval_opt_benchmarks: Benchmarks to evaluate. Defaults to 5
            leaderboard benchmarks (ifeval, bbh, mmlu_pro, musr, math_hard)
            that work without HF token or custom providers.
        phase_03_eval_opt_mlflow_experiment: MLflow experiment name
            (non-empty = enable, empty = disabled).
        phase_03_eval_opt_timeout: Max seconds to wait for evaluation.
        phase_03_eval_opt_kserve_gpu_count: GPUs for the KServe predictor.
        phase_03_eval_opt_kserve_cpu: CPU for the KServe predictor.
        phase_03_eval_opt_kserve_memory: Pod memory for the KServe predictor.
        phase_04_registry_man_address: Model Registry address (empty = skip).
        phase_04_registry_man_reg_author: Author name for the registered model.
        phase_04_registry_man_reg_name: Model name in registry.
        phase_04_registry_man_reg_version: Semantic version.
        phase_04_registry_opt_description: Model description.
        phase_04_registry_opt_format_name: Model format (pytorch, onnx).
        phase_04_registry_opt_format_version: Model format version.
        phase_04_registry_opt_port: Model registry server port.
    """
    # =========================================================================
    # Stage 1: Dataset Download
    # =========================================================================
    dataset_download_task = dataset_download(
        dataset_uri=phase_01_dataset_man_data_uri,
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        train_split_ratio=1.0,
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
    # Stage 2: OCI Model Resolution
    # =========================================================================
    oci_check = is_oci_uri(uri=phase_02_train_man_train_model)
    oci_check.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(oci_check, "IfNotPresent")

    with dsl.If(oci_check.output == "true"):
        model_import = dsl.importer(
            artifact_uri=phase_02_train_man_train_model,
            artifact_class=dsl.Model,
            reimport=False,
        )
        oci_copy = copy_oci_model_to_pvc(
            model=model_import.output,
            pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        )
        oci_copy.set_caching_options(False)
        kfp.kubernetes.set_image_pull_policy(oci_copy, "IfNotPresent")

    with dsl.Else():
        hf_pass = passthrough_uri(value=phase_02_train_man_train_model)
        kfp.kubernetes.set_image_pull_policy(hf_pass, "IfNotPresent")

    resolved_model = dsl.OneOf(oci_copy.output, hf_pass.output)

    # =========================================================================
    # Stage 3: LoRA Training
    # =========================================================================
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset=dataset_download_task.outputs["train_dataset"],
        training_base_model=resolved_model,
        training_effective_batch_size=phase_02_train_man_train_batch,
        training_max_tokens_per_gpu=phase_02_train_man_train_tokens,
        training_max_seq_len=phase_02_train_opt_max_seq_len,
        training_learning_rate=phase_02_train_opt_learning_rate,
        training_seed=phase_02_train_opt_seed,
        training_num_epochs=phase_02_train_man_train_epochs,
        training_lora_r=phase_02_train_man_lora_r,
        training_lora_alpha=phase_02_train_man_lora_alpha,
        training_lora_dropout=phase_02_train_opt_lora_dropout,
        training_lora_target_modules=phase_02_train_opt_lora_target_modules,
        training_lora_use_rslora=phase_02_train_opt_lora_use_rslora,
        training_lora_use_dora=phase_02_train_opt_lora_use_dora,
        training_lora_load_in_4bit=phase_02_train_opt_lora_load_in_4bit,
        training_lora_load_in_8bit=phase_02_train_opt_lora_load_in_8bit,
        training_lora_sample_packing=phase_02_train_opt_lora_sample_packing,
        training_use_liger=phase_02_train_opt_use_liger,
        training_lr_scheduler=phase_02_train_opt_lr_scheduler,
        training_lr_warmup_steps=phase_02_train_opt_lr_warmup,
        training_checkpoint_at_epoch=phase_02_train_opt_save_epoch,
        training_envs=phase_02_train_opt_env_vars,
        training_metadata_labels=phase_02_train_opt_labels,
        training_metadata_annotations=phase_02_train_opt_annotations,
        training_resource_cpu_per_worker=phase_02_train_opt_cpu,
        training_resource_gpu_per_worker=phase_02_train_man_train_gpu,
        training_resource_memory_per_worker=phase_02_train_opt_memory,
        training_resource_num_procs_per_worker=phase_02_train_opt_num_procs,
        training_resource_num_workers=1,
        training_micro_batch_size=phase_02_train_opt_micro_batch_size,
        training_gradient_accumulation_steps=phase_02_train_opt_grad_accum_steps,
        training_flash_attention=phase_02_train_opt_flash_attention,
        training_bf16=phase_02_train_opt_bf16,
        training_fp16=phase_02_train_opt_fp16,
        training_tf32=phase_02_train_opt_tf32,
        training_save_steps=phase_02_train_opt_save_steps,
        training_eval_steps=phase_02_train_opt_eval_steps,
        training_logging_steps=phase_02_train_opt_logging_steps,
        training_save_total_limit=phase_02_train_opt_save_total_limit,
        training_wandb_project=phase_02_train_opt_wandb_project,
        training_wandb_entity=phase_02_train_opt_wandb_entity,
        training_wandb_run_name=phase_02_train_opt_wandb_run_name,
        training_tensorboard_log_dir=phase_02_train_opt_tensorboard_log_dir,
        training_dataset_type=phase_02_train_opt_dataset_type,
        training_field_messages=phase_02_train_opt_field_messages,
        training_field_instruction=phase_02_train_opt_field_instruction,
        training_field_input=phase_02_train_opt_field_input,
        training_field_output=phase_02_train_opt_field_output,
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

    # =========================================================================
    # Stage 4: Evaluation via Eval Hub (KServe)
    # =========================================================================
    eval_task = evalhub_evaluator_kserve(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        model_artifact=training_task.outputs["output_model"],
        evalhub_url=phase_03_eval_opt_evalhub_url,
        collection_id=phase_03_eval_opt_collection,
        benchmarks=phase_03_eval_opt_benchmarks,
        evalhub_model_name="finetuned-model",
        base_model_name=resolved_model,
        evalhub_job_name="lora-pipeline-eval",
        evalhub_timeout=phase_03_eval_opt_timeout,
        evalhub_poll_interval=30,
        mlflow_experiment_name=phase_03_eval_opt_mlflow_experiment,
        gpu_count=phase_03_eval_opt_kserve_gpu_count,
        memory=phase_03_eval_opt_kserve_memory,
        cpu=phase_03_eval_opt_kserve_cpu,
    )
    eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_task, "IfNotPresent")

    for _task in [dataset_download_task, training_task, eval_task]:
        kfp.kubernetes.use_secret_as_env(
            task=_task,
            secret_name="hf-token",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
            optional=True,
        )

    # =========================================================================
    # Stage 5: Model Registry
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
        pipeline_func=lora_pipeline_evalhub,
        package_path=__file__.replace(".py", ".yaml"),
    )
