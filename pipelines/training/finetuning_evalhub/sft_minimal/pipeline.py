"""SFT (Supervised Fine-Tuning) Training Pipeline — Eval Hub Minimal variant.

A 5-stage pipeline for standard supervised fine-tuning with minimal
evaluation configuration:
1. Dataset Download
2. OCI Model Resolution
3. SFT Training (instructlab-training backend)
4. Evaluation via Eval Hub (KServe InferenceService for model serving)
5. Model Registry
"""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.evaluation.evalhub.kserve import evalhub_evaluator_kserve
from components.training.finetuning.oci_utils import (
    copy_oci_model_to_pvc,
    is_oci_uri,
    passthrough_uri,
)
from components.training.finetuning.sft import train_model

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "50Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "sft-pipeline-evalhub-easy"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="SFT Eval Hub pipeline (Minimal): 5 leaderboard benchmarks, KServe serving, minimal config",
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
def sft_pipeline_evalhub_easy(
    # =========================================================================
    # KEY PARAMETERS (Required/Important) - Sorted by step
    # =========================================================================
    phase_01_dataset_man_data_uri: str,
    phase_02_train_man_train_batch: int = 128,
    phase_02_train_man_epochs: int = 1,
    phase_02_train_man_gpu: int = 1,
    phase_02_train_man_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    phase_02_train_man_tokens: int = 10000,
    phase_02_train_man_workers: int = 4,
    phase_03_eval_opt_evalhub_url: str = "",
    phase_03_eval_opt_mlflow_experiment: str = "",
    phase_03_eval_opt_kserve_gpu_count: int = 1,
    phase_03_eval_opt_kserve_cpu: str = "2",
    phase_03_eval_opt_kserve_memory: str = "32Gi",
    phase_03_eval_opt_timeout: int = 7200,
    phase_04_registry_man_address: str = "",
    phase_04_registry_man_author: str = "pipeline",
    phase_04_registry_man_name: str = "sft-model",
    phase_04_registry_man_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by step
    # =========================================================================
    phase_01_dataset_opt_subset: int = 0,
    phase_02_train_opt_env_vars: str = (
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000"
    ),
    phase_02_train_opt_learning_rate: float = 5e-6,
    phase_02_train_opt_max_seq_len: int = 8192,
    phase_02_train_opt_fsdp_sharding: str = "FULL_SHARD",
    phase_02_train_opt_use_liger: bool = False,
    phase_02_train_opt_runtime: str = "training-hub",
    phase_04_registry_opt_port: int = 8080,
):
    """SFT Training Pipeline — Eval Hub Minimal.

    Minimal-config variant that runs 5 leaderboard benchmarks (ifeval, bbh,
    mmlu_pro, musr, math_hard) via KServe model serving — all public datasets,
    no HF token required, no trust_remote_code issues.

    Prerequisites: Eval Hub and KServe must be installed on the cluster.
    The pipeline ServiceAccount needs RBAC permissions for
    inferenceservices.serving.kserve.io and servingruntimes.serving.kserve.io
    resources (create, delete, get, list, patch). The workspace PVC must use
    ReadWriteMany access mode (NFS-backed) so the KServe predictor pod can
    mount the model. The eval component uses the in-cluster ServiceAccount
    token for K8s API access.

    Args:
        phase_01_dataset_man_data_uri: Dataset location (hf://, s3://, https://).
        phase_01_dataset_opt_subset: Limit to first N examples (0 = all).
        phase_02_train_man_train_batch: Effective batch size per optimizer step.
        phase_02_train_man_epochs: Number of training epochs.
        phase_02_train_man_gpu: GPUs per worker.
        phase_02_train_man_model: Base model (HuggingFace ID or path).
        phase_02_train_man_tokens: Max tokens per GPU (memory cap).
        phase_02_train_man_workers: Number of training pods.
        phase_02_train_opt_env_vars: Environment variables (KEY=VAL,...).
        phase_02_train_opt_fsdp_sharding: FSDP strategy (FULL_SHARD,
            HYBRID_SHARD).
        phase_02_train_opt_learning_rate: Learning rate for training.
        phase_02_train_opt_max_seq_len: Maximum sequence length in tokens.
        phase_02_train_opt_runtime: ClusterTrainingRuntime name.
        phase_02_train_opt_use_liger: Enable Liger kernel optimizations.
        phase_03_eval_opt_evalhub_url: Eval Hub API endpoint URL
            (empty = skip evaluation).
        phase_03_eval_opt_mlflow_experiment: MLflow experiment name
            (non-empty = enable, empty = disabled).
        phase_03_eval_opt_kserve_gpu_count: GPUs for the KServe predictor.
        phase_03_eval_opt_kserve_cpu: CPU for the KServe predictor.
        phase_03_eval_opt_kserve_memory: Pod memory for the KServe predictor.
        phase_03_eval_opt_timeout: Max seconds to wait for evaluation.
        phase_04_registry_man_address: Model Registry address (empty = skip).
        phase_04_registry_man_author: Author name for the registered model.
        phase_04_registry_man_name: Model name in registry.
        phase_04_registry_man_version: Semantic version.
        phase_04_registry_opt_port: Model Registry server port.
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
    oci_check = is_oci_uri(uri=phase_02_train_man_model)
    oci_check.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(oci_check, "IfNotPresent")

    with dsl.If(oci_check.output == "true"):
        model_import = dsl.importer(
            artifact_uri=phase_02_train_man_model,
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
        hf_pass = passthrough_uri(value=phase_02_train_man_model)
        kfp.kubernetes.set_image_pull_policy(hf_pass, "IfNotPresent")

    resolved_model = dsl.OneOf(oci_copy.output, hf_pass.output)

    # =========================================================================
    # Stage 3: SFT Training
    # =========================================================================
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset=dataset_download_task.outputs["train_dataset"],
        training_base_model=resolved_model,
        training_effective_batch_size=phase_02_train_man_train_batch,
        training_max_tokens_per_gpu=phase_02_train_man_tokens,
        training_max_seq_len=phase_02_train_opt_max_seq_len,
        training_learning_rate=phase_02_train_opt_learning_rate,
        training_seed=42,
        training_num_epochs=phase_02_train_man_epochs,
        training_use_liger=phase_02_train_opt_use_liger,
        training_lr_scheduler="cosine",
        training_lr_warmup_steps=0,
        training_checkpoint_at_epoch=True,
        training_save_samples=0,
        training_accelerate_full_state_at_epoch=False,
        training_fsdp_sharding_strategy=phase_02_train_opt_fsdp_sharding,
        training_envs=phase_02_train_opt_env_vars,
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
    # Stage 4: Evaluation via Eval Hub (KServe, hardcoded safe defaults)
    # =========================================================================
    eval_task = evalhub_evaluator_kserve(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        model_artifact=training_task.outputs["output_model"],
        evalhub_url=phase_03_eval_opt_evalhub_url,
        benchmarks=[
            {"id": "leaderboard_ifeval", "provider_id": "lm_evaluation_harness"},
            {"id": "leaderboard_bbh", "provider_id": "lm_evaluation_harness"},
            {"id": "leaderboard_mmlu_pro", "provider_id": "lm_evaluation_harness"},
            {"id": "leaderboard_musr", "provider_id": "lm_evaluation_harness"},
            {"id": "leaderboard_math_hard", "provider_id": "lm_evaluation_harness"},
        ],
        evalhub_model_name="finetuned-model",
        base_model_name=resolved_model,
        evalhub_job_name="sft-pipeline-eval",
        evalhub_timeout=phase_03_eval_opt_timeout,
        evalhub_poll_interval=30,
        mlflow_experiment_name=phase_03_eval_opt_mlflow_experiment,
        gpu_count=phase_03_eval_opt_kserve_gpu_count,
        memory=phase_03_eval_opt_kserve_memory,
        cpu=phase_03_eval_opt_kserve_cpu,
    )
    eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_task, "IfNotPresent")

    for _task in [dataset_download_task, training_task]:
        kfp.kubernetes.use_secret_as_env(
            task=_task,
            secret_name="hf-token",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
            optional=True,
        )

    # =========================================================================
    # Stage 5: Model Registry
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
        model_format_name="pytorch",
        model_format_version="1.0",
        model_description="",
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
        pipeline_func=sft_pipeline_evalhub_easy,
        package_path=__file__.replace(".py", ".yaml"),
    )
