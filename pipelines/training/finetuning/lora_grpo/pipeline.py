"""LoRA GRPO (Group Relative Policy Optimization) Training Pipeline.

A 4-stage pipeline for LoRA GRPO fine-tuning with promotion gating:
1. Dataset Preparation and LoRA GRPO Training - resolves the input source,
   then ART runs a 1 GPU TrainJob and merges LoRA adapters
2. GRPO Evaluation - extracts reward metrics, computes promotion gate
3. Model Registry - registers promoted model (gated by promotion check)
4. Model Deployment - deploys via KServe with vLLM (gated by promotion check)

Uses the Kubeflow Trainer with ART backend on a single GPU TrainJob.
Requires a user-provided ReadWriteMany PVC for persistent model storage
and serving.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.prepare_grpo_dataset import prepare_grpo_dataset
from components.deployment.kubeflow_model_registry import (
    kubeflow_model_registry as shared_model_registry,
)
from components.deployment.model_deployment import model_deployment
from components.evaluation.grpo_eval import grpo_eval
from components.training.finetuning.lora_grpo import train_model

# =============================================================================
# Pipeline Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_MOUNT_PATH = "/mnt/pipeline"
PIPELINE_NAME = "lora-grpo-pipeline"

# Temporary workaround: pin the registry SDK for this pipeline until the shared
# Model Registry component is updated to model-registry==0.3.9. Remove this
# override once that shared component carries the updated dependency.
GRPO_MODEL_REGISTRY = dsl.component(
    base_image="quay.io/opendatahub/odh-th06-cpu-torch291-py312:odh-3.4",
    packages_to_install=["model-registry==0.3.9"],
)(shared_model_registry.python_func)
# =============================================================================


@dsl.component(
    base_image="quay.io/opendatahub/odh-th-torch-cpu-py312:odh-3.6-ea.2",
    install_kfp_package=False,
)
def validate_replica_settings(min_replicas: int, max_replicas: int) -> None:
    """Validate serving replica bounds before expensive pipeline stages run."""
    if min_replicas < 1 or max_replicas < min_replicas:
        raise ValueError("replica settings must satisfy 1 <= min_replicas <= max_replicas")


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "LoRA GRPO pipeline: RLVR (Reinforcement Learning from Verifiable Rewards) "
        "using ART backend on a single GPU TrainJob"
    ),
)
def lora_grpo_pipeline(
    # =========================================================================
    # REQUIRED PARAMETERS (no defaults - must be supplied at run time)
    # =========================================================================
    p01_persistent_pvc: str,
    p02_dataset_uri: str,
    p03_training_model: str,
    p04_model_registry_address: str,
    p05_serving_namespace: str,
    p06_tokenizer_mode: str,
    # Optional inputs, ordered by pipeline stage.
    p07_dataset_max_invalid_row_ratio: float = 0.10,
    p08_dataset_data_connection_name: str = "s3-secret",
    p09_training_num_iterations: int = 50,
    p10_training_group_size: int = 4,
    p11_training_prompt_batch_size: int = 50,
    p12_training_lora_r: int = 16,
    p13_training_lora_alpha: int = 16,
    p14_training_n_train: int = 200,
    p15_training_completion_timeout_seconds: int = 7200,
    p16_training_learning_rate: float = 1e-5,
    p17_training_gpu_memory_utilization: float = 0.45,
    p18_training_enforce_eager: bool = True,
    p19_training_env_vars: str = "",
    p20_model_registry_name: str = "grpo-model",
    p21_model_registry_version: str = "1.0.0",
    p22_model_registry_port: int = 8080,
    p23_serving_gpu_count: int = 1,
    p24_serving_max_model_len: int = 4096,
    p25_serving_max_replicas: int = 1,
    p26_serving_cpu_requests: str = "2",
    p27_serving_memory_requests: str = "8Gi",
    # Full pipeline controls not exposed by the minimal pipeline.
    p28_training_data_config: str = "",
    p29_training_cpu: str = "4",
    p30_training_gpu: int = 1,
    p31_training_memory: str = "64Gi",
    p32_training_labels: str = "",
    p33_training_annotations: str = "",
    p34_training_runtime: str = "training-hub",
    p35_model_registry_author: str = "pipeline",
    p36_model_registry_description: str = "",
    p37_model_registry_format_name: str = "pytorch",
    p38_model_registry_format_version: str = "1.0",
    p39_serving_min_replicas: int = 1,
):
    """LoRA GRPO Training Pipeline - RLVR fine-tuning with promotion gating.

    RLVR means Reinforcement Learning from Verifiable Rewards.

    A 4-stage pipeline that prepares and trains a language model using LoRA
    GRPO (Group Relative Policy Optimization) for tool-calling tasks, evaluates
    training results, and conditionally registers and deploys the model only if
    training shows improvement.

    Prerequisites:
        - A ReadWriteMany PVC (e.g., NFS-backed) for model storage and serving.
          Create with: ``oc create pvc <name> --access-mode=ReadWriteMany
          --storage-class=nfs-csi --size=50Gi``
        - ``kubernetes-credentials`` secret with KUBERNETES_SERVER_URL and
          KUBERNETES_AUTH_TOKEN (required for TrainJob creation).
        - ``hf-token`` secret (optional, for gated HuggingFace models/datasets).
        - RHOAI data connection Secret (only required for private S3 datasets;
          defaults to ``s3-secret`` and is injected optionally).
        - Model Registry RoleBinding for the pipeline ServiceAccount (required
          for stage 4).
        - Pipeline ServiceAccount RBAC for KServe CRDs (required for stage 5).

    Args:
        p01_persistent_pvc: (Required) Name of the ReadWriteMany PVC used for
            training checkpoints and model serving. To create one: in RHOAI open
            your Data Science Project -> Cluster Storage -> Create storage, select
            ReadWriteMany access mode, minimum 50Gi. Enter only the PVC name (not
            a path). Example: ``grpo-pipeline-pvc``
        p02_dataset_uri: (Required) Dataset reference.
            Supported sources:
            • Hugging Face: ``hf://org/dataset:config``. References are loaded
              directly by Training Hub.
            • Local/PVC: ``pvc://data.jsonl``
            • S3: ``s3://bucket/path/data.jsonl``
            • HTTP(S): ``https://example.com/data.jsonl``
            Local JSON/JSONL, PVC, S3, and HTTP(S) sources are normalized by
            the preparation component onto the shared PVC. Non-Hugging Face
            rows are converted to Training Hub's canonical ``messages`` and
            ``tools`` format. See the [Training Hub loader
            format](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/lora_grpo.py#L381-L394).
        p03_training_model: (Required) HuggingFace model ID of the base model to
            fine-tune. If the model is gated, configure the ``hf-token`` secret.
            Examples: ``Qwen/Qwen3-4B`` or
            ``ibm-granite/granite-3.3-8b-instruct``
        p04_model_registry_address: (Required) Hostname or IP address of the
            Model Registry service. Pass only the host or IP, without
            ``http://``, ``https://``, or the ``/api/model_registry/v1alpha3``
            path.
        p05_serving_namespace: (Required) OpenShift project where the KServe
            InferenceService will be created. Must be the same namespace where
            your PVC is provisioned. Should have KServe enabled - typically your
            Data Science Project. Example: ``my-ds-project``
        p06_tokenizer_mode: (Required) vLLM tokenizer mode. Use ``auto`` when
            the fast tokenizer is compatible, ``slow`` when the model requires
            its HuggingFace slow tokenizer (for example, Qwen3 in the tested
            environment), or ``mistral`` for models that require the
            ``mistral_common`` tokenizer. Do not leave it empty.
        p07_dataset_max_invalid_row_ratio: Maximum fraction of invalid
            rows that the preparation component may skip for local, PVC, S3,
            and HTTP(S) sources before failing. Defaults to 10% (``0.10``).
            Enter a fraction between ``0.0`` and ``1.0``; for example, enter
            ``0.50`` for 50%. Use ``0.0`` for strict validation. It is not
            applied to direct Hugging Face loading.
        p08_dataset_data_connection_name: Name of the RHOAI data connection
            Secret used by the preparation task for private S3 input. It
            defaults to ``s3-secret``. It is optional for Hugging Face, HTTP(S),
            and local/PVC input. The Secret should provide
            ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and
            ``AWS_DEFAULT_REGION``; it may provide ``AWS_S3_ENDPOINT``.
        p09_training_num_iterations: Number of GRPO policy-gradient update
            steps. Each iteration generates ``group_size`` rollouts per prompt
            and updates the LoRA adapter. The promotion gate (stage 3) requires
            ``final_reward > initial_reward``. Increase the completion timeout
            when using higher iteration counts, larger models, or slower
            hardware.
        p10_training_group_size: Rollouts generated per prompt to estimate
            relative reward advantage. Higher = better gradient signal but more
            memory and time. Default 4 is suitable for single-GPU runs.
        p11_training_prompt_batch_size: Prompts processed per GRPO update step.
            Total rollout batch = ``prompt_batch_size x group_size`` (e.g., 50 x
            4 = 200). Reduce if you hit out-of-memory errors. Example: ``50``
        p12_training_lora_r: LoRA adapter rank. Higher = more adapter capacity
            but slower training and more memory. Typical values: 8 (lightweight),
            16 (balanced), 32 (high capacity). Example: ``16``
        p13_training_lora_alpha: LoRA scaling factor. Effective weight scale =
            ``lora_alpha / lora_r``. Set equal to ``lora_r`` for neutral scaling
            (recommended). Example: ``16``
        p14_training_n_train: Maximum rows ART draws from the dataset per run.
            This is independent of the total dataset size. Example: ``200`` for
            quick tests, ``2000`` for a fuller run.
        p15_training_completion_timeout_seconds: Maximum number of seconds to
            wait for the TrainJob to reach Complete or Failed after it starts.
            The wait returns immediately when the TrainJob completes. Set this
            to a larger value for longer runs or slower hardware, or a smaller
            value for smoke tests. The TrainJob startup wait remains unchanged.
        p16_training_learning_rate: AdamW learning rate for LoRA adapter
            updates. ``1e-5`` is a safe default; increase to ``5e-5`` if reward
            improvement is too slow.
        p17_training_gpu_memory_utilization: Fraction of GPU VRAM given to the
            vLLM rollout engine. Enter a fraction between ``0.0`` and ``1.0``;
            for example, enter ``0.50`` for 50%. The remainder goes to the
            policy model and optimizer. Default 0.45 is tuned for Qwen3-4B on a
            80GB A100. Reduce to 0.35 if you see OOM during rollout generation.
        p18_training_enforce_eager: Disables torch.compile and CUDA graph
            capture in vLLM. Must be True for Qwen3 models (known CUDAGraph
            incompatibility). Set False for other models for faster rollouts.
        p19_training_env_vars: Comma-separated ``KEY=VALUE`` environment
            variables injected into the training pod. Leave empty if not needed.
            Example: ``NCCL_DEBUG=INFO,TORCH_DISTRIBUTED_DEBUG=DETAIL``
        p20_model_registry_name: Name to register the model under. Must be
            unique within your Model Registry. Defaults to ``grpo-model``;
            override it with a unique name such as ``qwen3-grpo-tool-call`` for
            repeated or production runs.
        p21_model_registry_version: Version tag for this model revision. Use
            semantic versioning. Example: ``1.0.0``
        p22_model_registry_port: Port of the Model Registry service. Default
            8080 matches a typical internal HTTP registry.
        p23_serving_gpu_count: GPUs allocated to the KServe vLLM predictor pod.
            1 GPU is sufficient for Qwen3-4B. Example: ``1``
        p24_serving_max_model_len: Maximum total token length (prompt +
            completion) vLLM will handle. Longer contexts use more VRAM. Default
            4096 is adequate for tool-call traces. Increase to 8192 for longer
            prompts.
        p25_serving_max_replicas: Maximum predictor replicas for autoscaling.
            Must be greater than or equal to ``p39_serving_min_replicas``.
            Set equal to it to disable scaling. Example: ``1``
        p26_serving_cpu_requests: CPU cores requested and limited for the KServe
            predictor pod. Example: ``2``
        p27_serving_memory_requests: RAM requested and limited for the KServe
            predictor pod. Example: ``8Gi``
        p28_training_data_config: Optional Hugging Face dataset configuration
            name. Set it when a multi-configuration dataset does not include the
            configuration in its URI, for example ``Qwen3`` with
            ``hf://Agent-Ark/Toucan-1.5M``. Leave it empty when the URI already
            includes ``:config``, when the dataset has one configuration, or for
            local/PVC, S3, and HTTP(S) sources. Split selection is not supported;
            do not use this parameter to specify a split name.
        p29_training_cpu: CPU cores requested for the TrainJob worker pod.
            Example: ``4``
        p30_training_gpu: GPUs per TrainJob worker. ART requires exactly 1 GPU
            per worker - do not change this.
        p31_training_memory: RAM requested for the TrainJob worker pod. For
            Qwen3-4B on an 80GB GPU, 64Gi is sufficient. Example: ``64Gi``
        p32_training_labels: Comma-separated ``key=value`` labels applied to
            the TrainJob worker pod for cost tracking or scheduling constraints.
            Leave empty if not needed. Example: ``team=ml,project=grpo``
        p33_training_annotations: Comma-separated ``key=value`` annotations
            applied to the TrainJob worker pod. Leave empty if not needed.
        p34_training_runtime: Name of the ClusterTrainingRuntime CR that defines
            the base container image. Check available runtimes with
            ``oc get clustertrainingruntimes``. Default: ``training-hub``
        p35_model_registry_author: Author name attached to the registry entry.
            Example: ``your-team-name``
        p36_model_registry_description: Free-text description stored with the
            registered model. Example: ``Qwen3-4B fine-tuned with GRPO for
            tool-calling tasks``
        p37_model_registry_format_name: Model serialization format in the
            registry. Use ``pytorch`` for vLLM-served models. Example: ``pytorch``
        p38_model_registry_format_version: Version of the model format. Use
            ``1.0`` for PyTorch. Example: ``1.0``
        p39_serving_min_replicas: Minimum live KServe predictor replicas. Must
            be at least 1 and no greater than ``p25_serving_max_replicas``. Set
            to 1 to keep the endpoint always warm. Example: ``1``
    """

    def configure_training_task(training_task):
        """Apply shared runtime configuration to the training task."""
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
        kfp.kubernetes.use_secret_as_env(
            task=training_task,
            secret_name="hf-token",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
            optional=True,
        )
        kfp.kubernetes.mount_pvc(
            task=training_task,
            pvc_name=p01_persistent_pvc,
            mount_path=PVC_MOUNT_PATH,
        )
        return training_task

    validate_replicas_task = validate_replica_settings(
        min_replicas=p39_serving_min_replicas,
        max_replicas=p25_serving_max_replicas,
    )
    validate_replicas_task.set_caching_options(False)

    # =====================================================================
    # Stage 1: LoRA GRPO Training
    # =====================================================================
    prepare_dataset_task = prepare_grpo_dataset(
        dataset_uri=p02_dataset_uri,
        pvc_mount_path=PVC_MOUNT_PATH,
        dataset_config=p28_training_data_config,
        max_invalid_row_ratio=p07_dataset_max_invalid_row_ratio,
    )
    prepare_dataset_task.after(validate_replicas_task)
    prepare_dataset_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(prepare_dataset_task, "IfNotPresent")
    kfp.kubernetes.mount_pvc(
        task=prepare_dataset_task,
        pvc_name=p01_persistent_pvc,
        mount_path=PVC_MOUNT_PATH,
    )
    # KFP attaches this secret reference when the pipeline is compiled. Keep
    # it optional so HF, HTTP(S), and local/PVC inputs do not require a data
    # connection; S3 inputs use the injected values when the Secret exists.
    kfp.kubernetes.use_secret_as_env(
        task=prepare_dataset_task,
        secret_name=p08_dataset_data_connection_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
        optional=True,
    )

    training_task = configure_training_task(
        train_model(
            pvc_path=PVC_MOUNT_PATH,
            training_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            training_base_model=p03_training_model,
            training_data_path=prepare_dataset_task.outputs["resolved_data_path"],
            training_num_iterations=p09_training_num_iterations,
            training_group_size=p10_training_group_size,
            training_prompt_batch_size=p11_training_prompt_batch_size,
            training_n_train=p14_training_n_train,
            training_completion_timeout_seconds=p15_training_completion_timeout_seconds,
            training_learning_rate=p16_training_learning_rate,
            training_gpu_memory_utilization=p17_training_gpu_memory_utilization,
            training_enforce_eager=p18_training_enforce_eager,
            training_data_config=prepare_dataset_task.outputs["resolved_data_config"],
            training_lora_r=p12_training_lora_r,
            training_lora_alpha=p13_training_lora_alpha,
            training_envs=p19_training_env_vars,
            training_resource_cpu_per_worker=p29_training_cpu,
            training_resource_gpu_per_worker=p30_training_gpu,
            training_resource_memory_per_worker=p31_training_memory,
            training_metadata_labels=p32_training_labels,
            training_metadata_annotations=p33_training_annotations,
            training_runtime=p34_training_runtime,
            training_pvc_name=p01_persistent_pvc,
            training_use_hf_direct=prepare_dataset_task.outputs["use_hf_direct"],
        )
    )
    training_task.after(prepare_dataset_task)

    training_results_path = training_task.outputs["Output"]
    training_model = training_task.outputs["output_model"]
    training_metrics = training_task.outputs["output_metrics"]

    # =========================================================================
    # Stage 3: GRPO Evaluation
    # =========================================================================
    grpo_eval_task = grpo_eval(training_results_path=training_results_path)
    grpo_eval_task.after(training_task)
    grpo_eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(grpo_eval_task, "IfNotPresent")
    kfp.kubernetes.mount_pvc(
        task=grpo_eval_task,
        pvc_name=p01_persistent_pvc,
        mount_path=PVC_MOUNT_PATH,
    )

    # =========================================================================
    # Stages 4 + 5 run only when evaluation passes the promotion gate.
    # =========================================================================
    with dsl.If(
        grpo_eval_task.outputs["promotion_passed"] == True,  # noqa: E712 - KFP dsl.If requires explicit == comparison
        name="promotion-gate",
    ):
        # Stage 4: Model Registry
        registry_task = GRPO_MODEL_REGISTRY(
            pvc_mount_path=PVC_MOUNT_PATH,
            model_pvc_name=p01_persistent_pvc,
            input_model=training_model,
            input_metrics=training_metrics,
            eval_metrics=grpo_eval_task.outputs["output_metrics"],
            eval_results=grpo_eval_task.outputs["output_reward_chart"],
            registry_address=p04_model_registry_address,
            registry_port=p22_model_registry_port,
            model_name=p20_model_registry_name,
            model_version=p21_model_registry_version,
            model_format_name=p37_model_registry_format_name,
            model_format_version=p38_model_registry_format_version,
            model_description=p36_model_registry_description,
            author=p35_model_registry_author,
            shared_log_file="pipeline_log.txt",
            source_pipeline_name=PIPELINE_NAME,
            source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            source_namespace="",
        )
        registry_task.set_caching_options(False)
        kfp.kubernetes.set_image_pull_policy(registry_task, "IfNotPresent")
        kfp.kubernetes.mount_pvc(
            task=registry_task,
            pvc_name=p01_persistent_pvc,
            mount_path=PVC_MOUNT_PATH,
        )

        # Stage 5: Model Deployment
        deploy_task = model_deployment(
            model_name=p03_training_model,
            namespace=p05_serving_namespace,
            model_dir="final_model",
            model_cache_pvc=p01_persistent_pvc,
            model_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            gpu_count=p23_serving_gpu_count,
            max_model_len=p24_serving_max_model_len,
            min_replicas=p39_serving_min_replicas,
            max_replicas=p25_serving_max_replicas,
            cpu_requests=p26_serving_cpu_requests,
            memory_requests=p27_serving_memory_requests,
            cpu_limits=p26_serving_cpu_requests,
            memory_limits=p27_serving_memory_requests,
            tokenizer_mode=p06_tokenizer_mode,
        )
        deploy_task.after(registry_task)
        deploy_task.set_caching_options(False)
        kfp.kubernetes.set_image_pull_policy(deploy_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=lora_grpo_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
