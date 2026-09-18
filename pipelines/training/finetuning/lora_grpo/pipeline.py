"""LoRA GRPO (Group Relative Policy Optimization) Training Pipeline.

A 5-stage pipeline for LoRA GRPO fine-tuning with promotion gating:
1. Dataset Download - validates tool-call schema on CPU
2. LoRA GRPO Training - ART backend, 1 GPU TrainJob, merges LoRA adapters
3. GRPO Evaluation - extracts reward metrics, computes promotion gate
4. Model Registry - registers promoted model (gated by promotion check)
5. Model Deployment - deploys via KServe with vLLM (gated by promotion check)

Uses the Kubeflow Trainer with ART backend on a single GPU TrainJob.
Requires a user-provided ReadWriteMany PVC for persistent model storage
and serving.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.deployment.model_deployment import model_deployment
from components.evaluation.grpo_eval import grpo_eval
from components.training.finetuning.lora_grpo import train_model

# =============================================================================
# Pipeline Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_MOUNT_PATH = "/mnt/pipeline"
PIPELINE_NAME = "lora-grpo-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "LoRA GRPO pipeline: reinforcement learning from verifiable rewards using ART backend on a single GPU TrainJob"
    ),
)
def lora_grpo_pipeline(
    # =========================================================================
    # REQUIRED PARAMETERS (no defaults - must be supplied at run time)
    # =========================================================================
    phase_00_infra_man_pvc_name: str,
    phase_04_registry_man_address: str,
    phase_05_deploy_man_namespace: str,
    # =========================================================================
    # KEY PARAMETERS - Sorted by stage
    # =========================================================================
    # Stage 1: Dataset
    phase_01_dataset_man_data_uri: str = "hf://Agent-Ark/Toucan-1.5M:Qwen3",
    phase_01_dataset_man_data_split: float = 1.0,
    # Stage 2: Training
    phase_02_train_man_model: str = "Qwen/Qwen3-4B",
    phase_02_train_man_num_iterations: int = 50,
    phase_02_train_man_group_size: int = 4,
    phase_02_train_man_prompt_batch_size: int = 50,
    phase_02_train_man_lora_r: int = 16,
    phase_02_train_man_lora_alpha: int = 16,
    phase_04_registry_man_name: str = "grpo-model",
    phase_04_registry_man_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by stage
    # =========================================================================
    # Stage 1
    phase_01_dataset_opt_subset: int = 0,
    # Stage 2
    phase_02_train_opt_data_config: str = "Qwen3",
    phase_02_train_opt_n_train: int = 200,
    phase_02_train_opt_learning_rate: float = 1e-5,
    phase_02_train_opt_gpu_memory_utilization: float = 0.45,
    phase_02_train_opt_enforce_eager: bool = True,
    phase_02_train_opt_env_vars: str = "",
    phase_02_train_opt_cpu: str = "4",
    phase_02_train_opt_gpu: int = 1,
    phase_02_train_opt_memory: str = "64Gi",
    phase_02_train_opt_labels: str = "",
    phase_02_train_opt_annotations: str = "",
    phase_02_train_opt_runtime: str = "training-hub",
    # Stage 4
    phase_04_registry_opt_author: str = "pipeline",
    phase_04_registry_opt_description: str = "",
    phase_04_registry_opt_format_name: str = "pytorch",
    phase_04_registry_opt_format_version: str = "1.0",
    phase_04_registry_opt_port: int = 8080,
    # Stage 5
    phase_05_deploy_opt_gpu_count: int = 1,
    phase_05_deploy_opt_max_model_len: int = 4096,
    phase_05_deploy_opt_min_replicas: int = 1,
    phase_05_deploy_opt_max_replicas: int = 1,
    phase_05_deploy_opt_cpu_requests: str = "2",
    phase_05_deploy_opt_memory_requests: str = "8Gi",
):
    """LoRA GRPO Training Pipeline - RLVR fine-tuning with promotion gating.

    A 5-stage pipeline that trains a language model using LoRA GRPO
    (Group Relative Policy Optimization) for tool-calling tasks, evaluates
    training results, and conditionally registers and deploys the model
    only if training shows improvement.

    Prerequisites:
        - A ReadWriteMany PVC (e.g., NFS-backed) for model storage and serving.
          Create with: ``oc create pvc <name> --access-mode=ReadWriteMany
          --storage-class=nfs-csi --size=50Gi``
        - ``kubernetes-credentials`` secret with KUBERNETES_SERVER_URL and
          KUBERNETES_AUTH_TOKEN (required for TrainJob creation).
        - ``hf-token`` secret (optional, for gated HuggingFace models/datasets).
        - Pipeline ServiceAccount RBAC for KServe CRDs (required for stage 5).

    Args:
        phase_00_infra_man_pvc_name: Name of the ReadWriteMany PVC used for
            training checkpoints and model serving. To create one: in RHOAI open
            your Data Science Project -> Cluster Storage -> Create storage, select
            ReadWriteMany access mode, minimum 50Gi. Enter only the PVC name
            (not a path). Example: ``grpo-pipeline-pvc``
        phase_01_dataset_man_data_uri: URI of the dataset. Defaults to
            ``hf://Agent-Ark/Toucan-1.5M:Qwen3`` (a public tool-calling dataset
            used as a reference example) - override with your own dataset URI.
            Supported schemes:
            - HuggingFace: ``hf://org/dataset`` or ``hf://org/dataset:config``
              Some datasets have multiple subsets (configs). If the dataset has
              more than one config, you must append ``:config`` - otherwise
              ``load_dataset`` will fail asking you to choose. Single-config
              datasets do not need it.
              Example: ``hf://Agent-Ark/Toucan-1.5M:Qwen3``
            - S3: ``s3://bucket/path/to/file.jsonl``
            - HTTP: ``https://host/file.jsonl`` (must point to a raw file,
              not an HTML page)
            - PVC file: ``pvc://path/to/file.jsonl``
        phase_05_deploy_man_namespace: OpenShift project where the KServe
            InferenceService will be created. Must be the same namespace where
            your PVC is provisioned. Should have
            KServe enabled - typically your Data Science Project.
            Example: ``my-ds-project``
        phase_01_dataset_man_data_split: Fraction of the dataset used for
            training (remainder becomes eval). Default 1.0 uses all rows, which
            is correct for GRPO - improvement is measured from rollout rewards,
            not a held-out eval set.
        phase_02_train_man_model: HuggingFace model ID of the base model to
            fine-tune. If the model is gated, configure the ``hf-token`` secret.
            Example: ``Qwen/Qwen3-4B`` or ``ibm-granite/granite-3.3-8b-instruct``
        phase_02_train_man_num_iterations: Number of GRPO policy-gradient update
            steps. Each iteration generates ``group_size`` rollouts per prompt
            and updates the LoRA adapter. The promotion gate (stage 3) requires
            ``final_reward > initial_reward`` - fewer than 30 iterations rarely
            pass it. Default 50 runs in ~1-2 hours on one A100 80GB with
            Qwen3-4B. Use 100+ for a production-quality training run.
        phase_02_train_man_group_size: Rollouts generated per prompt to estimate
            relative reward advantage. Higher = better gradient signal but more
            memory and time. Default 4 is suitable for single-GPU runs.
        phase_02_train_man_prompt_batch_size: Prompts processed per GRPO update
            step. Total rollout batch = ``prompt_batch_size x group_size``
            (e.g., 50 x 4 = 200). Reduce if you hit out-of-memory errors.
            Example: ``50``
        phase_02_train_man_lora_r: LoRA adapter rank. Higher = more adapter
            capacity but slower training and more memory. Typical values: 8
            (lightweight), 16 (balanced), 32 (high capacity). Example: ``16``
        phase_02_train_man_lora_alpha: LoRA scaling factor. Effective weight
            scale = ``lora_alpha / lora_r``. Set equal to ``lora_r`` for neutral
            scaling (recommended). Example: ``16``
        phase_04_registry_man_address: Hostname of the Model Registry service
            (required). Find it in RHOAI Dashboard -> AI Hub -> Models ->
            Registry -> click View Details and copy the Server URL.
        phase_04_registry_man_name: Name to register the model under. Must be
            unique within your Model Registry.
            Example: ``qwen3-grpo-tool-call``
        phase_04_registry_man_version: Version tag for this model revision.
            Use semantic versioning. Example: ``1.0.0``
        phase_01_dataset_opt_subset: Download and use only this many rows
            (random sample). Set to 0 for all rows. Use a small number for
            quick pipeline smoke-tests. Example: ``500``
        phase_02_train_opt_data_config: HuggingFace dataset config (split or
            subset name). Only needed for multi-config datasets. Defaults to
            ``Qwen3`` to match the default dataset
            (``hf://Agent-Ark/Toucan-1.5M:Qwen3``). If you override
            ``phase_01_dataset_man_data_uri`` with a single-config dataset,
            set this to an empty string.
        phase_02_train_opt_n_train: Maximum rows ART draws from the dataset
            per run. Independent of ``phase_01_dataset_opt_subset``.
            Example: ``200`` for quick tests, ``2000`` for a fuller run.
        phase_02_train_opt_learning_rate: AdamW learning rate for LoRA adapter
            updates. ``1e-5`` is a safe default; increase to ``5e-5`` if reward
            improvement is too slow.
        phase_02_train_opt_gpu_memory_utilization: Fraction of GPU VRAM given
            to the vLLM rollout engine (remainder goes to the policy model and
            optimizer). Default 0.45 is tuned for Qwen3-4B on a 80GB A100.
            Reduce to 0.35 if you see OOM during rollout generation.
        phase_02_train_opt_enforce_eager: Disables torch.compile and CUDA graph
            capture in vLLM. Must be True for Qwen3 models (known CUDAGraph
            incompatibility). Set False for other models for faster rollouts.
        phase_02_train_opt_env_vars: Comma-separated ``KEY=VALUE`` environment
            variables injected into the training pod. Leave empty if not needed.
            Example: ``NCCL_DEBUG=INFO,TORCH_DISTRIBUTED_DEBUG=DETAIL``
        phase_02_train_opt_cpu: CPU cores requested for the TrainJob worker pod.
            Example: ``4``
        phase_02_train_opt_gpu: GPUs per TrainJob worker. ART requires exactly
            1 GPU per worker - do not change this.
        phase_02_train_opt_memory: RAM requested for the TrainJob worker pod.
            For Qwen3-4B on an 80GB GPU, 64Gi is sufficient.
            Example: ``64Gi``
        phase_02_train_opt_labels: Comma-separated ``key=value`` labels applied
            to the TrainJob worker pod for cost tracking or scheduling
            constraints. Leave empty if not needed.
            Example: ``team=ml,project=grpo``
        phase_02_train_opt_annotations: Comma-separated ``key=value``
            annotations applied to the TrainJob worker pod. Leave empty if
            not needed.
        phase_02_train_opt_runtime: Name of the ClusterTrainingRuntime CR that
            defines the base container image. Check available runtimes with:
            ``oc get clustertrainingruntimes``. Default: ``training-hub``
        phase_04_registry_opt_author: Author name attached to the registry
            entry. Example: ``your-team-name``
        phase_04_registry_opt_description: Free-text description stored with
            the registered model. Example: ``Qwen3-4B fine-tuned with GRPO
            for tool-calling tasks``
        phase_04_registry_opt_format_name: Model serialization format in the
            registry. Use ``pytorch`` for vLLM-served models.
            Example: ``pytorch``
        phase_04_registry_opt_format_version: Version of the model format.
            Use ``1.0`` for PyTorch. Example: ``1.0``
        phase_04_registry_opt_port: HTTP port of the Model Registry service.
            Default 8080 matches the standard RHOAI deployment.
        phase_05_deploy_opt_gpu_count: GPUs allocated to the KServe vLLM
            predictor pod. 1 GPU is sufficient for Qwen3-4B. Example: ``1``
        phase_05_deploy_opt_max_model_len: Maximum total token length
            (prompt + completion) vLLM will handle. Longer contexts use more
            VRAM. Default 4096 is adequate for tool-call traces. Increase to
            8192 for longer prompts.
        phase_05_deploy_opt_min_replicas: Minimum live KServe predictor
            replicas. Set to 1 to keep the endpoint always warm.
            Example: ``1``
        phase_05_deploy_opt_max_replicas: Maximum predictor replicas for
            autoscaling. Set equal to ``min_replicas`` to disable scaling.
            Example: ``1``
        phase_05_deploy_opt_cpu_requests: CPU cores requested and limited for
            the KServe predictor pod. Example: ``2``
        phase_05_deploy_opt_memory_requests: RAM requested and limited for the
            KServe predictor pod. Example: ``8Gi``
    """
    # =========================================================================
    # Stage 1: Dataset Download
    # =========================================================================
    dataset_download_task = dataset_download(
        dataset_uri=phase_01_dataset_man_data_uri,
        pvc_mount_path=PVC_MOUNT_PATH,
        train_split_ratio=phase_01_dataset_man_data_split,
        subset_count=phase_01_dataset_opt_subset,
        dataset_format="tool_call",
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
    # Stage 2: LoRA GRPO Training
    # =========================================================================
    training_task = train_model(
        pvc_path=PVC_MOUNT_PATH,
        dataset=dataset_download_task.outputs["train_dataset"],
        training_base_model=phase_02_train_man_model,
        training_data_path="",
        training_num_iterations=phase_02_train_man_num_iterations,
        training_group_size=phase_02_train_man_group_size,
        training_prompt_batch_size=phase_02_train_man_prompt_batch_size,
        training_n_train=phase_02_train_opt_n_train,
        training_learning_rate=phase_02_train_opt_learning_rate,
        training_gpu_memory_utilization=phase_02_train_opt_gpu_memory_utilization,
        training_enforce_eager=phase_02_train_opt_enforce_eager,
        training_data_config=phase_02_train_opt_data_config,
        training_lora_r=phase_02_train_man_lora_r,
        training_lora_alpha=phase_02_train_man_lora_alpha,
        training_envs=phase_02_train_opt_env_vars,
        training_resource_cpu_per_worker=phase_02_train_opt_cpu,
        training_resource_gpu_per_worker=phase_02_train_opt_gpu,
        training_resource_memory_per_worker=phase_02_train_opt_memory,
        training_metadata_labels=phase_02_train_opt_labels,
        training_metadata_annotations=phase_02_train_opt_annotations,
        training_runtime=phase_02_train_opt_runtime,
        training_pvc_name=phase_00_infra_man_pvc_name,
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
    # Stage 3: GRPO Evaluation
    # =========================================================================
    grpo_eval_task = grpo_eval(
        training_results_path=f"{PVC_MOUNT_PATH}/checkpoints/training_results.json",
    )
    grpo_eval_task.after(training_task)
    grpo_eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(grpo_eval_task, "IfNotPresent")

    # Attach HF token to tasks that may access gated resources
    for _task in [dataset_download_task, training_task]:
        kfp.kubernetes.use_secret_as_env(
            task=_task,
            secret_name="hf-token",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
            optional=True,
        )

    # Mount user PVC on all tasks that need shared file access
    for _task in [dataset_download_task, training_task, grpo_eval_task]:
        kfp.kubernetes.mount_pvc(
            task=_task,
            pvc_name=phase_00_infra_man_pvc_name,
            mount_path=PVC_MOUNT_PATH,
        )

    # =========================================================================
    # Stages 4 + 5: Gated by promotion check
    # Only run model registration and deployment if training improved
    # (final_reward > initial_reward across iterations).
    # =========================================================================
    with dsl.If(
        grpo_eval_task.outputs["promotion_passed"] == True,  # noqa: E712 - KFP dsl.If requires explicit == comparison
        name="promotion-gate",
    ):
        # Stage 4: Model Registry
        registry_task = kubeflow_model_registry(
            pvc_mount_path=PVC_MOUNT_PATH,
            input_model=training_task.outputs["output_model"],
            input_metrics=training_task.outputs["output_metrics"],
            eval_metrics=grpo_eval_task.outputs["output_metrics"],
            eval_results=grpo_eval_task.outputs["output_reward_chart"],
            registry_address=phase_04_registry_man_address,
            registry_port=phase_04_registry_opt_port,
            model_name=phase_04_registry_man_name,
            model_version=phase_04_registry_man_version,
            model_format_name=phase_04_registry_opt_format_name,
            model_format_version=phase_04_registry_opt_format_version,
            model_description=phase_04_registry_opt_description,
            author=phase_04_registry_opt_author,
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
            pvc_name=phase_00_infra_man_pvc_name,
            mount_path=PVC_MOUNT_PATH,
        )

        # Stage 5: Model Deployment
        deploy_task = model_deployment(
            model_name=phase_02_train_man_model,
            namespace=phase_05_deploy_man_namespace,
            model_dir="final_model",
            model_cache_pvc=phase_00_infra_man_pvc_name,
            gpu_count=phase_05_deploy_opt_gpu_count,
            max_model_len=phase_05_deploy_opt_max_model_len,
            min_replicas=phase_05_deploy_opt_min_replicas,
            max_replicas=phase_05_deploy_opt_max_replicas,
            cpu_requests=phase_05_deploy_opt_cpu_requests,
            memory_requests=phase_05_deploy_opt_memory_requests,
            cpu_limits=phase_05_deploy_opt_cpu_requests,
            memory_limits=phase_05_deploy_opt_memory_requests,
        )
        deploy_task.after(registry_task)
        deploy_task.set_caching_options(False)
        kfp.kubernetes.set_image_pull_policy(deploy_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=lora_grpo_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
