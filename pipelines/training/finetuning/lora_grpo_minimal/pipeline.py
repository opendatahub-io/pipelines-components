"""Quick-start LoRA GRPO pipeline with the full training-to-serving DAG.

The minimal variant keeps the standard LoRA GRPO stages and promotion gate,
while exposing a focused set of model, data, training, timeout, and serving
parameters. Low-level runtime and metadata settings use tested defaults.
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

PVC_MOUNT_PATH = "/mnt/pipeline"
PIPELINE_NAME = "lora-grpo-minimal-pipeline"

# Keep the GRPO-specific registry SDK pin aligned with the standard pipeline.
GRPO_MODEL_REGISTRY = dsl.component(
    base_image="quay.io/opendatahub/odh-th06-cpu-torch291-py312:odh-3.4",
    packages_to_install=["model-registry==0.3.9"],
)(shared_model_registry.python_func)


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "LoRA GRPO minimal pipeline: quick-start reinforcement learning from verifiable rewards "
        "with promotion-gated registry and deployment"
    ),
)
def lora_grpo_minimal_pipeline(
    # Required infrastructure and model inputs.
    phase_00_infra_man_pvc_name: str,
    phase_04_registry_man_address: str,
    phase_05_deploy_man_namespace: str,
    phase_01_dataset_man_data_uri: str,
    phase_02_train_man_model: str,
    phase_05_deploy_opt_tokenizer_mode: str,
    # Dataset inputs.
    phase_01_dataset_opt_max_invalid_row_ratio: float = 0.10,
    # Core GRPO and LoRA inputs.
    phase_02_train_man_num_iterations: int = 5,
    phase_02_train_man_group_size: int = 4,
    phase_02_train_man_prompt_batch_size: int = 50,
    phase_02_train_man_lora_r: int = 16,
    phase_02_train_man_lora_alpha: int = 16,
    phase_04_registry_man_name: str = "grpo-model",
    phase_04_registry_man_version: str = "1.0.0",
    # Useful training and troubleshooting inputs.
    phase_02_train_opt_n_train: int = 200,
    phase_02_train_opt_completion_timeout_seconds: int = 7200,
    phase_02_train_opt_learning_rate: float = 1e-5,
    phase_02_train_opt_gpu_memory_utilization: float = 0.45,
    phase_02_train_opt_enforce_eager: bool = True,
    phase_02_train_opt_env_vars: str = "",
    # Registry and serving inputs that commonly vary by installation.
    phase_04_registry_opt_port: int = 8080,
    phase_05_deploy_opt_gpu_count: int = 1,
    phase_05_deploy_opt_max_model_len: int = 4096,
    phase_05_deploy_opt_max_replicas: int = 1,
    phase_05_deploy_opt_cpu_requests: str = "2",
    phase_05_deploy_opt_memory_requests: str = "8Gi",
):
    """Run the complete quick-start LoRA GRPO workflow.

    The pipeline prepares the dataset, submits a single-GPU ART TrainJob,
    evaluates reward improvement, and registers and deploys the merged model
    only when the promotion gate passes.

    Private S3 dataset inputs use the optional RHOAI data connection Secret
    named ``s3-secret``. This is the default Secret name; do not change it.
    The Secret should provide ``AWS_ACCESS_KEY_ID`` and
    ``AWS_SECRET_ACCESS_KEY``, and optionally ``AWS_S3_ENDPOINT`` and
    ``AWS_DEFAULT_REGION``. Hugging Face, HTTP(S), and local/PVC inputs do
    not require this Secret.

    The following implementation details are intentionally fixed for the
    minimal variant: the Training Hub runtime, one training GPU, 4 CPU cores,
    64Gi worker memory, empty TrainJob labels/annotations, the PyTorch model
    format, and the standard registry/deployment metadata defaults.

    Args:
        phase_00_infra_man_pvc_name: Name of the ReadWriteMany PVC used for
            training checkpoints and model serving. To create one: in RHOAI open
            your Data Science Project -> Cluster Storage -> Create storage, select
            ReadWriteMany access mode, minimum 50Gi. Enter only the PVC name
            (not a path). Example: ``grpo-pipeline-pvc``
        phase_01_dataset_man_data_uri: Dataset reference. Hugging Face
            references are loaded directly by Training Hub; local JSON/JSONL,
            PVC, S3, and HTTP(S) sources are normalized by the preparation
            component onto the shared PVC. For Hugging Face datasets, use
            ``hf://org/dataset:config``; the ``hf://`` prefix is optional for
            repository IDs. Non-Hugging Face rows are
            converted to Training Hub's canonical ``messages`` and ``tools``
            format. See the [Training Hub loader format](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/lora_grpo.py#L381-L387).
        phase_01_dataset_opt_max_invalid_row_ratio: Maximum fraction of invalid
            rows that the preparation component may skip for local, PVC, S3,
            and HTTP(S) sources before failing. Defaults to 10%; use ``0.0``
            for strict validation. It is not applied to direct Hugging Face
            loading.
        phase_05_deploy_man_namespace: OpenShift project where the KServe
            InferenceService will be created. Must be the same namespace where
            your PVC is provisioned. Should have
            KServe enabled - typically your Data Science Project.
            Example: ``my-ds-project``
        phase_02_train_man_model: HuggingFace model ID of the base model to
            fine-tune. If the model is gated, configure the ``hf-token`` secret.
            Examples: ``Qwen/Qwen3-4B`` or
            ``ibm-granite/granite-3.3-8b-instruct``
        phase_02_train_man_num_iterations: Number of GRPO policy-gradient update
            steps. Each iteration generates ``group_size`` rollouts per prompt
            and updates the LoRA adapter. The promotion gate (stage 3) requires
            ``final_reward > initial_reward``. Increase the completion timeout
            when using higher iteration counts, larger models, or slower
            hardware.
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
        phase_04_registry_man_address: Hostname or IP address of the Model
            Registry service (required). Pass only the host or IP, without
            ``http://``, ``https://``, or the ``/api/model_registry/v1alpha3``
            path.
        phase_04_registry_man_name: Name to register the model under. Must be
            unique within your Model Registry. Defaults to ``grpo-model``;
            override it with a unique name such as ``qwen3-grpo-tool-call``
            for repeated or production runs.
        phase_04_registry_man_version: Version tag for this model revision.
            Use semantic versioning. Example: ``1.0.0``
        phase_02_train_opt_n_train: Maximum rows ART draws from the dataset
            per run. This is independent of the total dataset size.
            Example: ``200`` for quick tests, ``2000`` for a fuller run.
        phase_02_train_opt_completion_timeout_seconds: Maximum number of
            seconds to wait for the TrainJob to reach Complete or Failed after
            it starts. The wait returns immediately when the TrainJob completes.
            Set this to a larger value for longer runs or slower hardware, or
            a smaller value for smoke tests. The TrainJob startup wait remains
            unchanged.
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
        phase_04_registry_opt_port: Port of the Model Registry service.
            Default 8080 matches a typical internal HTTP registry.
        phase_05_deploy_opt_gpu_count: GPUs allocated to the KServe vLLM
            predictor pod. 1 GPU is sufficient for Qwen3-4B. Example: ``1``
        phase_05_deploy_opt_max_model_len: Maximum total token length
            (prompt + completion) vLLM will handle. Longer contexts use more
            VRAM. Default 4096 is adequate for tool-call traces. Increase to
            8192 for longer prompts.
        phase_05_deploy_opt_max_replicas: Maximum predictor replicas for
            autoscaling. Set equal to ``min_replicas`` to disable scaling.
            Example: ``1``
        phase_05_deploy_opt_cpu_requests: CPU cores requested and limited for
            the KServe predictor pod. Example: ``2``
        phase_05_deploy_opt_memory_requests: RAM requested and limited for the
            KServe predictor pod. Example: ``8Gi``
        phase_05_deploy_opt_tokenizer_mode: Required vLLM tokenizer mode.
            Use ``auto`` when the fast tokenizer is compatible, ``slow`` when
            the model requires its HuggingFace slow tokenizer (for example,
            Qwen3 in the tested environment), or ``mistral`` for models that
            require the ``mistral_common`` tokenizer. Do not leave it empty.
    """

    def configure_training_task(training_task):
        """Apply the standard Kubernetes configuration to the TrainJob task."""
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
            pvc_name=phase_00_infra_man_pvc_name,
            mount_path=PVC_MOUNT_PATH,
        )
        return training_task

    # Stage 1: Dataset preparation.
    prepare_dataset_task = prepare_grpo_dataset(
        dataset_uri=phase_01_dataset_man_data_uri,
        pvc_mount_path=PVC_MOUNT_PATH,
        max_invalid_row_ratio=phase_01_dataset_opt_max_invalid_row_ratio,
    )
    prepare_dataset_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(prepare_dataset_task, "IfNotPresent")
    kfp.kubernetes.mount_pvc(
        task=prepare_dataset_task,
        pvc_name=phase_00_infra_man_pvc_name,
        mount_path=PVC_MOUNT_PATH,
    )
    kfp.kubernetes.use_secret_as_env(
        task=prepare_dataset_task,
        secret_name="s3-secret",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
        optional=True,
    )

    # Stage 2: LoRA GRPO training. The fixed one-GPU settings match ART's
    # single-GPU contract while timeout and rollout controls remain visible.
    training_task = configure_training_task(
        train_model(
            pvc_path=PVC_MOUNT_PATH,
            training_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            training_base_model=phase_02_train_man_model,
            training_data_path=prepare_dataset_task.outputs["resolved_data_path"],
            training_num_iterations=phase_02_train_man_num_iterations,
            training_group_size=phase_02_train_man_group_size,
            training_prompt_batch_size=phase_02_train_man_prompt_batch_size,
            training_n_train=phase_02_train_opt_n_train,
            training_completion_timeout_seconds=phase_02_train_opt_completion_timeout_seconds,
            training_learning_rate=phase_02_train_opt_learning_rate,
            training_gpu_memory_utilization=phase_02_train_opt_gpu_memory_utilization,
            training_enforce_eager=phase_02_train_opt_enforce_eager,
            training_data_config=prepare_dataset_task.outputs["resolved_data_config"],
            training_lora_r=phase_02_train_man_lora_r,
            training_lora_alpha=phase_02_train_man_lora_alpha,
            training_envs=phase_02_train_opt_env_vars,
            training_resource_cpu_per_worker="4",
            training_resource_gpu_per_worker=1,
            training_resource_memory_per_worker="64Gi",
            training_metadata_labels="",
            training_metadata_annotations="",
            training_runtime="training-hub",
            training_pvc_name=phase_00_infra_man_pvc_name,
            training_use_hf_direct=prepare_dataset_task.outputs["use_hf_direct"],
        )
    )
    training_task.after(prepare_dataset_task)

    training_results_path = training_task.outputs["Output"]
    training_model = training_task.outputs["output_model"]
    training_metrics = training_task.outputs["output_metrics"]

    # Stage 3: GRPO evaluation and promotion decision.
    grpo_eval_task = grpo_eval(training_results_path=training_results_path)
    grpo_eval_task.after(training_task)
    grpo_eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(grpo_eval_task, "IfNotPresent")
    kfp.kubernetes.mount_pvc(
        task=grpo_eval_task,
        pvc_name=phase_00_infra_man_pvc_name,
        mount_path=PVC_MOUNT_PATH,
    )

    # Stages 4 and 5: register and deploy only after reward improvement.
    with dsl.If(
        grpo_eval_task.outputs["promotion_passed"] == True,  # noqa: E712 - KFP dsl.If requires explicit == comparison
        name="promotion-gate",
    ):
        registry_task = GRPO_MODEL_REGISTRY(
            pvc_mount_path=PVC_MOUNT_PATH,
            model_pvc_name=phase_00_infra_man_pvc_name,
            input_model=training_model,
            input_metrics=training_metrics,
            eval_metrics=grpo_eval_task.outputs["output_metrics"],
            eval_results=grpo_eval_task.outputs["output_reward_chart"],
            registry_address=phase_04_registry_man_address,
            registry_port=phase_04_registry_opt_port,
            model_name=phase_04_registry_man_name,
            model_version=phase_04_registry_man_version,
            model_format_name="pytorch",
            model_format_version="1.0",
            model_description="",
            author="pipeline",
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

        deploy_task = model_deployment(
            model_name=phase_02_train_man_model,
            namespace=phase_05_deploy_man_namespace,
            model_dir="final_model",
            model_cache_pvc=phase_00_infra_man_pvc_name,
            model_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            gpu_count=phase_05_deploy_opt_gpu_count,
            max_model_len=phase_05_deploy_opt_max_model_len,
            min_replicas=1,
            max_replicas=phase_05_deploy_opt_max_replicas,
            cpu_requests=phase_05_deploy_opt_cpu_requests,
            memory_requests=phase_05_deploy_opt_memory_requests,
            cpu_limits=phase_05_deploy_opt_cpu_requests,
            memory_limits=phase_05_deploy_opt_memory_requests,
            tokenizer_mode=phase_05_deploy_opt_tokenizer_mode,
        )
        deploy_task.after(registry_task)
        deploy_task.set_caching_options(False)
        kfp.kubernetes.set_image_pull_policy(deploy_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=lora_grpo_minimal_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
