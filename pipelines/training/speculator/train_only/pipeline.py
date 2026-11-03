"""Train, evaluate, register, and serve a Speculator draft model."""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.deployment.speculative_decoding_serving import speculative_decoding_serving
from components.evaluation.speculator import evaluate_draft_model
from components.training.finetuning.speculator import train_draft_model

PVC_SIZE = "200Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "speculator-training-pipeline"


def _attach_k8s(task) -> None:
    """Attach the Kubernetes credentials and Hugging Face token to a task."""
    task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(task, "IfNotPresent")
    kfp.kubernetes.use_secret_as_env(
        task,
        secret_name="kubernetes-credentials",
        secret_key_to_env={
            "KUBERNETES_SERVER_URL": "KUBERNETES_SERVER_URL",
            "KUBERNETES_AUTH_TOKEN": "KUBERNETES_AUTH_TOKEN",
        },
        optional=False,
    )
    kfp.kubernetes.use_secret_as_env(
        task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Train and evaluate an Eagle3 draft from PVC-stored artifacts, register it, and serve it.",
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
def speculator_training_pipeline(
    p01_persistent_pvc: str,
    p02_model_registry_address: str,
    p03_model_registry_name: str,
    p04_serving_namespace: str,
    p05_serving_endpoint_name: str,
    p06_verifier_model_pvc: str,
    p07_evaluation_dataset_uri: str,
    p08_training_data_path: str = "speculator-output",
    p09_hidden_states_path: str = "speculator-output/hidden_states",
    p10_output_dir: str = "speculator-output",
    p11_total_seq_len: int = 2048,
    p12_speculator_type: str = "eagle3",
    p13_target_layer_ids: str = "2,14,25,28",
    p14_training_epochs: int = 3,
    p15_training_lr: float = 1e-4,
    p16_training_draft_vocab_size: int = 0,
    p17_training_num_layers: int = 1,
    p18_training_test_time_steps: int = 1,
    p19_training_norm_before_residual: bool = True,
    p20_training_norm_before_fc: bool = False,
    p21_training_embed_requires_grad: bool = False,
    p22_training_hidden_states_dtype: str = "bfloat16",
    p23_training_scheduler_type: str = "linear",
    p24_training_scheduler_warmup_steps: int = 0,
    p25_training_scheduler_total_steps: int = 0,
    p26_training_scheduler_num_cosine_cycles: float = 0.5,
    p27_training_checkpoint_freq: float = 1.0,
    p28_training_save_best: bool = False,
    p29_training_log_freq: int = 1,
    p30_training_resume_from_checkpoint: bool = False,
    p31_training_resource_cpu: str = "4",
    p32_training_resource_gpu: int = 1,
    p33_training_resource_memory: str = "64Gi",
    p34_job_envs: str = "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1",
    p35_job_runtime: str = "speculator-model-opt-cuda",
    p36_packages_to_install: str = "",
    p37_pip_index_urls: str = "",
    p38_evaluation_max_samples: int = 80,
    p39_evaluation_max_tokens: int = 256,
    p40_evaluation_temperature: float = 0.0,
    p41_num_speculative_tokens: int = 5,
    p42_gpu_memory_utilization: float = 0.9,
    p43_evaluation_startup_timeout_seconds: int = 900,
    p44_evaluation_sample_seed: int = 42,
    p45_model_registry_port: int = 8080,
    p46_model_registry_version: str = "1.0.0",
    p47_serving_runtime_image: str = "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a",  # noqa: E501
):
    """Train, evaluate, register, and serve a draft from existing artifacts.

    This pipeline trains a draft from existing artifacts and expects the verifier model,
    hidden states, and the dataset from the data extraction pipeline to already exist
    on the shared PVC.
    It does not download the verifier. Relative storage paths are resolved against
    ``p01_persistent_pvc``. Explicit ``pvc://<claim>/<path>`` paths must reference that
    same PVC.

    For example, with ``p01_persistent_pvc="shared"``,
    ``p06_verifier_model_pvc="models/Qwen--Qwen3-0.6B"`` is equivalent to
    ``pvc://shared/models/Qwen--Qwen3-0.6B``. Likewise,
    ``p10_output_dir="speculator-output"`` and
    ``p09_hidden_states_path="speculator-output/hidden_states"`` can be written as
    ``pvc://shared/speculator-output`` and
    ``pvc://shared/speculator-output/hidden_states``. The default
    ``p08_training_data_path`` and ``p10_output_dir`` are both ``speculator-output``,
    matching the data-only pipeline defaults. If you customize the data-only
    output directory, update ``p08_training_data_path`` and ``p09_hidden_states_path`` too.
    The hidden-state default matches the data-only pipeline's default output
    directory; it is not discovered automatically from a previous pipeline run.

    The trained draft is copied to ``pvc://<p01_persistent_pvc>/final_model`` for
    evaluation, Model Registry, and serving. The resulting KServe endpoint uses
    cluster-local visibility.

    Model Registry setup:
        In the RHOAI dashboard, open **AI Hub > Models > Model registries** and
        create or select a registry. Copy its HTTP REST service URL. This pipeline
        takes the host and port separately: put only the hostname in
        ``p02_model_registry_address`` and the service port in ``p45_model_registry_port``.
        For a URL such as
        ``http://registry-service.registry-namespace.svc:8080/api/model_registry/v1alpha3``,
        use ``registry-service.registry-namespace.svc`` and ``8080``; omit the
        scheme and API path. The pipeline ServiceAccount must have a RoleBinding
        to ``registry-user-<registry-name>`` in the registry namespace.
        ``p02_model_registry_address`` is required because this pipeline registers the
        evaluated model before serving it. Use a new ``p46_model_registry_version`` for
        each trained revision.

    Args:
        p01_persistent_pvc: Name of an existing ReadWriteMany PVC mounted at
            ``/mnt/persistent``; it stores the verifier, hidden states, training data,
            and model output.
        p02_model_registry_address: Required Model Registry REST host; enter only the hostname.
        p03_model_registry_name: Required model name, unique in the selected registry.
        p04_serving_namespace: Namespace where KServe creates the endpoint; it must match
            the PVC namespace and have required permissions.
        p05_serving_endpoint_name: Required DNS-compatible KServe InferenceService name.
        p06_verifier_model_pvc: Required existing verifier directory on ``p01_persistent_pvc``.
            Use a relative path or matching ``pvc://`` URI.
        p07_evaluation_dataset_uri: Required Hugging Face dataset ID or local JSONL file
            on ``p01_persistent_pvc``. Use a relative path or matching ``pvc://`` URI.
        p08_training_data_path: Directory containing the preprocessed dataset from the
            data extraction pipeline. Default: ``speculator-output``. Use a relative path
            or matching URI.
        p09_hidden_states_path: Existing hidden-state directory from data-only extraction.
            Default: ``speculator-output/hidden_states``. Use a relative path or matching URI.
        p10_output_dir: Checkpoint and output directory on ``p01_persistent_pvc``.
            Default: ``speculator-output``. Use a relative path or matching URI.
        p11_total_seq_len: Positive sequence length for preprocessing and training; keep
            consistent with the data-only extraction run.
        p12_speculator_type: Draft architecture. ``eagle3`` is currently supported.
        p13_target_layer_ids: Four comma-separated verifier layer IDs. Use the same IDs as
            data-only extraction; set matching IDs for another verifier.
        p14_training_epochs: Positive number of training epochs.
        p15_training_lr: Positive learning rate for draft-model training.
        p16_training_draft_vocab_size: Draft vocabulary size. ``0`` lets the SDK derive it.
        p17_training_num_layers: Number of layers in the draft model.
        p18_training_test_time_steps: Number of test-time training steps configured for Speculator.
        p19_training_norm_before_residual: Apply normalization before the residual connection.
        p20_training_norm_before_fc: Apply normalization before the fully connected layer.
        p21_training_embed_requires_grad: Whether draft embedding weights are trainable.
        p22_training_hidden_states_dtype: Hidden-state dtype; it must match the verifier.
        p23_training_scheduler_type: Learning-rate scheduler: ``linear``, ``cosine``, or ``none``.
        p24_training_scheduler_warmup_steps: Warmup steps; ``0`` lets the SDK calculate 1% of total steps.
        p25_training_scheduler_total_steps: Total scheduler steps; ``0`` lets the SDK calculate them.
        p26_training_scheduler_num_cosine_cycles: Cosine cycles, used only by the cosine scheduler.
        p27_training_checkpoint_freq: Checkpoint interval in epochs.
        p28_training_save_best: Save only the best checkpoint by validation loss.
        p29_training_log_freq: Training steps between log updates.
        p30_training_resume_from_checkpoint: Resume when the runtime supports it.
        p31_training_resource_cpu: CPU request for the training worker.
        p32_training_resource_gpu: Number of GPUs requested for the training worker.
        p33_training_resource_memory: Memory request for the training worker.
        p34_job_envs: Comma-separated ``KEY=VALUE`` environment overrides for the job.
        p35_job_runtime: Name of the runtime used by the job.
        p36_packages_to_install: Comma-separated Python packages to install for the training job.
        p37_pip_index_urls: Comma-separated package index URLs to use for installs in the training job.
        p38_evaluation_max_samples: Maximum requests per evaluation sweep point; must be positive.
        p39_evaluation_max_tokens: Maximum generated tokens per evaluation request; must be positive.
        p40_evaluation_temperature: GuideLLM sampling temperature; ``0`` is deterministic.
        p41_num_speculative_tokens: Draft tokens proposed per step for evaluation and serving.
        p42_gpu_memory_utilization: GPU memory fraction for evaluation and serving. It
            must be greater than 0 and at most 1.0 (for example, 0.1, 0.2, 0.3, up to 1.0).
        p43_evaluation_startup_timeout_seconds: Maximum wait for the evaluation vLLM server.
        p44_evaluation_sample_seed: Seed for sampling from a local evaluation dataset.
        p45_model_registry_port: HTTP REST service port copied from the registry endpoint; ``8080`` is the default.
        p46_model_registry_version: Version label for this registration; use a new version per revision.
        p47_serving_runtime_image: Red Hat CUDA runtime image used by KServe for Eagle3
            draft serving. The pinned default contains vLLM 0.28.0+rhaiv.7. When upgrading,
            select a compatible vLLM image and replace the digest.
    """
    # Keep concise internal names while exposing zero-padded KFP parameter names.
    persistent_pvc = p01_persistent_pvc
    model_registry_address = p02_model_registry_address
    model_registry_name = p03_model_registry_name
    serving_namespace = p04_serving_namespace
    serving_endpoint_name = p05_serving_endpoint_name
    verifier_model_pvc = p06_verifier_model_pvc
    evaluation_dataset_uri = p07_evaluation_dataset_uri
    training_data_path = p08_training_data_path
    hidden_states_path = p09_hidden_states_path
    output_dir = p10_output_dir
    total_seq_len = p11_total_seq_len
    speculator_type = p12_speculator_type
    target_layer_ids = p13_target_layer_ids
    training_epochs = p14_training_epochs
    training_lr = p15_training_lr
    training_draft_vocab_size = p16_training_draft_vocab_size
    training_num_layers = p17_training_num_layers
    training_ttt_steps = p18_training_test_time_steps
    training_norm_before_residual = p19_training_norm_before_residual
    training_norm_before_fc = p20_training_norm_before_fc
    training_embed_requires_grad = p21_training_embed_requires_grad
    training_hidden_states_dtype = p22_training_hidden_states_dtype
    training_scheduler_type = p23_training_scheduler_type
    training_scheduler_warmup_steps = p24_training_scheduler_warmup_steps
    training_scheduler_total_steps = p25_training_scheduler_total_steps
    training_scheduler_num_cosine_cycles = p26_training_scheduler_num_cosine_cycles
    training_checkpoint_freq = p27_training_checkpoint_freq
    training_save_best = p28_training_save_best
    training_log_freq = p29_training_log_freq
    training_resume_from_checkpoint = p30_training_resume_from_checkpoint
    training_resource_cpu = p31_training_resource_cpu
    training_resource_gpu = p32_training_resource_gpu
    training_resource_memory = p33_training_resource_memory
    job_envs = p34_job_envs
    job_runtime = p35_job_runtime
    packages_to_install = p36_packages_to_install
    pip_index_urls = p37_pip_index_urls
    evaluation_max_samples = p38_evaluation_max_samples
    evaluation_max_tokens = p39_evaluation_max_tokens
    evaluation_temperature = p40_evaluation_temperature
    num_speculative_tokens = p41_num_speculative_tokens
    gpu_memory_utilization = p42_gpu_memory_utilization
    evaluation_startup_timeout_seconds = p43_evaluation_startup_timeout_seconds
    evaluation_sample_seed = p44_evaluation_sample_seed
    model_registry_port = p45_model_registry_port
    model_registry_version = p46_model_registry_version
    serving_runtime_image = p47_serving_runtime_image
    train_task = train_draft_model(
        verifier_model=verifier_model_pvc,
        verifier_model_pvc=verifier_model_pvc,
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        hidden_states_path=hidden_states_path,
        training_data_path=training_data_path,
        total_seq_len=total_seq_len,
        speculator_type=speculator_type,
        target_layer_ids=target_layer_ids,
        training_epochs=training_epochs,
        training_lr=training_lr,
        training_draft_vocab_size=training_draft_vocab_size,
        training_num_layers=training_num_layers,
        training_ttt_steps=training_ttt_steps,
        training_norm_before_residual=training_norm_before_residual,
        training_norm_before_fc=training_norm_before_fc,
        training_embed_requires_grad=training_embed_requires_grad,
        training_hidden_states_dtype=training_hidden_states_dtype,
        training_scheduler_type=training_scheduler_type,
        training_scheduler_warmup_steps=training_scheduler_warmup_steps,
        training_scheduler_total_steps=training_scheduler_total_steps,
        training_scheduler_num_cosine_cycles=training_scheduler_num_cosine_cycles,
        training_checkpoint_freq=training_checkpoint_freq,
        training_save_best=training_save_best,
        training_log_freq=training_log_freq,
        training_resume_from_checkpoint=training_resume_from_checkpoint,
        training_resource_cpu=training_resource_cpu,
        training_resource_gpu=training_resource_gpu,
        training_resource_memory=training_resource_memory,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
        training_envs=job_envs,
        training_runtime=job_runtime,
        persistent_pvc=persistent_pvc,
        persistent_mount_path="/mnt/persistent",
    )
    kfp.kubernetes.mount_pvc(train_task, pvc_name=persistent_pvc, mount_path="/mnt/persistent")
    _attach_k8s(train_task)

    evaluation_task = evaluate_draft_model(
        verifier_model=verifier_model_pvc,
        draft_model_path="/mnt/persistent/final_model",
        evaluation_dataset_uri=evaluation_dataset_uri,
        evaluation_dataset_pvc=persistent_pvc,
        evaluation_max_samples=evaluation_max_samples,
        evaluation_max_tokens=evaluation_max_tokens,
        evaluation_temperature=evaluation_temperature,
        evaluation_num_speculative_tokens=num_speculative_tokens,
        evaluation_speculator_type=speculator_type,
        evaluation_gpu_memory_utilization=gpu_memory_utilization,
        evaluation_startup_timeout_seconds=evaluation_startup_timeout_seconds,
        evaluation_sample_seed=evaluation_sample_seed,
    )
    evaluation_task.after(train_task)
    kfp.kubernetes.mount_pvc(evaluation_task, pvc_name=persistent_pvc, mount_path="/mnt/persistent")
    evaluation_task.set_accelerator_type("nvidia.com/gpu")
    evaluation_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(evaluation_task, "IfNotPresent")
    kfp.kubernetes.use_secret_as_env(
        evaluation_task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )

    model_registry_task = kubeflow_model_registry(
        pvc_mount_path="/mnt/persistent",
        model_pvc_name=persistent_pvc,
        input_model=train_task.outputs["output_model"],
        input_metrics=train_task.outputs["output_metrics"],
        eval_metrics=evaluation_task.outputs["output_metrics"],
        eval_results=evaluation_task.outputs["output_results"],
        registry_address=model_registry_address,
        registry_port=model_registry_port,
        model_name=model_registry_name,
        model_version=model_registry_version,
        model_description="Speculator draft model trained by the train-only pipeline.",
        shared_log_file="pipeline_log.txt",
        source_pipeline_name=PIPELINE_NAME,
        source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        source_namespace="",
    )
    model_registry_task.after(evaluation_task)
    kfp.kubernetes.mount_pvc(
        model_registry_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    model_registry_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(model_registry_task, "IfNotPresent")

    serving_task = speculative_decoding_serving(
        namespace=serving_namespace,
        endpoint_name=serving_endpoint_name,
        model_cache_pvc=persistent_pvc,
        verifier_model_dir=verifier_model_pvc,
        draft_model_dir="final_model",
        runtime_image=serving_runtime_image,
        max_model_len=total_seq_len,
        num_speculative_tokens=num_speculative_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    serving_task.after(model_registry_task)
    serving_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(serving_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=speculator_training_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
