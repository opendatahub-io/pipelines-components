"""Speculator hidden-state extraction pipeline with managed or remote vLLM."""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.download_model import download_model as download_model_component
from components.training.finetuning.speculator import extract_hidden_states

PVC_SIZE = "200Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "speculator-hidden-state-extraction-pipeline"


def _attach_k8s(task) -> None:
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
    description="Extract verifier hidden states for speculative decoding training using managed or remote vLLM.",
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
def speculator_hidden_state_extraction_pipeline(
    p01_persistent_pvc: str,
    p02_verifier_model: str = "",
    p03_verifier_model_pvc: str = "",
    p04_download_model: bool = False,
    p05_dataset_name: str = "ultrachat",
    p06_output_dir: str = "speculator-output",
    p07_hidden_states_path: str = "speculator-output/hidden_states",
    p08_max_samples: int = 16,
    p09_regenerate_responses: bool = False,
    p10_vllm_mode: str = "managed",
    p11_vllm_endpoint: str = "",
    p12_vllm_resource_gpu: int = 1,
    p13_vllm_resource_memory: str = "96Gi",
    p14_vllm_gpu_memory_utilization: float = 0.9,
    p15_vllm_readiness_timeout_minutes: int = 60,
    p16_total_seq_len: int = 2048,
    p17_speculator_type: str = "eagle3",
    p18_target_layer_ids: str = "",
    p19_data_extraction_concurrency: int = 1,
    p20_packages_to_install: str = "",
    p21_pip_index_urls: str = "",
    p22_job_envs: str = "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1",
    p23_job_runtime: str = "vllm-extract-cuda",
):
    """Extract verifier hidden states for speculative decoding training using managed or remote vLLM.

    Managed mode runs vLLM in the Trainer job. When ``p04_download_model`` is true,
    ``p02_verifier_model`` is downloaded to ``p01_persistent_pvc``; otherwise,
    managed mode reuses ``p03_verifier_model_pvc``. Remote mode uses
    ``p11_vllm_endpoint`` and requires the verifier and hidden-state PVC paths to be
    accessible to the external service. In remote mode, configure
    ``p07_hidden_states_path`` to match the hidden-state location used by the remote
    vLLM service on the shared PVC. Managed mode writes hidden states under
    ``p06_output_dir/hidden_states``.

    Relative storage paths are resolved against ``p01_persistent_pvc``. For example,
    with ``p01_persistent_pvc="shared"``, ``p06_output_dir="speculator-output"``
    resolves to ``pvc://shared/speculator-output``. Numeric prefixes preserve a
    logical parameter order in the KFP UI, which sorts inputs alphabetically.

    Args:
        p01_persistent_pvc: Name of the existing PVC in the pipeline namespace,
            mounted at ``/mnt/persistent``, that stores models, datasets, and
            extraction outputs. It is required and has no default.
        p02_verifier_model: Hugging Face Hub model ID. Set this when
            ``p04_download_model`` is true to download it to the PVC. In managed mode,
            when ``p04_download_model`` is false and ``p03_verifier_model_pvc`` is
            empty, this ID is passed directly to vLLM.
        p03_verifier_model_pvc: Optional verifier directory on ``p01_persistent_pvc``.
            In managed mode with ``p04_download_model`` false, set this to reuse a
            cached model; if empty, ``p02_verifier_model`` is used directly. Remote
            mode always requires this PVC-backed path. It is ignored in managed mode
            when ``p04_download_model`` is true. Use a relative path or a ``pvc://`` URI.
        p04_download_model: Download ``p02_verifier_model`` to the PVC. When false,
            managed mode uses ``p03_verifier_model_pvc`` if set, or passes
            ``p02_verifier_model`` directly otherwise. Remote mode uses
            ``p03_verifier_model_pvc`` for the verifier visible to the external endpoint.
        p05_dataset_name: Use ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered),
            ``ultrachat`` (HuggingFaceH4/ultrachat_200k; default), or ``gsm8k``
            (openai/gsm8k). For custom data, use an existing JSON/JSONL file on the
            PVC (relative path or ``pvc://<claim>/<path>``); arbitrary Hugging Face
            dataset IDs are not supported.
        p06_output_dir: Directory on ``p01_persistent_pvc`` where extraction outputs
            are written. The default is ``speculator-output``; use another relative
            path or URI to choose a different directory.
        p07_hidden_states_path: Remote-mode output location for extracted hidden
            states. It must match the location on the shared PVC used by the remote
            vLLM service. Managed mode ignores this input and writes to
            ``<p06_output_dir>/hidden_states`` automatically.
        p08_max_samples: Maximum samples to process; ``0`` processes the full dataset.
        p09_regenerate_responses: Re-generate the assistant responses in the dataset
            with the verifier model instead of using the original responses. Only
            supported for the built-in datasets ``magpie``, ``ultrachat``, and ``gsm8k``.
        p10_vllm_mode: Use ``managed`` for a vLLM sidecar in the job or ``remote`` for
            an external vLLM service.
        p11_vllm_endpoint: URL of the external vLLM service; required for ``remote`` and blank for ``managed``.
        p12_vllm_resource_gpu: GPUs requested by managed vLLM; the SDK currently supports exactly ``1``.
        p13_vllm_resource_memory: Memory requested by the managed vLLM sidecar.
        p14_vllm_gpu_memory_utilization: Fraction of GPU memory available to vLLM, in the range ``(0, 1]``.
        p15_vllm_readiness_timeout_minutes: Maximum wait for the managed or remote vLLM
            endpoint to become ready, in minutes; minimum ``1``.
        p16_total_seq_len: Maximum sequence length for dataset preprocessing and vLLM extraction; must be positive.
        p17_speculator_type: Draft architecture; ``eagle3`` is currently the only supported type.
        p18_target_layer_ids: Exactly four comma-separated verifier layer indices.
            Required when using ``p03_verifier_model_pvc``, when
            ``p04_download_model`` is true, or in remote mode. It may be omitted only
            in managed mode when ``p03_verifier_model_pvc`` is empty,
            ``p02_verifier_model`` is a Hugging Face model ID, and
            ``p04_download_model`` is false; the SDK can then infer indices from the
            model config.
        p19_data_extraction_concurrency: Number of concurrent vLLM requests for
            hidden-state extraction; use ``1`` on shared or NFS storage.
        p20_packages_to_install: Comma-separated Python packages to install in the job.
        p21_pip_index_urls: Comma-separated package-index URLs used for job installs.
        p22_job_envs: Comma-separated ``KEY=VALUE`` environment overrides for the job.
        p23_job_runtime: Name of the runtime used by the extraction job.
    """
    # Keep concise internal names while exposing zero-padded KFP parameter names.
    persistent_pvc = p01_persistent_pvc
    verifier_model = p02_verifier_model
    verifier_model_pvc = p03_verifier_model_pvc
    download_model = p04_download_model
    dataset_name = p05_dataset_name
    output_dir = p06_output_dir
    hidden_states_path = p07_hidden_states_path
    max_samples = p08_max_samples
    regenerate_responses = p09_regenerate_responses
    vllm_mode = p10_vllm_mode
    vllm_endpoint = p11_vllm_endpoint
    vllm_resource_gpu = p12_vllm_resource_gpu
    vllm_resource_memory = p13_vllm_resource_memory
    vllm_gpu_memory_utilization = p14_vllm_gpu_memory_utilization
    vllm_readiness_timeout_minutes = p15_vllm_readiness_timeout_minutes
    total_seq_len = p16_total_seq_len
    speculator_type = p17_speculator_type
    target_layer_ids = p18_target_layer_ids
    data_extraction_concurrency = p19_data_extraction_concurrency
    packages_to_install = p20_packages_to_install
    pip_index_urls = p21_pip_index_urls
    job_envs = p22_job_envs
    job_runtime = p23_job_runtime

    common_kwargs = dict(
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset_name=dataset_name,
        hidden_states_path=hidden_states_path,
        total_seq_len=total_seq_len,
        speculator_type=speculator_type,
        target_layer_ids=target_layer_ids,
        data_extraction_max_samples=max_samples,
        data_extraction_regenerate_responses=regenerate_responses,
        data_extraction_concurrency=data_extraction_concurrency,
        vllm_resource_gpu=vllm_resource_gpu,
        vllm_resource_memory=vllm_resource_memory,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_readiness_timeout_minutes=vllm_readiness_timeout_minutes,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
        training_envs=job_envs,
        training_runtime=job_runtime,
        persistent_pvc=persistent_pvc,
        persistent_mount_path="/mnt/persistent",
        download_model=download_model,
        verifier_model_pvc=verifier_model_pvc,
    )

    download_task = download_model_component(
        model_name=verifier_model,
        model_cache_pvc=persistent_pvc,
        model_cache_mount="/mnt/persistent/models",
        download_enabled=download_model,
    )
    download_task.set_caching_options(False)
    kfp.kubernetes.use_secret_as_env(
        download_task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )
    kfp.kubernetes.mount_pvc(
        download_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )

    with dsl.If(vllm_mode == "managed", name="managed-vllm"):
        managed_task = extract_hidden_states(
            verifier_model=verifier_model,
            vllm_source="managed",
            **common_kwargs,
        )
        managed_task.after(download_task)
        kfp.kubernetes.mount_pvc(
            managed_task,
            pvc_name=persistent_pvc,
            mount_path="/mnt/persistent",
        )
        _attach_k8s(managed_task)

    with dsl.Else(name="remote-vllm"):
        remote_task = extract_hidden_states(
            verifier_model=verifier_model_pvc,
            vllm_source=vllm_mode,
            vllm_endpoint=vllm_endpoint,
            **common_kwargs,
        )
        remote_task.after(download_task)
        kfp.kubernetes.mount_pvc(
            remote_task,
            pvc_name=persistent_pvc,
            mount_path="/mnt/persistent",
        )
        _attach_k8s(remote_task)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=speculator_hidden_state_extraction_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
