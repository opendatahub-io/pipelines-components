"""Mode-specific Speculator hidden-state extraction component."""

import os
from typing import Optional

from kfp import dsl

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")


@dsl.component(
    base_image=(
        "quay.io/opendatahub/odh-th-torch-cuda-py312:odh-3.6-ea.1@"
        "sha256:9ad2d72ebe892dffd3554eeb81e2a47c9fcf0d97f89ed261d48f4e8bbc367b4b"
    ),
    packages_to_install=["kfp==2.17.0", "kubernetes", "olot"],
    install_kfp_package=False,
    embedded_artifact_path=_SHARED_DIR,
    task_config_passthroughs=[
        dsl.TaskConfigField.RESOURCES,
        dsl.TaskConfigField.KUBERNETES_TOLERATIONS,
        dsl.TaskConfigField.KUBERNETES_NODE_SELECTOR,
        dsl.TaskConfigField.KUBERNETES_AFFINITY,
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.ENV, apply_to_task=True),
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.KUBERNETES_VOLUMES, apply_to_task=True),
    ],
)
def extract_hidden_states(
    verifier_model: str,
    output_dir: str,
    pvc_path: str,
    dataset_name: str,
    verifier_model_pvc: Optional[str] = None,
    vllm_source: str = "managed",
    dataset_download_enabled: bool = False,
    downloaded_dataset_uri: Optional[str] = None,
    hidden_states_path: Optional[str] = None,
    vllm_endpoint: Optional[str] = None,
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: Optional[str] = None,
    data_extraction_max_samples: Optional[int] = None,
    data_extraction_regenerate_responses: bool = False,
    data_extraction_concurrency: int = 4,
    vllm_resource_gpu: int = 1,
    vllm_resource_memory: str = "96Gi",
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_readiness_timeout_minutes: int = 60,
    enable_progression_tracking: bool = True,
    metrics_port: int = 28080,
    metrics_poll_interval_seconds: int = 30,
    packages_to_install: Optional[str] = None,
    pip_index_urls: Optional[str] = None,
    training_envs: str = "",
    training_runtime: str = "speculator-model-opt-cuda",
    persistent_pvc: str = "",
    persistent_mount_path: str = "/mnt/persistent",
    download_model: bool = False,
    output_hidden_states: dsl.Output[dsl.Artifact] = None,
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Extract verifier hidden states with managed or remote vLLM.

    Dataset handling:
        Supported built-in names are ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered),
        ``ultrachat`` (HuggingFaceH4/ultrachat_200k), and ``gsm8k`` (openai/gsm8k).
        For custom data, use a local JSON/JSONL file under ``persistent_mount_path``;
        relative and absolute paths are converted to a Trainer PVC URI. Existing
        ``pvc://<claim>/<path>`` values keep their Trainer URI semantics. Arbitrary
        Hugging Face dataset IDs are not supported. When ``dataset_download_enabled``
        is true, ``downloaded_dataset_uri`` is used.
        If response regeneration is enabled, responses are generated with the verifier.

    vLLM mode:
        ``vllm_source="managed"`` uses an in-job sidecar. If ``download_model`` is
        false and ``verifier_model_pvc`` is set, the cached model path is used.
        Managed mode requires ``vllm_endpoint`` to be unset or blank.
        ``vllm_source="remote"`` requires a non-empty ``vllm_endpoint`` and
        PVC-backed verifier and hidden-state paths accessible to the external
        server. A missing remote endpoint raises an error; it never falls back to
        the managed sidecar. PVC-backed verifier models require exactly four
        ``target_layer_ids``; remote vLLM also requires them.

    PVC paths:
        ``verifier_model_pvc``, ``output_dir``, and ``hidden_states_path`` accept
        relative paths or explicit ``pvc://<claim>/<path>`` Trainer URIs. Relative
        paths resolve against ``persistent_pvc`` when set, or the KFP workspace PVC
        otherwise.

    For example, with ``persistent_pvc="shared"``, the relative
    ``verifier_model_pvc="models/verifier"`` is equivalent to
    ``pvc://shared/models/verifier``. The same applies to
    ``output_dir="speculator-output"`` and
    ``hidden_states_path="speculator-output/hidden_states"``.

    Args:
        verifier_model: Hugging Face verifier ID to download, or the verifier path used by the runtime.
        output_dir: Output directory as a relative PVC path or an explicit ``pvc://<claim>/<path>`` URI.
        pvc_path: KFP workspace mount path used to resolve the workspace PVC.
        dataset_name: ``magpie``, ``ultrachat``, ``gsm8k``, local JSON/JSONL path, or Trainer PVC URI.
        verifier_model_pvc: Cached verifier path as a relative PVC path or explicit PVC URI.
        vllm_source: ``managed`` for the in-job sidecar or ``remote`` for an external vLLM service.
        dataset_download_enabled: Use the downloaded dataset URI instead of ``dataset_name``.
        downloaded_dataset_uri: URI emitted by the dataset download step when enabled.
        hidden_states_path: Hidden-state directory as a relative PVC path or explicit PVC URI.
        vllm_endpoint: Required remote vLLM URL; unset for managed mode.
        total_seq_len: Maximum sequence length for dataset preprocessing and vLLM extraction; must be positive.
        speculator_type: Draft architecture; ``eagle3`` is currently the only supported type.
        target_layer_ids: Exactly four comma-separated verifier layer IDs; required
            for PVC-backed models and external vLLM.
        data_extraction_max_samples: Number of samples to process; zero or None means all.
        data_extraction_regenerate_responses: Regenerate responses for ``magpie``,
            ``ultrachat``, or ``gsm8k`` with the verifier.
        data_extraction_concurrency: Number of concurrent vLLM requests for hidden-state extraction.
        vllm_resource_gpu: GPUs requested by managed vLLM; the SDK supports exactly one.
        vllm_resource_memory: Memory requested by managed vLLM.
        vllm_gpu_memory_utilization: Fraction of GPU memory available to vLLM, in the range ``(0, 1]``.
        vllm_readiness_timeout_minutes: Maximum wait for the managed or remote vLLM
            endpoint to become ready, in minutes.
        enable_progression_tracking: Enable progress reporting.
        metrics_port: HTTP port for extraction metrics; valid range is ``1024``-``65535``.
        metrics_poll_interval_seconds: Seconds between metrics polls; valid range is ``5``-``300``.
        packages_to_install: Comma-separated packages for the Trainer job.
        pip_index_urls: Comma-separated package indexes for the Trainer job.
        training_envs: Comma-separated environment overrides for the Trainer job.
        training_runtime: ClusterTrainingRuntime used for extraction.
        persistent_pvc: Name of the shared PVC mounted at ``persistent_mount_path``.
        persistent_mount_path: Local mount path corresponding to ``persistent_pvc``.
        download_model: Download the verifier model to the shared PVC before extraction.
        output_hidden_states: Optional KFP artifact for the extracted hidden states.
        kubernetes_config: KFP task Kubernetes configuration passed through to Trainer.
    """
    from pathlib import Path

    # Pre-strip params consumed by inline logic below; all string inputs are also
    # stripped centrally in run_speculator.
    verifier_model = verifier_model.strip()
    output_dir = output_dir.strip()
    pvc_path = pvc_path.strip()
    dataset_name = dataset_name.strip()
    speculator_type = speculator_type.strip()
    vllm_resource_memory = vllm_resource_memory.strip()
    training_envs = training_envs.strip()
    training_runtime = training_runtime.strip()
    if persistent_pvc != persistent_pvc.strip():
        raise ValueError(f"persistent_pvc must not contain leading or trailing whitespace (got {persistent_pvc!r}).")
    persistent_mount_path = persistent_mount_path.strip()
    verifier_model_pvc = verifier_model_pvc.strip() if verifier_model_pvc else verifier_model_pvc
    downloaded_dataset_uri = downloaded_dataset_uri.strip() if downloaded_dataset_uri else downloaded_dataset_uri
    hidden_states_path = hidden_states_path.strip() if hidden_states_path else hidden_states_path
    target_layer_ids = target_layer_ids.strip() if target_layer_ids else target_layer_ids
    packages_to_install = packages_to_install.strip() if packages_to_install else packages_to_install
    pip_index_urls = pip_index_urls.strip() if pip_index_urls else pip_index_urls

    def resolve_local_dataset_path(dataset_uri: str) -> str:
        if not dataset_uri:
            return dataset_uri

        if dataset_uri.startswith("pvc://"):
            claim, separator, dataset_path = dataset_uri.removeprefix("pvc://").partition("/")
            if not claim or not separator or not dataset_path:
                raise ValueError("PVC dataset URIs must use pvc://<claim>/<path>")
            requested_path = Path(dataset_path)
            if claim != persistent_pvc:
                if requested_path.is_absolute() or ".." in requested_path.parts:
                    raise ValueError("PVC dataset paths must remain relative")
                if requested_path.suffix.lower() not in {".json", ".jsonl"}:
                    raise ValueError("dataset files on the shared PVC must end in .json or .jsonl")
                return dataset_uri
        elif "://" in dataset_uri:
            return dataset_uri

        else:
            candidate = Path(dataset_uri)
            if candidate.suffix.lower() not in {".json", ".jsonl"}:
                return dataset_uri
            dataset_path = dataset_uri

        if not persistent_pvc:
            raise ValueError("persistent_pvc is required for local JSON/JSONL dataset paths")

        pvc_root = Path(persistent_mount_path).resolve()
        candidate = Path(dataset_path)
        candidate = candidate if candidate.is_absolute() else pvc_root / candidate
        resolved = candidate.resolve()
        try:
            relative_path = resolved.relative_to(pvc_root)
        except ValueError as exc:
            raise ValueError("local dataset paths must remain within persistent_mount_path") from exc

        if not relative_path.parts:
            raise ValueError("dataset path must be a JSON or JSONL file under persistent_mount_path")
        if relative_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("dataset files on the shared PVC must end in .json or .jsonl")
        if not resolved.is_file():
            raise FileNotFoundError(f"dataset file not found on persistent_pvc: {resolved}")

        return f"pvc://{persistent_pvc}/{relative_path.as_posix()}"

    vllm_source = vllm_source.strip().lower()
    if vllm_source not in {"managed", "remote"}:
        raise ValueError("vllm_source must be either 'managed' or 'remote'")
    vllm_endpoint = (vllm_endpoint or "").strip() or None
    if vllm_source == "remote" and vllm_endpoint is None:
        raise ValueError("vllm_endpoint is required when vllm_source='remote'")
    if vllm_source == "managed" and vllm_endpoint is not None:
        raise ValueError("vllm_endpoint must be omitted when vllm_source='managed'")

    if vllm_source == "remote":
        verifier_model = verifier_model_pvc or verifier_model
    elif verifier_model_pvc and not download_model:
        verifier_model = verifier_model_pvc
    if dataset_download_enabled:
        if not downloaded_dataset_uri:
            raise ValueError("downloaded_dataset_uri is required when dataset download is enabled")
        dataset_name = downloaded_dataset_uri
    else:
        dataset_name = resolve_local_dataset_path(dataset_name)

    from speculator import run_speculator

    return run_speculator("data_only", locals())
