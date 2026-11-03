# Speculator Extraction ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract verifier hidden states with managed or remote vLLM.

Dataset handling: Supported built-in names are ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered), ``ultrachat`` (HuggingFaceH4/ultrachat_200k), and ``gsm8k`` (openai/gsm8k). For custom data, use a local JSON/JSONL file under ``persistent_mount_path``; relative and absolute paths are
converted to a Trainer PVC URI. Existing ``pvc://<claim>/<path>`` values keep their Trainer URI semantics. Arbitrary Hugging Face dataset IDs are not supported. When ``dataset_download_enabled`` is true, ``downloaded_dataset_uri`` is used. If response regeneration is enabled, responses are generated
with the verifier.

vLLM mode: ``vllm_source="managed"`` uses an in-job sidecar. If ``download_model`` is false and ``verifier_model_pvc`` is set, the cached model path is used. Managed mode requires ``vllm_endpoint`` to be unset or blank. ``vllm_source="remote"`` requires a non-empty ``vllm_endpoint`` and PVC-backed
verifier and hidden-state paths accessible to the external server. A missing remote endpoint raises an error; it never falls back to the managed sidecar. PVC-backed verifier models require exactly four ``target_layer_ids``; remote vLLM also requires them.

PVC paths: ``verifier_model_pvc``, ``output_dir``, and ``hidden_states_path`` accept relative paths or explicit ``pvc://<claim>/<path>`` Trainer URIs. Relative paths resolve against ``persistent_pvc`` when set, or the KFP workspace PVC otherwise.

For example, with ``persistent_pvc="shared"``, the relative ``verifier_model_pvc="models/verifier"`` is equivalent to ``pvc://shared/models/verifier``. The same applies to ``output_dir="speculator-output"`` and ``hidden_states_path="speculator-output/hidden_states"``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `verifier_model` | `str` | `None` | Hugging Face verifier ID to download, or the verifier path used by the runtime. |
| `output_dir` | `str` | `None` | Output directory as a relative PVC path or an explicit ``pvc://<claim>/<path>`` URI. |
| `pvc_path` | `str` | `None` | KFP workspace mount path used to resolve the workspace PVC. |
| `dataset_name` | `str` | `None` | ``magpie``, ``ultrachat``, ``gsm8k``, local JSON/JSONL path, or Trainer PVC URI. |
| `verifier_model_pvc` | `Optional[str]` | `None` | Cached verifier path as a relative PVC path or explicit PVC URI. |
| `vllm_source` | `str` | `managed` | ``managed`` for the in-job sidecar or ``remote`` for an external vLLM service. |
| `dataset_download_enabled` | `bool` | `False` | Use the downloaded dataset URI instead of ``dataset_name``. |
| `downloaded_dataset_uri` | `Optional[str]` | `None` | URI emitted by the dataset download step when enabled. |
| `hidden_states_path` | `Optional[str]` | `None` | Hidden-state directory as a relative PVC path or explicit PVC URI. |
| `vllm_endpoint` | `Optional[str]` | `None` | Required remote vLLM URL; unset for managed mode. |
| `total_seq_len` | `int` | `2048` | Maximum sequence length for dataset preprocessing and vLLM extraction; must be positive. |
| `speculator_type` | `str` | `eagle3` | Draft architecture; ``eagle3`` is currently the only supported type. |
| `target_layer_ids` | `Optional[str]` | `None` | Exactly four comma-separated verifier layer IDs; required for PVC-backed models and external vLLM. |
| `data_extraction_max_samples` | `Optional[int]` | `None` | Number of samples to process; zero or None means all. |
| `data_extraction_regenerate_responses` | `bool` | `False` | Regenerate responses for ``magpie``, ``ultrachat``, or ``gsm8k`` with the verifier. |
| `data_extraction_concurrency` | `int` | `4` | Number of concurrent vLLM requests for hidden-state extraction. |
| `vllm_resource_gpu` | `int` | `1` | GPUs requested by managed vLLM; the SDK supports exactly one. |
| `vllm_resource_memory` | `str` | `96Gi` | Memory requested by managed vLLM. |
| `vllm_gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory available to vLLM, in the range ``(0, 1]``. |
| `vllm_readiness_timeout_minutes` | `int` | `60` | Maximum wait for the managed or remote vLLM endpoint to become ready, in minutes. |
| `enable_progression_tracking` | `bool` | `True` | Enable progress reporting. |
| `metrics_port` | `int` | `28080` | HTTP port for extraction metrics; valid range is ``1024``-``65535``. |
| `metrics_poll_interval_seconds` | `int` | `30` | Seconds between metrics polls; valid range is ``5``-``300``. |
| `packages_to_install` | `Optional[str]` | `None` | Comma-separated packages for the Trainer job. |
| `pip_index_urls` | `Optional[str]` | `None` | Comma-separated package indexes for the Trainer job. |
| `training_envs` | `str` | `""` | Comma-separated environment overrides for the Trainer job. |
| `training_runtime` | `str` | `speculator-model-opt-cuda` | ClusterTrainingRuntime used for extraction. |
| `persistent_pvc` | `str` | `""` | Name of the shared PVC mounted at ``persistent_mount_path``. |
| `persistent_mount_path` | `str` | `/mnt/persistent` | Local mount path corresponding to ``persistent_pvc``. |
| `download_model` | `bool` | `False` | Download the verifier model to the shared PVC before extraction. |
| `output_hidden_states` | `dsl.Output[dsl.Artifact]` | `None` | Optional KFP artifact for the extracted hidden states. |
| `kubernetes_config` | `dsl.TaskConfig` | `None` | KFP task Kubernetes configuration passed through to Trainer. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` |  |

## Metadata 🗂️

- **Name**: speculator_extraction
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - speculator
  - speculative_decoding
  - hidden_state_extraction
- **Last Verified**: 2026-10-20 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina
    - JaZeeGH
    - Sridhar1030
  - Reviewers:
    - ChughShilpa
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
