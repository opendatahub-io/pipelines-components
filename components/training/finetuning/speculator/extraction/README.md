# Speculator Extraction ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract verifier hidden states using managed or external vLLM.

``dataset_name`` selects the source dataset. When regeneration is enabled, responses are regenerated through the verifier before hidden-state extraction. Managed mode uses the in-job vLLM sidecar; providing ``vllm_endpoint`` uses an external vLLM service and requires matching shared model and
hidden-state paths.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `verifier_model` | `str` | `None` |  |
| `output_dir` | `str` | `None` |  |
| `pvc_path` | `str` | `None` |  |
| `dataset_name` | `str` | `None` |  |
| `hidden_states_path` | `Optional[str]` | `None` |  |
| `vllm_endpoint` | `Optional[str]` | `None` |  |
| `total_seq_len` | `int` | `2048` |  |
| `speculator_type` | `str` | `eagle3` |  |
| `target_layer_ids` | `Optional[str]` | `None` |  |
| `data_extraction_max_samples` | `Optional[int]` | `None` |  |
| `data_extraction_regenerate_responses` | `bool` | `False` |  |
| `data_extraction_concurrency` | `int` | `4` |  |
| `vllm_resource_gpu` | `int` | `1` |  |
| `vllm_resource_memory` | `str` | `96Gi` |  |
| `vllm_gpu_memory_utilization` | `float` | `0.9` |  |
| `vllm_readiness_timeout_minutes` | `int` | `60` |  |
| `enable_progression_tracking` | `bool` | `True` |  |
| `metrics_port` | `int` | `28080` |  |
| `metrics_poll_interval_seconds` | `int` | `30` |  |
| `packages_to_install` | `Optional[str]` | `None` |  |
| `pip_index_urls` | `Optional[str]` | `None` |  |
| `training_envs` | `str` | `""` |  |
| `training_runtime` | `str` | `speculator-model-opt-cuda` |  |
| `persistent_pvc` | `str` | `""` |  |
| `persistent_mount_path` | `str` | `/mnt/persistent` |  |
| `download_model` | `bool` | `False` |  |
| `output_hidden_states` | `dsl.Output[dsl.Artifact]` | `None` |  |
| `kubernetes_config` | `dsl.TaskConfig` | `None` |  |

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
- **Last Verified**: 2026-09-10 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
