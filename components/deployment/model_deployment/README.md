# Model Deployment ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Deploy a model on OpenShift AI using vLLM InferenceService.

Creates a vLLM NVIDIA GPU ServingRuntime and InferenceService matching the RHOAI dashboard deployment pattern. Uses a PVC-cached model to avoid re-downloading on every pod restart.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `model_name` | `str` | `None` | HuggingFace model name (e.g. 'mistralai/Mistral-7B-Instruct-v0.3'). |
| `namespace` | `str` | `None` | Namespace to deploy into. |
| `model_dir` | `str` | `None` | Sub-path on the PVC where model files are stored (output from download_model). |
| `model_cache_pvc` | `str` | `None` | Name of the PVC containing cached model weights. |
| `hardware_profile_name` | `str` | `gpu-profile` | HardwareProfile CR name for GPU resources. |
| `hardware_profile_namespace` | `str` | `redhat-ods-applications` | Namespace of the HardwareProfile CR. |
| `min_replicas` | `int` | `1` | Minimum number of replicas. |
| `max_replicas` | `int` | `1` | Maximum number of replicas. |
| `gpu_count` | `int` | `1` | Number of GPUs per replica. |
| `max_model_len` | `int` | `4096` | Maximum context length (limits KV cache memory usage). |
| `cpu_requests` | `str` | `2` | CPU requests for the predictor pod. |
| `memory_requests` | `str` | `8Gi` | Memory requests for the predictor pod. |
| `cpu_limits` | `str` | `2` | CPU limits for the predictor pod. |
| `memory_limits` | `str` | `8Gi` | Memory limits for the predictor pod. |
| `force_recreate` | `bool` | `False` | If True, delete and recreate the InferenceService (causes downtime). If False (default), patch in place. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The inference endpoint URL. |

## Metadata 🗂️

- **Name**: model_deployment
- **Description**: Deploy a model on OpenShift AI using vLLM ServingRuntime via KServe InferenceService.

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KServe, Version: >=0.11.0
    - Name: vLLM ServingRuntime, Version: >=0.4.0
- **Tags**:
  - deployment
  - model_serving
  - vllm
  - kserve
  - rag
- **Last Verified**: 2026-04-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - szaher
  - Reviewers:
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
