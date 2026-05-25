# Deploy Embedding Model ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Deploy a text embedding model using KServe InferenceService.

Uses vLLM with ``--task embedding`` on GPU to serve an OpenAI-compatible ``/v1/embeddings`` endpoint.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `model_name` | `str` | `None` | HuggingFace model ID (e.g. 'ibm-granite/granite-embedding-125m-english'). |
| `namespace` | `str` | `None` | Namespace to deploy the InferenceService. |
| `serving_runtime_name` | `str` | `embedding-runtime` | Name of the embedding ServingRuntime CR. |
| `runtime_image` | `str` | `_DEFAULT_RUNTIME_IMAGE` | Container image for the vLLM embedding server. |
| `min_replicas` | `int` | `1` | Minimum number of replicas. |
| `max_replicas` | `int` | `1` | Maximum number of replicas. |
| `cpu_requests` | `str` | `2` | CPU requests per replica. |
| `cpu_limits` | `str` | `4` | CPU limits per replica. |
| `memory_requests` | `str` | `4Gi` | Memory requests per replica. |
| `memory_limits` | `str` | `8Gi` | Memory limits per replica. |
| `gpu_count` | `int` | `1` | Number of GPUs per replica. |
| `max_model_len` | `int` | `512` | Maximum sequence length for the embedding model. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The embedding service endpoint URL. |

## Metadata 🗂️

- **Name**: deploy_embedding_model
- **Description**: Deploy a text embedding model on OpenShift AI using a KServe InferenceService with a TEI-compatible ServingRuntime.

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: OpenShift AI (KServe), Version: >=2.10.0
- **Tags**:
  - embeddings
  - model_serving
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
