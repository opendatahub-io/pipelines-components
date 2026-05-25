# Download Model ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Download a HuggingFace model to a PVC for caching.

If the model is already present on the PVC, skips the download.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `model_name` | `str` | `None` | HuggingFace model ID (e.g. 'mistralai/Mistral-7B-Instruct-v0.3'). |
| `model_cache_pvc` | `str` | `None` | Name of the PVC to store models (unused here, mounted via pipeline). |
| `model_cache_mount` | `str` | `/mnt/models` | Mount path for the model cache PVC. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The PVC sub-path where the model is stored. |

## Metadata 🗂️

- **Name**: download_model
- **Description**: Download a HuggingFace model to a PVC for caching. Skips the download if the model is already present.

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: HuggingFace Hub, Version: >=0.20.0
- **Tags**:
  - data_processing
  - model_download
  - huggingface
  - caching
  - rag
- **Last Verified**: 2026-04-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - szaher
  - Reviewers:
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
