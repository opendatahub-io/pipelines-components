# Yoda Data Processor ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Prepare the training and evaluation datasets by downloading and preprocessing.

Downloads the yoda_sentences dataset from HuggingFace, renames columns to match the expected format for training (prompt/completion), splits into train/eval sets, and saves them as output artifacts.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `yoda_train_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset for training. |
| `yoda_eval_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset for evaluation. |
| `yoda_input_dataset` | `str` | `dvgodoy/yoda_sentences` | Dataset to download from HuggingFace |
| `train_split_ratio` | `float` | `0.8` | Ratio for training (0.0-1.0), defaults to 0.8 |

## Metadata 🗂️

- **Name**: yoda_data_processor
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: HuggingFace Datasets, Version: >=4.4.2
- **Tags**:
  - data_processing
  - dataset_preparation
  - text_processing
  - yoda_speak
  - translation
- **Last Verified**: 2025-12-19 11:30:16+00:00
- **Owners**:
  - Approvers:
    - mprahl
    - nsingla
    - HumairAK

## Additional Resources 📚

- **Dataset**: [https://huggingface.co/datasets/dvgodoy/yoda_sentences](https://huggingface.co/datasets/dvgodoy/yoda_sentences)
