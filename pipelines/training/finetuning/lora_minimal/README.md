# Lora Minimal Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

LoRA Minimal Training Pipeline - Parameter-efficient fine-tuning.

A minimal 4-stage ML pipeline for fine-tuning language models with LoRA:

1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP 2) LoRA Training - Fine-tunes using unsloth backend (low-rank adapters) 3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.) 4) Model Registry - Registers trained model to Kubeflow Model Registry

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split (0.9 = 90% train/10% eval, 1.0 = no split, all for training) |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step) |
| `phase_02_train_man_train_epochs` | `int` | `2` | Number of training epochs. LoRA typically needs 2-3 |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker |
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_train_tokens` | `int` | `32000` | Max tokens per GPU (memory cap). 32000 for LoRA |
| `phase_02_train_man_lora_r` | `int` | `16` | [LoRA] Rank of the low-rank matrices (4, 8, 16, 32, 64) |
| `phase_02_train_man_lora_alpha` | `int` | `32` | [LoRA] Scaling factor (typically 2x lora_r) |
| `phase_03_eval_man_eval_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.) |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_name` | `str` | `lora-model` | Model name in registry |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_learning_rate` | `float` | `0.0002` | Learning rate. 2e-4 recommended for LoRA |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_use_liger` | `bool` | `True` | Enable Liger kernel optimizations |
| `phase_02_train_opt_lora_dropout` | `float` | `0.0` | [LoRA] Dropout rate for LoRA layers |
| `phase_02_train_opt_lora_target_modules` | `str` | `""` | [LoRA] Modules to apply LoRA (empty=auto-detect) |
| `phase_02_train_opt_lora_load_in_4bit` | `bool` | `True` | [QLoRA] Enable 4-bit quantization (cannot use with 8-bit) |
| `phase_02_train_opt_lora_load_in_8bit` | `bool` | `False` | [QLoRA] Enable 8-bit quantization (cannot use with 4-bit) |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime to use. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port |

## Metadata 🗂️

- **Name**: lora_minimal_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Model Registry, Version: >=0.3.4
- **Tags**:
  - training
  - fine_tuning
  - lora
  - peft
  - parameter_efficient
  - minimal
  - llm
  - language_model
  - pipeline
- **Last Verified**: 2026-01-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
