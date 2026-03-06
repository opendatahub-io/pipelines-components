# Sft Minimal Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

A 4-stage ML pipeline for fine-tuning language models:

1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP 2) SFT Training - Fine-tunes using instructlab-training backend 3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.) 4) Model Registry - Registers trained model to Kubeflow Model Registry

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split (0.9 = 90% train/10% eval, 1.0 = no split, all for training) |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step). Start with 128 |
| `phase_02_train_man_epochs` | `int` | `1` | Number of training epochs. 1 is often sufficient |
| `phase_02_train_man_gpu` | `int` | `1` | GPUs per worker. KEEP AT 1 to avoid /dev/shm issues |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_tokens` | `int` | `10000` | Max tokens per GPU (memory cap). 10000 for SFT |
| `phase_02_train_man_workers` | `int` | `4` | Number of training pods. 4 pods × 1 GPU = 4 total GPUs |
| `phase_03_eval_man_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.) |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_name` | `str` | `sft-model` | Model name in registry |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, NCCL_DEBUG=INFO, NCCL_P2P_DISABLE=1, INSTRUCTLAB_NCCL_TIMEOUT_MS=60000` | Env vars (KEY=VAL,...) with NCCL timeout and memory optimization |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate (1e-6 to 1e-4). 5e-6 recommended |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_fsdp_sharding` | `str` | `FULL_SHARD` | FSDP strategy (FULL_SHARD, HYBRID_SHARD, NO_SHARD) |
| `phase_02_train_opt_use_liger` | `bool` | `False` | Enable Liger kernel optimizations |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime to use. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model Registry server port. |

## Metadata 🗂️

- **Name**: sft_minimal_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - fine_tuning
  - sft
  - supervised_fine_tuning
  - minimal
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
