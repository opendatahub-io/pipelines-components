# Sft Pipeline ✨

## Overview 🧾

SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

A 4-stage ML pipeline for fine-tuning language models:

1) Dataset Download - Prepares training data from HuggingFace, S3, HTTP, or PVC
2) SFT Training - Fine-tunes using instructlab-training backend
3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
4) Model Registry - Registers trained model to Kubeflow Model Registry

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset location (hf://, s3://, https://, pvc://). |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split ratio (0.9 = 90% train). |
| `phase_02_train_man_batch` | `int` | `128` | Effective batch size per optimizer step. |
| `phase_02_train_man_epochs` | `int` | `1` | Number of training epochs. |
| `phase_02_train_man_gpu` | `int` | `1` | GPUs per worker. Keep at 1 to avoid /dev/shm issues. |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path). |
| `phase_02_train_man_tokens` | `int` | `10000` | Max tokens per GPU (memory cap). |
| `phase_02_train_man_workers` | `int` | `4` | Number of training pods. |
| `phase_03_eval_man_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, etc.). |
| `phase_04_registry_man_address` | `str` | `` | Model Registry address (empty = skip). |
| `phase_04_registry_man_author` | `str` | `pipeline` | Author name for the registered model. |
| `phase_04_registry_man_name` | `str` | `sft-model` | Model name in registry. |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version (major.minor.patch). |
| `phase_01_dataset_opt_hf_token` | `str` | `` | HuggingFace token for private datasets. |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit dataset to N samples (0 = all). |
| `phase_02_train_opt_annotations` | `str` | `` | Pod annotations as key=value,key=value. |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker. |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000` | Environment variables as KEY=VAL,KEY=VAL. |
| `phase_02_train_opt_hf_token` | `str` | `` | HuggingFace token for gated models. |
| `phase_02_train_opt_labels` | `str` | `` | Pod labels as key=value,key=value. |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate for training. |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Learning rate warmup steps. |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | LR scheduler type (cosine, linear). |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Maximum sequence length in tokens. |
| `phase_02_train_opt_memory` | `str` | `64Gi` | Memory per worker (e.g., 64Gi). |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker (auto or int). |
| `phase_02_train_opt_pull_secret` | `str` | `` | Pull secret for container registry. |
| `phase_02_train_opt_save_epoch` | `bool` | `True` | Save checkpoint at each epoch. |
| `phase_02_train_opt_save_full_state` | `bool` | `False` | Save full accelerate state at epoch. |
| `phase_02_train_opt_fsdp_sharding` | `str` | `FULL_SHARD` | FSDP strategy (FULL_SHARD, HYBRID_SHARD). |
| `phase_02_train_opt_save_samples` | `int` | `0` | Number of samples to save (0 = none). |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility. |
| `phase_02_train_opt_use_liger` | `bool` | `False` | Enable Liger kernel optimizations. |
| `phase_03_eval_opt_batch` | `str` | `auto` | Batch size for evaluation (auto or int). |
| `phase_03_eval_opt_gen_kwargs` | `dict` | `{}` | Generation kwargs for evaluation. |
| `phase_03_eval_opt_limit` | `int` | `-1` | Limit examples per task (-1 = no limit). |
| `phase_03_eval_opt_log_samples` | `bool` | `True` | Log individual evaluation samples. |
| `phase_03_eval_opt_model_args` | `dict` | `{}` | Model initialization arguments. |
| `phase_03_eval_opt_verbosity` | `str` | `INFO` | Logging verbosity (DEBUG, INFO, etc.). |
| `phase_04_registry_opt_description` | `str` | `` | Model description for registry. |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx). |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model Registry server port. |

## Metadata 🗂️

- **Name**: sft_pipeline
- **Tier**: core
- **Stability**: dev preview
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
  - sft
  - supervised_fine_tuning
  - llm
  - language_model
  - pipeline
- **Last Verified**: 2026-01-09 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
