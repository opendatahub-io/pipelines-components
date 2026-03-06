# Osft Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

OSFT Training Pipeline - Continual learning without catastrophic forgetting.

A 4-stage ML pipeline for fine-tuning language models with OSFT:

1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP 2) OSFT Training - Fine-tunes using mini-trainer backend (orthogonal subspace) 3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.) 4) Model Registry - Registers trained model to Kubeflow Model Registry

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split (0.9 = 90%/10%, 1.0 = no split) |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step) |
| `phase_02_train_man_train_epochs` | `int` | `1` | Number of training epochs. OSFT typically needs 1-2 |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker. OSFT handles multi-GPU well |
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_train_tokens` | `int` | `64000` | Max tokens per GPU (memory cap). 64000 for OSFT |
| `phase_02_train_man_train_unfreeze` | `float` | `0.25` | [OSFT] Fraction to unfreeze (0.1=minimal, 0.25=balanced, 0.5=strong) |
| `phase_02_train_man_train_workers` | `int` | `1` | Number of training pods. OSFT efficient single-node (1) |
| `phase_03_eval_man_eval_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.) |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_author` | `str` | `pipeline` | Author name for the registered model |
| `phase_04_registry_man_reg_name` | `str` | `osft-model` | Model name in registry |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_annotations` | `str` | `""` | K8s annotations (key=val,...) |
| `phase_02_train_opt_cpu` | `str` | `8` | CPU cores per worker. 8 recommended for OSFT |
| `phase_02_train_opt_env_vars` | `str` | `""` | Env vars (KEY=VAL,...). OSFT typically doesn't need special vars |
| `phase_02_train_opt_labels` | `str` | `""` | K8s labels (key=val,...) |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate (1e-6 to 1e-4). 5e-6 recommended |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | [OSFT] LR schedule (cosine, linear, constant) |
| `phase_02_train_opt_lr_scheduler_kwargs` | `str` | `""` | [OSFT] Extra scheduler params (key=val,...) |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Warmup steps before full LR |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_memory` | `str` | `32Gi` | RAM per worker. 32Gi usually sufficient for OSFT |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker ('auto' = one per GPU) |
| `phase_02_train_opt_processed_data` | `bool` | `False` | [OSFT] True if dataset already has tokenized input_ids |
| `phase_02_train_opt_save_epoch` | `bool` | `False` | Save checkpoint at each epoch. Usually False for OSFT |
| `phase_02_train_opt_save_final` | `bool` | `True` | [OSFT] Save final checkpoint after all epochs |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility |
| `phase_02_train_opt_target_patterns` | `str` | `""` | [OSFT] Module patterns to unfreeze (empty=auto) |
| `phase_02_train_opt_unmask` | `bool` | `False` | [OSFT] Unmask all tokens (False=assistant only) |
| `phase_02_train_opt_use_liger` | `bool` | `True` | [OSFT] Enable Liger kernel optimizations. Recommended |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime to use. |
| `phase_03_eval_opt_batch` | `str` | `auto` | Eval batch size ('auto' or integer) |
| `phase_03_eval_opt_gen_kwargs` | `dict` | `{}` | Generation params dict (max_tokens, temperature) |
| `phase_03_eval_opt_limit` | `int` | `-1` | Max samples per task (-1 = all) |
| `phase_03_eval_opt_log_samples` | `bool` | `True` | Log individual predictions |
| `phase_03_eval_opt_model_args` | `dict` | `{}` | Model init args dict (dtype, gpu_memory_utilization) |
| `phase_03_eval_opt_verbosity` | `str` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `phase_04_registry_opt_description` | `str` | `""` | Model description |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx, tensorflow) |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port |

## Metadata 🗂️

- **Name**: osft_pipeline
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
  - osft
  - orthogonal_subspace_fine_tuning
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
