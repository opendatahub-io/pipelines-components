# Lora Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

LoRA Training Pipeline - Parameter-efficient fine-tuning.

A 4-stage ML pipeline for fine-tuning language models with LoRA:

    1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
    2) LoRA Training - Fine-tunes using unsloth backend (low-rank adapters)
    3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
    4) Model Registry - Registers trained model to Kubeflow Model Registry

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split (0.9 = 90%/10%, 1.0 = no split) |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step) |
| `phase_02_train_man_train_epochs` | `int` | `2` | Number of training epochs. LoRA typically needs 2-3 |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker |
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_train_tokens` | `int` | `32000` | Max tokens per GPU (memory cap). 32000 for LoRA |
| ~~`phase_02_train_man_train_workers`~~ | `int` | `1` | ~~Number of training pods~~ — **Disabled: LoRA (unsloth) only supports single-node training. Hardcoded to 1 until multi-node support is added.** |
| `phase_02_train_man_lora_r` | `int` | `16` | [LoRA] Rank of the low-rank matrices (4, 8, 16, 32, 64) |
| `phase_02_train_man_lora_alpha` | `int` | `32` | [LoRA] Scaling factor (typically 2x lora_r) |
| `phase_03_eval_man_eval_tasks` | `list` | `['arc_easy']` | lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.) |
| `phase_04_registry_man_address` | `str` | `` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_author` | `str` | `pipeline` | Author name for the registered model |
| `phase_04_registry_man_reg_name` | `str` | `lora-model` | Model name in registry |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_annotations` | `str` | `` | K8s annotations (key=val,...) |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker |
| `phase_02_train_opt_env_vars` | `str` | `` | Env vars (KEY=VAL,...) |
| `phase_02_train_opt_labels` | `str` | `` | K8s labels (key=val,...) |
| `phase_02_train_opt_learning_rate` | `float` | `0.0002` | Learning rate. 2e-4 recommended for LoRA |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | LR schedule (cosine, linear, constant) |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Warmup steps before full LR |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_memory` | `str` | `32Gi` | RAM per worker |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker ('auto' = one per GPU) |
| `phase_02_train_opt_save_epoch` | `bool` | `False` | Save checkpoint at each epoch |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility |
| `phase_02_train_opt_use_liger` | `bool` | `True` | Enable Liger kernel optimizations |
| `phase_02_train_opt_lora_dropout` | `float` | `0.0` | [LoRA] Dropout rate for LoRA layers |
| `phase_02_train_opt_lora_target_modules` | `str` | `` | [LoRA] Modules to apply LoRA (empty=auto-detect) |
| `phase_02_train_opt_lora_use_rslora` | `bool` | `False` | [LoRA] Use Rank-Stabilized LoRA |
| `phase_02_train_opt_lora_use_dora` | `bool` | `False` | [LoRA] Use Weight-Decomposed LoRA (DoRA) |
| `phase_02_train_opt_lora_load_in_4bit` | `bool` | `True` | [QLoRA] Enable 4-bit quantization |
| `phase_02_train_opt_lora_load_in_8bit` | `bool` | `False` | [QLoRA] Enable 8-bit quantization |
| `phase_02_train_opt_lora_sample_packing` | `bool` | `False` | [LoRA] Pack multiple samples for efficiency |
| `phase_03_eval_opt_batch` | `str` | `auto` | Eval batch size ('auto' or integer) |
| `phase_03_eval_opt_gen_kwargs` | `dict` | `{}` | Generation params dict (max_tokens, temperature) |
| `phase_03_eval_opt_limit` | `int` | `-1` | Max samples per task (-1 = all) |
| `phase_03_eval_opt_log_samples` | `bool` | `True` | Log individual predictions |
| `phase_03_eval_opt_model_args` | `dict` | `{}` | Model init args dict (dtype, gpu_memory_utilization) |
| `phase_03_eval_opt_verbosity` | `str` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `phase_04_registry_opt_description` | `str` | `` | Model description |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx, tensorflow) |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port |

## Metadata 🗂️

- **Name**: lora_pipeline
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
  - lora
  - peft
  - parameter_efficient
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
