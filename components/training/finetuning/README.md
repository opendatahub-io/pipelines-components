# Finetuning ✨

## Overview 🧾

Train model using TrainingHub (OSFT/SFT). Outputs model artifact and metrics.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pvc_path` | `str` | `None` | Workspace PVC root path (use dsl.WORKSPACE_PATH_PLACEHOLDER). |
| `output_model` | `dsl.Output[dsl.Model]` | `None` | Output model artifact. |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | Output training metrics artifact. |
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` | Input training dataset artifact. |
| `training_base_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or local path). |
| `training_algorithm` | `str` | `OSFT` | Training algorithm: OSFT or SFT. |
| `training_effective_batch_size` | `int` | `128` | Effective batch size per optimizer step. |
| `training_max_tokens_per_gpu` | `int` | `64000` | Max tokens per GPU (memory cap). |
| `training_max_seq_len` | `int` | `8192` | Max sequence length in tokens. |
| `training_learning_rate` | `Optional[float]` | `None` | Learning rate (default: 5e-6). |
| `training_backend` | `str` | `mini-trainer` | Backend: mini-trainer (OSFT) or instructlab-training (SFT). |
| `training_lr_warmup_steps` | `Optional[int]` | `None` | Learning rate warmup steps. |
| `training_checkpoint_at_epoch` | `Optional[bool]` | `None` | Save checkpoint at each epoch. |
| `training_num_epochs` | `Optional[int]` | `None` | Number of training epochs. |
| `training_data_output_dir` | `Optional[str]` | `None` | Directory for processed training data. |
| `training_hf_token` | `str` | `` | HuggingFace token for gated models. |
| `training_pull_secret` | `str` | `` | Pull secret for container registry. |
| `training_envs` | `str` | `` | Environment overrides as KEY=VAL,KEY=VAL. |
| `training_resource_cpu_per_worker` | `str` | `8` | CPU cores per worker. |
| `training_resource_gpu_per_worker` | `int` | `1` | GPUs per worker. |
| `training_resource_memory_per_worker` | `str` | `32Gi` | Memory per worker (e.g., 32Gi). |
| `training_resource_num_procs_per_worker` | `str` | `auto` | Processes per worker (auto or int). |
| `training_resource_num_workers` | `int` | `1` | Number of training pods. |
| `training_metadata_labels` | `str` | `` | Pod labels as key=value,key=value. |
| `training_metadata_annotations` | `str` | `` | Pod annotations as key=value,key=value. |
| `training_unfreeze_rank_ratio` | `float` | `0.25` | [OSFT] Fraction of parameters to unfreeze. |
| `training_osft_memory_efficient_init` | `bool` | `True` | [OSFT] Use memory-efficient initialization. |
| `training_target_patterns` | `str` | `` | [OSFT] Target layer patterns (comma-separated). |
| `training_seed` | `Optional[int]` | `None` | Random seed for reproducibility. |
| `training_use_liger` | `Optional[bool]` | `None` | Enable Liger kernel optimizations. |
| `training_use_processed_dataset` | `Optional[bool]` | `None` | Use pre-processed dataset. |
| `training_unmask_messages` | `Optional[bool]` | `None` | Unmask assistant messages during training. |
| `training_lr_scheduler` | `Optional[str]` | `None` | LR scheduler type (cosine, linear, etc.). |
| `training_lr_scheduler_kwargs` | `str` | `` | LR scheduler kwargs as key=val,key=val. |
| `training_save_final_checkpoint` | `Optional[bool]` | `None` | Save final checkpoint after training. |
| `training_save_samples` | `Optional[int]` | `None` | [SFT] Number of samples to save. |
| `training_accelerate_full_state_at_epoch` | `Optional[bool]` | `None` | [SFT] Save full accelerate state. |
| `training_fsdp_sharding_strategy` | `Optional[str]` | `None` | [SFT] FSDP sharding strategy. |
| `kubernetes_config` | `dsl.TaskConfig` | `None` | KFP TaskConfig for volumes/env/resources passthrough. |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `str` |  |

## Metadata 🗂️

- **Name**: finetuning
- **Tier**: core
- **Stability**: dev preview
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
  - finetuning
  - osft
  - sft
  - llm
  - language_model
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
