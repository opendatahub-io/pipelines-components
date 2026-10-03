# Speculator Training ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train a draft model from hidden states already stored on the PVC.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `verifier_model` | `str` | `None` |  |
| `output_dir` | `str` | `None` |  |
| `pvc_path` | `str` | `None` |  |
| `hidden_states_path` | `str` | `None` |  |
| `verifier_model_pvc` | `Optional[str]` | `None` |  |
| `training_data_path` | `Optional[str]` | `None` |  |
| `total_seq_len` | `int` | `2048` |  |
| `speculator_type` | `str` | `eagle3` |  |
| `target_layer_ids` | `Optional[str]` | `None` |  |
| `training_epochs` | `int` | `3` |  |
| `training_lr` | `float` | `0.0001` |  |
| `training_draft_vocab_size` | `Optional[int]` | `None` |  |
| `training_num_layers` | `int` | `1` |  |
| `training_ttt_steps` | `int` | `1` |  |
| `training_norm_before_residual` | `bool` | `True` |  |
| `training_norm_before_fc` | `bool` | `False` |  |
| `training_embed_requires_grad` | `bool` | `False` |  |
| `training_hidden_states_dtype` | `str` | `bfloat16` |  |
| `training_scheduler_type` | `str` | `linear` |  |
| `training_scheduler_warmup_steps` | `Optional[int]` | `None` |  |
| `training_scheduler_total_steps` | `Optional[int]` | `None` |  |
| `training_scheduler_num_cosine_cycles` | `float` | `0.5` |  |
| `training_checkpoint_freq` | `float` | `1.0` |  |
| `training_save_best` | `bool` | `False` |  |
| `training_log_freq` | `int` | `1` |  |
| `training_resume_from_checkpoint` | `bool` | `False` |  |
| `training_from_pretrained` | `Optional[str]` | `None` |  |
| `training_resource_cpu` | `str` | `4` |  |
| `training_resource_gpu` | `int` | `1` |  |
| `training_resource_memory` | `str` | `64Gi` |  |
| `data_extraction_concurrency` | `int` | `4` |  |
| `enable_progression_tracking` | `bool` | `True` |  |
| `metrics_port` | `int` | `28080` |  |
| `metrics_poll_interval_seconds` | `int` | `30` |  |
| `packages_to_install` | `Optional[str]` | `None` |  |
| `pip_index_urls` | `Optional[str]` | `None` |  |
| `training_envs` | `str` | `""` |  |
| `training_runtime` | `str` | `speculator-model-opt-cuda` |  |
| `persistent_pvc` | `str` | `""` |  |
| `persistent_mount_path` | `str` | `/mnt/persistent` |  |
| `output_model` | `dsl.Output[dsl.Model]` | `None` |  |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` |  |
| `kubernetes_config` | `dsl.TaskConfig` | `None` |  |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` |  |

## Metadata 🗂️

- **Name**: speculator_training
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - speculator
  - speculative_decoding
  - draft_model_training
- **Last Verified**: 2026-09-10 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
