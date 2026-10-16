# Speculator Online ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract and train a draft model with a managed vLLM sidecar.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `verifier_model` | `str` | `None` |  |
| `output_dir` | `str` | `None` |  |
| `pvc_path` | `str` | `None` |  |
| `dataset_name` | `str` | `None` |  |
| `verifier_model_pvc` | `Optional[str]` | `None` |  |
| `persistent_pvc` | `str` | `""` |  |
| `persistent_mount_path` | `str` | `/mnt/persistent` |  |
| `total_seq_len` | `int` | `2048` |  |
| `speculator_type` | `str` | `eagle3` |  |
| `target_layer_ids` | `Optional[str]` | `None` |  |
| `data_extraction_max_samples` | `Optional[int]` | `None` |  |
| `data_extraction_regenerate_responses` | `bool` | `False` |  |
| `data_extraction_concurrency` | `int` | `4` |  |
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
| `vllm_resource_gpu` | `int` | `1` |  |
| `vllm_resource_memory` | `str` | `96Gi` |  |
| `vllm_gpu_memory_utilization` | `float` | `0.9` |  |
| `vllm_readiness_timeout_minutes` | `int` | `60` |  |
| `enable_progression_tracking` | `bool` | `True` |  |
| `metrics_port` | `int` | `28080` |  |
| `metrics_poll_interval_seconds` | `int` | `30` |  |
| `packages_to_install` | `Optional[str]` | `None` |  |
| `pip_index_urls` | `Optional[str]` | `None` |  |
| `training_envs` | `str` | `""` |  |
| `training_runtime` | `str` | `vllm-extract-cuda` |  |
| `output_model` | `dsl.Output[dsl.Model]` | `None` |  |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` |  |
| `kubernetes_config` | `dsl.TaskConfig` | `None` |  |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` |  |

## Metadata 🗂️

- **Name**: speculator_online
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
  - online_training
- **Last Verified**: 2026-09-10 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
