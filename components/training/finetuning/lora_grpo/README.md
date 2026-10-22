# Lora Grpo ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train using LoRA GRPO (Group Relative Policy Optimization).

Uses ART backend with co-located vLLM for single-GPU RLVR training. Trains LoRA adapters on tool-calling agents, then merges adapters into the base model via PEFT merge_and_unload() for downstream deployment.

Training is delegated entirely to Training Hub's built-in lora_grpo() algorithm (no custom func is passed to TrainingHubTrainer). This ensures the SDK wraps the training call with a `__main__` guard, which is required because ART uses `multiprocessing.spawn` internally.

After the TrainJob completes, this component reads the adapter checkpoints from the shared PVC and merges them into the base model on CPU.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pvc_path` | `str` | `None` | Workspace PVC root path (use dsl.WORKSPACE_PATH_PLACEHOLDER). |
| `output_model` | `dsl.Output[dsl.Model]` | `None` | Output model artifact (merged model). |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | Output training metrics artifact. |
| `training_run_id` | `str` | `""` | Unique identifier for this pipeline run. The pipeline supplies the KFP run ID so checkpoints and Training Hub dataset caches from previous runs are not reused accidentally on a persistent PVC. If empty, a UUID is used. |
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` | Input training dataset artifact (tool-call traces). |
| `training_base_model` | `str` | `Qwen/Qwen3-4B` | Base model (HuggingFace ID or local path). |
| `training_data_path` | `str` | `""` | Dataset path or HuggingFace ID (e.g., Agent-Ark/Toucan-1.5M). When empty, uses the dataset artifact. When set, overrides the artifact. |
| `training_num_iterations` | `int` | `5` | Number of GRPO training iterations. |
| `training_completion_timeout_seconds` | `int` | `1800` |  |
| `training_group_size` | `int` | `4` |  |
| `training_prompt_batch_size` | `int` | `50` |  |
| `training_n_train` | `int` | `200` |  |
| `training_learning_rate` | `Optional[float]` | `None` |  |
| `training_gpu_memory_utilization` | `float` | `0.45` |  |
| `training_enforce_eager` | `bool` | `True` |  |
| `training_data_config` | `str` | `Qwen3` |  |
| `training_lora_r` | `int` | `16` |  |
| `training_lora_alpha` | `int` | `8` |  |
| `training_envs` | `str` | `""` |  |
| `training_resource_cpu_per_worker` | `str` | `4` |  |
| `training_resource_gpu_per_worker` | `int` | `1` |  |
| `training_resource_memory_per_worker` | `str` | `64Gi` |  |
| `training_metadata_labels` | `str` | `""` |  |
| `training_metadata_annotations` | `str` | `""` |  |
| `training_runtime` | `str` | `training-hub` |  |
| `training_pvc_name` | `str` | `""` |  |
| `training_use_hf_direct` | `bool` | `False` |  |
| `kubernetes_config` | `dsl.TaskConfig` | `None` |  |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The path to this run's ``training_results.json`` on the shared PVC. |

## Metadata 🗂️

- **Name**: lora_grpo
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Training Hub, Version: >=0.9.2
    - Name: ART (OpenPipe), Version: >=0.1.0
- **Tags**:
  - training
  - fine_tuning
  - grpo
  - lora
  - peft
  - rlvr
  - reinforcement_learning
  - tool_calling
  - llm
- **Last Verified**: 2026-09-16 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina
    - JaZeeGH
    - Sridhar1030
  - Reviewers:
    - ChughShilpa
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
