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
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` | Input training dataset artifact (tool-call traces). |
| `training_base_model` | `str` | `Qwen/Qwen3-4B` | Base model (HuggingFace ID or local path). |
| `training_data_path` | `str` | `""` | Dataset path or HuggingFace ID (e.g., Agent-Ark/Toucan-1.5M). When empty, uses dataset artifact. When set, overrides the artifact. |
| `training_num_iterations` | `int` | `5` | Number of GRPO training iterations. |
| `training_group_size` | `int` | `4` | Number of rollouts per prompt for GRPO advantage estimation. |
| `training_prompt_batch_size` | `int` | `50` | Number of prompts per training batch. |
| `training_n_train` | `int` | `200` | Number of training samples to use from the dataset. |
| `training_learning_rate` | `Optional[float]` | `None` | Learning rate (default: 1e-5). |
| `training_gpu_memory_utilization` | `float` | `0.45` | Fraction of GPU memory for vLLM (rest for training). |
| `training_enforce_eager` | `bool` | `True` | Disable torch.compile/CUDAGraphs in vLLM (Qwen3 workaround). |
| `training_data_config` | `str` | `Qwen3` | Dataset config name for HuggingFace datasets (e.g., Qwen3). |
| `training_lora_r` | `int` | `16` | LoRA rank (controls adapter capacity). |
| `training_lora_alpha` | `int` | `8` | LoRA scaling factor. |
| `training_envs` | `str` | `""` | Environment overrides as KEY=VAL,KEY=VAL. |
| `training_resource_cpu_per_worker` | `str` | `4` | CPU cores per worker. |
| `training_resource_gpu_per_worker` | `int` | `1` | GPUs per worker (should be 1 for ART). |
| `training_resource_memory_per_worker` | `str` | `64Gi` | Memory per worker (e.g., 64Gi). |
| `training_metadata_labels` | `str` | `""` | Pod labels as key=value,key=value. |
| `training_metadata_annotations` | `str` | `""` | Pod annotations as key=value,key=value. |
| `training_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime to use. |
| `training_pvc_name` | `str` | `""` | PVC name to mount on the TrainJob pod for shared storage. |
| `kubernetes_config` | `dsl.TaskConfig` | `None` | KFP TaskConfig for volumes/env/resources passthrough. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` |  |

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
