# Lora ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train model using LoRA (Low-Rank Adaptation). Outputs model artifact and metrics.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pvc_path` | `str` | `None` | Workspace PVC root path (use dsl.WORKSPACE_PATH_PLACEHOLDER). |
| `output_model` | `dsl.Output[dsl.Model]` | `None` | Output model artifact. |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | Output training metrics artifact. |
| `output_loss_chart` | `dsl.Output[dsl.HTML]` | `None` | Output HTML artifact with training loss chart. |
| `dataset` | `dsl.Input[dsl.Dataset]` | `None` | Input training dataset artifact. |
| `training_base_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or local path). |
| `training_effective_batch_size` | `int` | `128` | Effective batch size per optimizer step. |
| `training_max_tokens_per_gpu` | `int` | `32000` | Max tokens per GPU (memory cap). |
| `training_max_seq_len` | `int` | `8192` | Max sequence length in tokens. |
| `training_learning_rate` | `Optional[float]` | `None` | Learning rate (default: 2e-4). |
| `training_lr_warmup_steps` | `Optional[int]` | `None` | Learning rate warmup steps. |
| `training_checkpoint_at_epoch` | `Optional[bool]` | `None` | Save checkpoint at each epoch. |
| `training_num_epochs` | `Optional[int]` | `None` | Number of training epochs (default: 3). |
| `training_data_output_dir` | `Optional[str]` | `None` | Directory for processed training data. |
| `training_envs` | `str` | `""` | Environment overrides as KEY=VAL,KEY=VAL. |
| `training_resource_cpu_per_worker` | `str` | `4` | CPU cores per worker. |
| `training_resource_gpu_per_worker` | `int` | `1` | GPUs per worker. |
| `training_resource_memory_per_worker` | `str` | `32Gi` | Memory per worker (e.g., 32Gi). |
| `training_resource_num_procs_per_worker` | `str` | `auto` | Processes per worker (auto or int). |
| `training_resource_num_workers` | `int` | `1` | Number of training pods. |
| `training_metadata_labels` | `str` | `""` | Pod labels as key=value,key=value. |
| `training_metadata_annotations` | `str` | `""` | Pod annotations as key=value,key=value. |
| `training_lora_r` | `int` | `16` | LoRA rank (controls model capacity). |
| `training_lora_alpha` | `int` | `32` | LoRA scaling factor. |
| `training_lora_dropout` | `float` | `0.0` | Dropout rate for LoRA layers. |
| `training_lora_target_modules` | `str` | `""` | Comma-separated list of modules to apply LoRA (empty = auto-detect). |
| `training_lora_use_rslora` | `Optional[bool]` | `None` | Use Rank-Stabilized LoRA variant. |
| `training_lora_use_dora` | `Optional[bool]` | `None` | Use Weight-Decomposed LoRA (DoRA). |
| `training_lora_load_in_4bit` | `Optional[bool]` | `None` | Enable 4-bit quantization (QLoRA). |
| `training_lora_load_in_8bit` | `Optional[bool]` | `None` | Enable 8-bit quantization. |
| `training_lora_bnb_4bit_quant_type` | `Optional[str]` | `None` | Quantization type (e.g., "nf4"). Reserved for future training_hub support. |
| `training_lora_bnb_4bit_compute_dtype` | `Optional[str]` | `None` | Compute dtype (e.g., "bfloat16"). Reserved for future support. |
| `training_lora_bnb_4bit_use_double_quant` | `Optional[bool]` | `None` | Enable double quantization. Reserved for future training_hub support. |
| `training_lora_sample_packing` | `Optional[bool]` | `None` | Pack multiple samples for efficiency (default: True in training_hub). |
| `training_seed` | `Optional[int]` | `None` | Random seed for reproducibility. |
| `training_use_liger` | `Optional[bool]` | `None` | Enable Liger kernel optimizations. |
| `training_lr_scheduler` | `Optional[str]` | `None` | LR scheduler type (cosine, linear, etc.). Training_hub default: linear. |
| `training_micro_batch_size` | `Optional[int]` | `None` | Micro batch size per GPU. |
| `training_gradient_accumulation_steps` | `Optional[int]` | `None` | Gradient accumulation steps. |
| `training_flash_attention` | `Optional[bool]` | `None` | Enable flash attention. |
| `training_bf16` | `Optional[bool]` | `None` | Use bfloat16 precision. |
| `training_fp16` | `Optional[bool]` | `None` | Use float16 precision. |
| `training_tf32` | `Optional[bool]` | `None` | Enable TF32 on Ampere+ GPUs. |
| `training_save_steps` | `Optional[int]` | `None` | Save checkpoint every N steps. |
| `training_eval_steps` | `Optional[int]` | `None` | Run evaluation every N steps. |
| `training_logging_steps` | `Optional[int]` | `None` | Log metrics every N steps. |
| `training_save_total_limit` | `Optional[int]` | `None` | Max checkpoints to keep. |
| `training_wandb_project` | `Optional[str]` | `None` | Weights & Biases project name. |
| `training_wandb_entity` | `Optional[str]` | `None` | Weights & Biases entity/team. |
| `training_wandb_run_name` | `Optional[str]` | `None` | Weights & Biases run name. |
| `training_tensorboard_log_dir` | `Optional[str]` | `None` | TensorBoard log directory. |
| `training_mlflow_tracking_uri` | `Optional[str]` | `None` | MLflow tracking server URI. |
| `training_mlflow_experiment_name` | `Optional[str]` | `None` | MLflow experiment name. |
| `training_mlflow_run_name` | `Optional[str]` | `None` | MLflow run name. |
| `training_dataset_type` | `Optional[str]` | `None` | Dataset format type. |
| `training_field_messages` | `Optional[str]` | `None` | Field name for messages in dataset. |
| `training_field_instruction` | `Optional[str]` | `None` | Field name for instruction in dataset. |
| `training_field_input` | `Optional[str]` | `None` | Field name for input in dataset. |
| `training_field_output` | `Optional[str]` | `None` | Field name for output in dataset. |
| `training_enable_model_splitting` | `Optional[bool]` | `None` | Enable model splitting across GPUs. |
| `training_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime to use. |
| `kubernetes_config` | `dsl.TaskConfig` | `None` | KFP TaskConfig for volumes/env/resources passthrough. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` |  |

## Metadata 🗂️

- **Name**: lora
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Unsloth, Version: >=2024.11.0
- **Tags**:
  - training
  - fine_tuning
  - lora
  - peft
  - llm
  - parameter_efficient
- **Last Verified**: 2026-02-23 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
