# Lora Evalhub Pipeline ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

LoRA Training Pipeline with Eval Hub evaluation (KServe).

A 5-stage ML pipeline for fine-tuning language models with LoRA:

1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP 2) OCI Model Resolution - Resolves OCI URIs to PVC paths or passes through HF URIs 3) LoRA Training - Fine-tunes using unsloth backend (low-rank adapters) 4) Evaluation - Evaluates via Eval Hub with a KServe InferenceService
for model serving. Results optionally tracked in MLflow. 5) Model Registry - Registers trained model to Kubeflow Model Registry

Prerequisites: Eval Hub and KServe must be installed on the cluster. The pipeline ServiceAccount needs RBAC permissions for inferenceservices.serving.kserve.io and servingruntimes.serving.kserve.io resources (create, delete, get, list, patch). The workspace PVC must use ReadWriteMany access mode
(NFS-backed) so the KServe predictor pod can mount the model. The eval component uses the in-cluster ServiceAccount token for K8s API access.

Known limitations: Some HuggingFace datasets used by benchmarks require trust_remote_code=True. The 5 default leaderboard benchmarks (ifeval, bbh, mmlu_pro, musr, math_hard) work without it. For other benchmarks, a custom provider ConfigMap with HF_DATASETS_TRUST_REMOTE_CODE=1 must be configured.
The base_model_name parameter is needed for tokenizer resolution since the served model is a local checkpoint.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset location (hf://, s3://, https://). |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step). |
| `phase_02_train_man_train_epochs` | `int` | `2` | Number of training epochs. |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker. |
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path). |
| `phase_02_train_man_train_tokens` | `int` | `32000` | Max tokens per GPU (memory cap). |
| `phase_02_train_man_lora_r` | `int` | `16` | Rank of the low-rank matrices (4, 8, 16, 32, 64). |
| `phase_02_train_man_lora_alpha` | `int` | `32` | Scaling factor (typically 2x lora_r). |
| `phase_03_eval_opt_evalhub_url` | `str` | `""` | Eval Hub API endpoint URL (empty = skip evaluation). |
| `phase_03_eval_opt_collection` | `str` | `""` | Eval Hub collection ID (overrides benchmarks list). Available: "leaderboard-v2", "safety-and-fairness-v1", "toxicity-and-ethical-principles". |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip). |
| `phase_04_registry_man_reg_author` | `str` | `pipeline` | Author name for the registered model. |
| `phase_04_registry_man_reg_name` | `str` | `lora-model` | Model name in registry. |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version. |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all). |
| `phase_02_train_opt_annotations` | `str` | `""` | K8s annotations (key=val,...). |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker. |
| `phase_02_train_opt_env_vars` | `str` | `""` | Env vars (KEY=VAL,...). |
| `phase_02_train_opt_labels` | `str` | `""` | K8s labels (key=val,...). |
| `phase_02_train_opt_learning_rate` | `float` | `0.0002` | Learning rate. 2e-4 for LoRA. |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | LR schedule (cosine, linear). |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Warmup steps before full LR. |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens. |
| `phase_02_train_opt_memory` | `str` | `32Gi` | RAM per worker. |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker (auto = per GPU). |
| `phase_02_train_opt_save_epoch` | `bool` | `False` | Save checkpoint at each epoch. |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility. |
| `phase_02_train_opt_use_liger` | `bool` | `True` | Enable Liger kernel optimizations. |
| `phase_02_train_opt_lora_dropout` | `float` | `0.0` | Dropout rate for LoRA layers. |
| `phase_02_train_opt_lora_target_modules` | `str` | `""` | Modules to apply LoRA (empty=auto-detect). |
| `phase_02_train_opt_lora_use_rslora` | `bool` | `False` | Use Rank-Stabilized LoRA. |
| `phase_02_train_opt_lora_use_dora` | `bool` | `False` | Use Weight-Decomposed LoRA (DoRA). |
| `phase_02_train_opt_lora_load_in_4bit` | `bool` | `True` | Enable 4-bit quantization. |
| `phase_02_train_opt_lora_load_in_8bit` | `bool` | `False` | Enable 8-bit quantization. |
| `phase_02_train_opt_lora_sample_packing` | `bool` | `False` | Pack multiple samples. |
| `phase_02_train_opt_micro_batch_size` | `int` | `2` | Micro batch size per GPU. |
| `phase_02_train_opt_grad_accum_steps` | `int` | `1` | Gradient accumulation steps. |
| `phase_02_train_opt_flash_attention` | `bool` | `True` | Enable flash attention. |
| `phase_02_train_opt_bf16` | `bool` | `True` | Use bfloat16 precision. |
| `phase_02_train_opt_fp16` | `bool` | `False` | Use float16 precision. |
| `phase_02_train_opt_tf32` | `bool` | `True` | Enable TF32 on Ampere+ GPUs. |
| `phase_02_train_opt_save_steps` | `int` | `500` | Save checkpoint every N steps. |
| `phase_02_train_opt_eval_steps` | `int` | `500` | Run evaluation every N steps. |
| `phase_02_train_opt_logging_steps` | `int` | `10` | Log metrics every N steps. |
| `phase_02_train_opt_save_total_limit` | `int` | `3` | Max checkpoints to keep. |
| `phase_02_train_opt_wandb_project` | `str` | `""` | Weights & Biases project name. |
| `phase_02_train_opt_wandb_entity` | `str` | `""` | Weights & Biases entity/team. |
| `phase_02_train_opt_wandb_run_name` | `str` | `""` | Weights & Biases run name. |
| `phase_02_train_opt_tensorboard_log_dir` | `str` | `""` | TensorBoard log directory. |
| `phase_02_train_opt_dataset_type` | `str` | `""` | Dataset format type. |
| `phase_02_train_opt_field_messages` | `str` | `""` | Field name for messages in dataset. |
| `phase_02_train_opt_field_instruction` | `str` | `""` | Field name for instruction. |
| `phase_02_train_opt_field_input` | `str` | `""` | Field name for input in dataset. |
| `phase_02_train_opt_field_output` | `str` | `""` | Field name for output in dataset. |
| `phase_02_train_opt_enable_model_splitting` | `bool` | `False` | Enable model splitting across GPUs. |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | ClusterTrainingRuntime name. |
| `phase_03_eval_opt_benchmarks` | `list` | `[{'id': 'leaderboard_ifeval', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_bbh', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_mmlu_pro', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_musr', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_math_hard', 'provider_id': 'lm_evaluation_harness'}]` | Benchmarks to evaluate. Defaults to 5 leaderboard benchmarks (ifeval, bbh, mmlu_pro, musr, math_hard) that work without HF token or custom providers. |
| `phase_03_eval_opt_mlflow_experiment` | `str` | `""` | MLflow experiment name (non-empty = enable, empty = disabled). |
| `phase_03_eval_opt_timeout` | `int` | `7200` | Max seconds to wait for evaluation. |
| `phase_03_eval_opt_kserve_gpu_count` | `int` | `1` | GPUs for the KServe predictor. |
| `phase_03_eval_opt_kserve_cpu` | `str` | `2` | CPU for the KServe predictor. |
| `phase_03_eval_opt_kserve_memory` | `str` | `32Gi` | Pod memory for the KServe predictor. |
| `phase_04_registry_opt_description` | `str` | `""` | Model description. |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx). |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port. |

## Metadata 🗂️

- **Name**: lora_evalhub_pipeline
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Eval Hub, Version: >=0.1.0
    - Name: KServe, Version: >=0.11.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Model Registry, Version: >=0.3.4
- **Tags**:
  - training
  - fine_tuning
  - lora
  - peft
  - parameter_efficient
  - eval_hub
  - kserve
  - pipeline
- **Last Verified**: 2026-05-20 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - efazal
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/opendatahub-io/eval-hub](https://github.com/opendatahub-io/eval-hub)
