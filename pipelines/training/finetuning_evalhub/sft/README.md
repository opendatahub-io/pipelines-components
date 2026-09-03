# Sft Evalhub Pipeline ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

SFT Training Pipeline with Eval Hub evaluation (KServe).

A 5-stage ML pipeline for fine-tuning language models:

1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP 2) OCI Model Resolution - Resolves OCI URIs to PVC paths or passes through HF URIs 3) SFT Training - Fine-tunes using instructlab-training backend 4) Evaluation - Evaluates via Eval Hub with a KServe InferenceService for
model serving. Results optionally tracked in MLflow. 5) Model Registry - Registers trained model to Kubeflow Model Registry

Prerequisites: Eval Hub and KServe must be installed on the cluster. The pipeline ServiceAccount needs RBAC permissions for inferenceservices.serving.kserve.io and servingruntimes.serving.kserve.io resources (create, delete, get, list, patch). The workspace PVC must use ReadWriteMany access mode
(NFS-backed) so the KServe predictor pod can mount the model. The eval component uses the in-cluster ServiceAccount token for K8s API access.

Known limitations: Some HuggingFace datasets used by benchmarks require trust_remote_code=True. The 5 default leaderboard benchmarks (ifeval, bbh, mmlu_pro, musr, math_hard) work without it. For other benchmarks, a custom provider ConfigMap with HF_DATASETS_TRUST_REMOTE_CODE=1 must be configured.
The base_model_name parameter is needed for tokenizer resolution since the served model is a local checkpoint.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset location (hf://, s3://, https://). |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size per optimizer step. |
| `phase_02_train_man_epochs` | `int` | `1` | Number of training epochs. |
| `phase_02_train_man_gpu` | `int` | `1` | GPUs per worker. |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path). |
| `phase_02_train_man_tokens` | `int` | `10000` | Max tokens per GPU (memory cap). |
| `phase_02_train_man_workers` | `int` | `4` | Number of training pods. |
| `phase_03_eval_opt_evalhub_url` | `str` | `""` | Eval Hub API endpoint URL (empty = skip evaluation). |
| `phase_03_eval_opt_collection` | `str` | `""` | Eval Hub collection ID (overrides benchmarks list). Available: "leaderboard-v2", "safety-and-fairness-v1", "toxicity-and-ethical-principles". |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip). |
| `phase_04_registry_man_author` | `str` | `pipeline` | Author name for the registered model. |
| `phase_04_registry_man_name` | `str` | `sft-model` | Model name in registry. |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version. |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all). |
| `phase_02_train_opt_annotations` | `str` | `""` | Pod annotations (key=value,...). |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker. |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000` | Environment variables (KEY=VAL,...). |
| `phase_02_train_opt_labels` | `str` | `""` | Pod labels (key=value,...). |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate for training. |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Learning rate warmup steps. |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | LR scheduler type (cosine, linear). |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Maximum sequence length in tokens. |
| `phase_02_train_opt_memory` | `str` | `64Gi` | Memory per worker (e.g., 64Gi). |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker (auto or int). |
| `phase_02_train_opt_save_epoch` | `bool` | `True` | Save checkpoint at each epoch. |
| `phase_02_train_opt_save_full_state` | `bool` | `False` | Save full accelerate state. |
| `phase_02_train_opt_fsdp_sharding` | `str` | `FULL_SHARD` | FSDP strategy (FULL_SHARD, HYBRID_SHARD). |
| `phase_02_train_opt_save_samples` | `int` | `0` | Number of samples to save (0 = none). |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility. |
| `phase_02_train_opt_use_liger` | `bool` | `False` | Enable Liger kernel optimizations. |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | ClusterTrainingRuntime name. |
| `phase_03_eval_opt_benchmarks` | `list` | `[{'id': 'leaderboard_ifeval', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_bbh', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_mmlu_pro', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_musr', 'provider_id': 'lm_evaluation_harness'}, {'id': 'leaderboard_math_hard', 'provider_id': 'lm_evaluation_harness'}]` | Benchmarks to evaluate. Defaults to 5 leaderboard benchmarks (ifeval, bbh, mmlu_pro, musr, math_hard) that work without HF token or custom providers. |
| `phase_03_eval_opt_mlflow_experiment` | `str` | `""` | MLflow experiment name (non-empty = enable, empty = disabled). |
| `phase_03_eval_opt_timeout` | `int` | `7200` | Max seconds to wait for evaluation. |
| `phase_03_eval_opt_kserve_gpu_count` | `int` | `1` | GPUs for the KServe predictor. |
| `phase_03_eval_opt_kserve_cpu` | `str` | `2` | CPU for the KServe predictor. |
| `phase_03_eval_opt_kserve_memory` | `str` | `32Gi` | Pod memory for the KServe predictor. |
| `phase_04_registry_opt_description` | `str` | `""` | Model description for registry. |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx). |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model Registry server port. |

## Metadata 🗂️

- **Name**: sft_evalhub_pipeline
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
  - sft
  - supervised_fine_tuning
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
