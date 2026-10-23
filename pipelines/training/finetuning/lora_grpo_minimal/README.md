# Lora Grpo Minimal Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Run the complete quick-start LoRA GRPO workflow.

The pipeline prepares the dataset, submits a single-GPU ART TrainJob, evaluates reward improvement, and registers and deploys the merged model only when the promotion gate passes.

Private S3 dataset inputs use the optional RHOAI data connection Secret named ``s3-secret``. This is the default Secret name; do not change it. The Secret should provide ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``, and optionally ``AWS_S3_ENDPOINT`` and ``AWS_DEFAULT_REGION``. Hugging Face,
HTTP(S), and local/PVC inputs do not require this Secret.

The following implementation details are intentionally fixed for the minimal variant: the Training Hub runtime, one training GPU, 4 CPU cores, 64Gi worker memory, empty TrainJob labels/annotations, the PyTorch model format, and the standard registry/deployment metadata defaults.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_00_infra_man_pvc_name` | `str` | `None` | Name of the ReadWriteMany PVC used for training checkpoints and model serving. To create one: in RHOAI open your Data Science Project -> Cluster Storage -> Create storage, select ReadWriteMany access mode, minimum 50Gi. Enter only the PVC name (not a path). Example: ``grpo-pipeline-pvc`` |
| `phase_04_registry_man_address` | `str` | `None` | Hostname or IP address of the Model Registry service (required). Pass only the host or IP, without ``http://``, ``https://``, or the ``/api/model_registry/v1alpha3`` path. |
| `phase_05_deploy_man_namespace` | `str` | `None` | OpenShift project where the KServe InferenceService will be created. Must be the same namespace where your PVC is provisioned. Should have KServe enabled - typically your Data Science Project. Example: ``my-ds-project`` |
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset reference. Hugging Face references are loaded directly by Training Hub; local JSON/JSONL, PVC, S3, and HTTP(S) sources are normalized by the preparation component onto the shared PVC. For Hugging Face datasets, use ``hf://org/dataset:config``; the ``hf://`` prefix is optional for repository IDs. Non-Hugging Face rows are converted to Training Hub's canonical ``messages`` and ``tools`` format. See the [Training Hub loader format](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/lora_grpo.py#L381-L387). |
| `phase_02_train_man_model` | `str` | `None` | HuggingFace model ID of the base model to fine-tune. If the model is gated, configure the ``hf-token`` secret. Examples: ``Qwen/Qwen3-4B`` or ``ibm-granite/granite-3.3-8b-instruct`` |
| `phase_05_deploy_opt_tokenizer_mode` | `str` | `None` | Required vLLM tokenizer mode. Use ``auto`` when the fast tokenizer is compatible, ``slow`` when the model requires its HuggingFace slow tokenizer (for example, Qwen3 in the tested environment), or ``mistral`` for models that require the ``mistral_common`` tokenizer. Do not leave it empty. |
| `phase_01_dataset_opt_max_invalid_row_ratio` | `float` | `0.1` | Maximum fraction of invalid rows that the preparation component may skip for local, PVC, S3, and HTTP(S) sources before failing. Defaults to 10%; use ``0.0`` for strict validation. It is not applied to direct Hugging Face loading. |
| `phase_02_train_man_num_iterations` | `int` | `5` | Number of GRPO policy-gradient update steps. Each iteration generates ``group_size`` rollouts per prompt and updates the LoRA adapter. The promotion gate (stage 3) requires ``final_reward > initial_reward``. Increase the completion timeout when using higher iteration counts, larger models, or slower hardware. |
| `phase_02_train_man_group_size` | `int` | `4` | Rollouts generated per prompt to estimate relative reward advantage. Higher = better gradient signal but more memory and time. Default 4 is suitable for single-GPU runs. |
| `phase_02_train_man_prompt_batch_size` | `int` | `50` | Prompts processed per GRPO update step. Total rollout batch = ``prompt_batch_size x group_size`` (e.g., 50 x 4 = 200). Reduce if you hit out-of-memory errors. Example: ``50`` |
| `phase_02_train_man_lora_r` | `int` | `16` | LoRA adapter rank. Higher = more adapter capacity but slower training and more memory. Typical values: 8 (lightweight), 16 (balanced), 32 (high capacity). Example: ``16`` |
| `phase_02_train_man_lora_alpha` | `int` | `16` | LoRA scaling factor. Effective weight scale = ``lora_alpha / lora_r``. Set equal to ``lora_r`` for neutral scaling (recommended). Example: ``16`` |
| `phase_04_registry_man_name` | `str` | `grpo-model` | Name to register the model under. Must be unique within your Model Registry. Defaults to ``grpo-model``; override it with a unique name such as ``qwen3-grpo-tool-call`` for repeated or production runs. |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Version tag for this model revision. Use semantic versioning. Example: ``1.0.0`` |
| `phase_02_train_opt_n_train` | `int` | `200` | Maximum rows ART draws from the dataset per run. This is independent of the total dataset size. Example: ``200`` for quick tests, ``2000`` for a fuller run. |
| `phase_02_train_opt_completion_timeout_seconds` | `int` | `7200` | Maximum number of seconds to wait for the TrainJob to reach Complete or Failed after it starts. The wait returns immediately when the TrainJob completes. Set this to a larger value for longer runs or slower hardware, or a smaller value for smoke tests. The TrainJob startup wait remains unchanged. |
| `phase_02_train_opt_learning_rate` | `float` | `1e-05` | AdamW learning rate for LoRA adapter updates. ``1e-5`` is a safe default; increase to ``5e-5`` if reward improvement is too slow. |
| `phase_02_train_opt_gpu_memory_utilization` | `float` | `0.45` | Fraction of GPU VRAM given to the vLLM rollout engine (remainder goes to the policy model and optimizer). Default 0.45 is tuned for Qwen3-4B on a 80GB A100. Reduce to 0.35 if you see OOM during rollout generation. |
| `phase_02_train_opt_enforce_eager` | `bool` | `True` | Disables torch.compile and CUDA graph capture in vLLM. Must be True for Qwen3 models (known CUDAGraph incompatibility). Set False for other models for faster rollouts. |
| `phase_02_train_opt_env_vars` | `str` | `""` | Comma-separated ``KEY=VALUE`` environment variables injected into the training pod. Leave empty if not needed. Example: ``NCCL_DEBUG=INFO,TORCH_DISTRIBUTED_DEBUG=DETAIL`` |
| `phase_04_registry_opt_port` | `int` | `8080` | Port of the Model Registry service. Default 8080 matches a typical internal HTTP registry. |
| `phase_05_deploy_opt_gpu_count` | `int` | `1` | GPUs allocated to the KServe vLLM predictor pod. 1 GPU is sufficient for Qwen3-4B. Example: ``1`` |
| `phase_05_deploy_opt_max_model_len` | `int` | `4096` | Maximum total token length (prompt + completion) vLLM will handle. Longer contexts use more VRAM. Default 4096 is adequate for tool-call traces. Increase to 8192 for longer prompts. |
| `phase_05_deploy_opt_max_replicas` | `int` | `1` | Maximum predictor replicas for autoscaling. Set equal to ``min_replicas`` to disable scaling. Example: ``1`` |
| `phase_05_deploy_opt_cpu_requests` | `str` | `2` | CPU cores requested and limited for the KServe predictor pod. Example: ``2`` |
| `phase_05_deploy_opt_memory_requests` | `str` | `8Gi` | RAM requested and limited for the KServe predictor pod. Example: ``8Gi`` |

## Metadata 🗂️

- **Name**: lora_grpo_minimal_pipeline
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
    - Name: Model Registry, Version: >=0.3.4
    - Name: KServe, Version: >=0.11.0
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
  - pipeline
  - minimal
- **Last Verified**: 2026-10-23 00:00:00+00:00
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
