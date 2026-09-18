# Lora Grpo Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

LoRA GRPO Training Pipeline - RLVR fine-tuning with promotion gating.

A 5-stage pipeline that trains a language model using LoRA GRPO (Group Relative Policy Optimization) for tool-calling tasks, evaluates training results, and conditionally registers and deploys the model only if training shows improvement.

Prerequisites: - A ReadWriteMany PVC (e.g., NFS-backed) for model storage and serving. Create with: ``oc create pvc <name> --access-mode=ReadWriteMany --storage-class=nfs-csi --size=50Gi`` - ``kubernetes-credentials`` secret with KUBERNETES_SERVER_URL and KUBERNETES_AUTH_TOKEN (required for TrainJob
creation). - ``hf-token`` secret (optional, for gated HuggingFace models/datasets). - Pipeline ServiceAccount RBAC for KServe CRDs (required for stage 5).

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_00_infra_man_pvc_name` | `str` | `None` | Name of the ReadWriteMany PVC used for training checkpoints and model serving. To create one: in RHOAI open your Data Science Project -> Cluster Storage -> Create storage, select ReadWriteMany access mode, minimum 50Gi. Enter only the PVC name (not a path). Example: ``grpo-pipeline-pvc`` |
| `phase_04_registry_man_address` | `str` | `None` | Hostname of the Model Registry service (required). Find it in RHOAI Dashboard -> AI Hub -> Models -> Registry -> click View Details and copy the Server URL. |
| `phase_05_deploy_man_namespace` | `str` | `None` | OpenShift project where the KServe InferenceService will be created. Must be the same namespace where your PVC is provisioned. Should have KServe enabled - typically your Data Science Project. Example: ``my-ds-project`` |
| `phase_01_dataset_man_data_uri` | `str` | `hf://Agent-Ark/Toucan-1.5M:Qwen3` | URI of the dataset. Defaults to ``hf://Agent-Ark/Toucan-1.5M:Qwen3`` (a public tool-calling dataset used as a reference example) - override with your own dataset URI. Supported schemes: - HuggingFace: ``hf://org/dataset`` or ``hf://org/dataset:config``   Some datasets have multiple subsets (configs). If the dataset has   more than one config, you must append ``:config`` - otherwise   ``load_dataset`` will fail asking you to choose. Single-config   datasets do not need it.   Example: ``hf://Agent-Ark/Toucan-1.5M:Qwen3`` - S3: ``s3://bucket/path/to/file.jsonl`` - HTTP: ``https://host/file.jsonl`` (must point to a raw file,   not an HTML page) - PVC file: ``pvc://path/to/file.jsonl`` |
| `phase_01_dataset_man_data_split` | `float` | `1.0` | Fraction of the dataset used for training (remainder becomes eval). Default 1.0 uses all rows, which is correct for GRPO - improvement is measured from rollout rewards, not a held-out eval set. |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen3-4B` | HuggingFace model ID of the base model to fine-tune. If the model is gated, configure the ``hf-token`` secret. Example: ``Qwen/Qwen3-4B`` or ``ibm-granite/granite-3.3-8b-instruct`` |
| `phase_02_train_man_num_iterations` | `int` | `50` | Number of GRPO policy-gradient update steps. Each iteration generates ``group_size`` rollouts per prompt and updates the LoRA adapter. The promotion gate (stage 3) requires ``final_reward > initial_reward`` - fewer than 30 iterations rarely pass it. Default 50 runs in ~1-2 hours on one A100 80GB with Qwen3-4B. Use 100+ for a production-quality training run. |
| `phase_02_train_man_group_size` | `int` | `4` | Rollouts generated per prompt to estimate relative reward advantage. Higher = better gradient signal but more memory and time. Default 4 is suitable for single-GPU runs. |
| `phase_02_train_man_prompt_batch_size` | `int` | `50` | Prompts processed per GRPO update step. Total rollout batch = ``prompt_batch_size x group_size`` (e.g., 50 x 4 = 200). Reduce if you hit out-of-memory errors. Example: ``50`` |
| `phase_02_train_man_lora_r` | `int` | `16` | LoRA adapter rank. Higher = more adapter capacity but slower training and more memory. Typical values: 8 (lightweight), 16 (balanced), 32 (high capacity). Example: ``16`` |
| `phase_02_train_man_lora_alpha` | `int` | `16` | LoRA scaling factor. Effective weight scale = ``lora_alpha / lora_r``. Set equal to ``lora_r`` for neutral scaling (recommended). Example: ``16`` |
| `phase_04_registry_man_name` | `str` | `grpo-model` | Name to register the model under. Must be unique within your Model Registry. Example: ``qwen3-grpo-tool-call`` |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Version tag for this model revision. Use semantic versioning. Example: ``1.0.0`` |
| `phase_01_dataset_opt_subset` | `int` | `0` | Download and use only this many rows (random sample). Set to 0 for all rows. Use a small number for quick pipeline smoke-tests. Example: ``500`` |
| `phase_02_train_opt_data_config` | `str` | `Qwen3` | HuggingFace dataset config (split or subset name). Only needed for multi-config datasets. Defaults to ``Qwen3`` to match the default dataset (``hf://Agent-Ark/Toucan-1.5M:Qwen3``). If you override ``phase_01_dataset_man_data_uri`` with a single-config dataset, set this to an empty string. |
| `phase_02_train_opt_n_train` | `int` | `200` | Maximum rows ART draws from the dataset per run. Independent of ``phase_01_dataset_opt_subset``. Example: ``200`` for quick tests, ``2000`` for a fuller run. |
| `phase_02_train_opt_learning_rate` | `float` | `1e-05` | AdamW learning rate for LoRA adapter updates. ``1e-5`` is a safe default; increase to ``5e-5`` if reward improvement is too slow. |
| `phase_02_train_opt_gpu_memory_utilization` | `float` | `0.45` | Fraction of GPU VRAM given to the vLLM rollout engine (remainder goes to the policy model and optimizer). Default 0.45 is tuned for Qwen3-4B on a 80GB A100. Reduce to 0.35 if you see OOM during rollout generation. |
| `phase_02_train_opt_enforce_eager` | `bool` | `True` | Disables torch.compile and CUDA graph capture in vLLM. Must be True for Qwen3 models (known CUDAGraph incompatibility). Set False for other models for faster rollouts. |
| `phase_02_train_opt_env_vars` | `str` | `""` | Comma-separated ``KEY=VALUE`` environment variables injected into the training pod. Leave empty if not needed. Example: ``NCCL_DEBUG=INFO,TORCH_DISTRIBUTED_DEBUG=DETAIL`` |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores requested for the TrainJob worker pod. Example: ``4`` |
| `phase_02_train_opt_gpu` | `int` | `1` | GPUs per TrainJob worker. ART requires exactly 1 GPU per worker - do not change this. |
| `phase_02_train_opt_memory` | `str` | `64Gi` | RAM requested for the TrainJob worker pod. For Qwen3-4B on an 80GB GPU, 64Gi is sufficient. Example: ``64Gi`` |
| `phase_02_train_opt_labels` | `str` | `""` | Comma-separated ``key=value`` labels applied to the TrainJob worker pod for cost tracking or scheduling constraints. Leave empty if not needed. Example: ``team=ml,project=grpo`` |
| `phase_02_train_opt_annotations` | `str` | `""` | Comma-separated ``key=value`` annotations applied to the TrainJob worker pod. Leave empty if not needed. |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime CR that defines the base container image. Check available runtimes with: ``oc get clustertrainingruntimes``. Default: ``training-hub`` |
| `phase_04_registry_opt_author` | `str` | `pipeline` | Author name attached to the registry entry. Example: ``your-team-name`` |
| `phase_04_registry_opt_description` | `str` | `""` | Free-text description stored with the registered model. Example: ``Qwen3-4B fine-tuned with GRPO for tool-calling tasks`` |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model serialization format in the registry. Use ``pytorch`` for vLLM-served models. Example: ``pytorch`` |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Version of the model format. Use ``1.0`` for PyTorch. Example: ``1.0`` |
| `phase_04_registry_opt_port` | `int` | `8080` | HTTP port of the Model Registry service. Default 8080 matches the standard RHOAI deployment. |
| `phase_05_deploy_opt_gpu_count` | `int` | `1` | GPUs allocated to the KServe vLLM predictor pod. 1 GPU is sufficient for Qwen3-4B. Example: ``1`` |
| `phase_05_deploy_opt_max_model_len` | `int` | `4096` | Maximum total token length (prompt + completion) vLLM will handle. Longer contexts use more VRAM. Default 4096 is adequate for tool-call traces. Increase to 8192 for longer prompts. |
| `phase_05_deploy_opt_min_replicas` | `int` | `1` | Minimum live KServe predictor replicas. Set to 1 to keep the endpoint always warm. Example: ``1`` |
| `phase_05_deploy_opt_max_replicas` | `int` | `1` | Maximum predictor replicas for autoscaling. Set equal to ``min_replicas`` to disable scaling. Example: ``1`` |
| `phase_05_deploy_opt_cpu_requests` | `str` | `2` | CPU cores requested and limited for the KServe predictor pod. Example: ``2`` |
| `phase_05_deploy_opt_memory_requests` | `str` | `8Gi` | RAM requested and limited for the KServe predictor pod. Example: ``8Gi`` |

## Metadata 🗂️

- **Name**: lora_grpo_pipeline
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
- **Last Verified**: 2026-09-18 00:00:00+00:00
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
