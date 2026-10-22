# Lora Grpo Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

LoRA GRPO Training Pipeline - RLVR fine-tuning with promotion gating.

A 4-stage pipeline that prepares and trains a language model using LoRA GRPO (Group Relative Policy Optimization) for tool-calling tasks, evaluates training results, and conditionally registers and deploys the model only if training shows improvement.

Prerequisites: - A ReadWriteMany PVC (e.g., NFS-backed) for model storage and serving. Create with: ``oc create pvc <name> --access-mode=ReadWriteMany --storage-class=nfs-csi --size=50Gi`` - ``kubernetes-credentials`` secret with KUBERNETES_SERVER_URL and KUBERNETES_AUTH_TOKEN (required for TrainJob
creation). - ``hf-token`` secret (optional, for gated HuggingFace models/datasets). - RHOAI data connection Secret (only required for private S3 datasets; defaults to ``s3-secret`` and is injected optionally). - Model Registry RoleBinding for the pipeline ServiceAccount (required for stage 4). -
Pipeline ServiceAccount RBAC for KServe CRDs (required for stage 5).

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_00_infra_man_pvc_name` | `str` | `None` | Name of the ReadWriteMany PVC used for training checkpoints and model serving. To create one: in RHOAI open your Data Science Project -> Cluster Storage -> Create storage, select ReadWriteMany access mode, minimum 50Gi. Enter only the PVC name (not a path). Example: ``grpo-pipeline-pvc`` |
| `phase_04_registry_man_address` | `str` | `None` | Hostname or IP address of the Model Registry service (required). Pass only the host or IP, without ``http://``, ``https://``, or the ``/api/model_registry/v1alpha3`` path. |
| `phase_05_deploy_man_namespace` | `str` | `None` | OpenShift project where the KServe InferenceService will be created. Must be the same namespace where your PVC is provisioned. Should have KServe enabled - typically your Data Science Project. Example: ``my-ds-project`` |
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset reference. Hugging Face references are loaded directly by Training Hub; local JSON/JSONL, PVC, S3, and HTTP(S) sources are normalized by the preparation component onto the shared PVC. For Hugging Face datasets, use ``hf://org/dataset`` or ``hf://org/dataset:config``; the ``hf://`` prefix is optional for repository IDs. Non-Hugging Face rows are converted to Training Hub's canonical ``messages`` and ``tools`` format. See the [Training Hub loader format](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/lora_grpo.py#L381-L387). |
| `phase_02_train_man_model` | `str` | `None` | HuggingFace model ID of the base model to fine-tune. If the model is gated, configure the ``hf-token`` secret. Examples: ``Qwen/Qwen3-4B`` or ``ibm-granite/granite-3.3-8b-instruct`` |
| `phase_05_deploy_opt_tokenizer_mode` | `str` | `None` |  |
| `phase_01_dataset_opt_max_invalid_row_ratio` | `float` | `0.1` | Maximum fraction of invalid rows that the preparation component may skip for local, PVC, S3, and HTTP(S) sources before failing. Defaults to 10%; use ``0.0`` for strict validation. It is not applied to direct Hugging Face loading. |
| `phase_01_dataset_opt_data_connection_name` | `str` | `s3-secret` | Name of the RHOAI data connection Secret used by the preparation task for private S3 input. It defaults to ``s3-secret`` and is optional for Hugging Face, HTTP(S), and local/PVC input. The Secret should provide ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and optionally ``AWS_S3_ENDPOINT`` and ``AWS_DEFAULT_REGION``. |
| `phase_02_train_opt_data_config` | `str` | `""` | Optional Hugging Face dataset configuration name. This is used only for Hugging Face sources. Set it when a multi-configuration dataset does not include the configuration in its URI, for example ``Qwen3`` with ``hf://Agent-Ark/Toucan-1.5M``. Leave it empty when the URI already includes ``:config``, when the Hugging Face dataset has one configuration, or for local/PVC, S3, and HTTP(S) sources. Split selection is not supported by this pipeline; do not use this parameter to specify a split name. |
| `phase_02_train_man_num_iterations` | `int` | `50` | Number of GRPO policy-gradient update steps. Each iteration generates ``group_size`` rollouts per prompt and updates the LoRA adapter. The promotion gate (stage 3) requires ``final_reward > initial_reward`` - fewer than 30 iterations rarely pass it. Default 50 runs in ~1-2 hours on one A100 80GB with Qwen3-4B. Use 100+ for a production-quality training run. |
| `phase_02_train_man_group_size` | `int` | `4` | Rollouts generated per prompt to estimate relative reward advantage. Higher = better gradient signal but more memory and time. Default 4 is suitable for single-GPU runs. |
| `phase_02_train_man_prompt_batch_size` | `int` | `50` | Prompts processed per GRPO update step. Total rollout batch = ``prompt_batch_size x group_size`` (e.g., 50 x 4 = 200). Reduce if you hit out-of-memory errors. Example: ``50`` |
| `phase_02_train_man_lora_r` | `int` | `16` | LoRA adapter rank. Higher = more adapter capacity but slower training and more memory. Typical values: 8 (lightweight), 16 (balanced), 32 (high capacity). Example: ``16`` |
| `phase_02_train_man_lora_alpha` | `int` | `16` | LoRA scaling factor. Effective weight scale = ``lora_alpha / lora_r``. Set equal to ``lora_r`` for neutral scaling (recommended). Example: ``16`` |
| `phase_04_registry_man_name` | `str` | `grpo-model` | Name to register the model under. Must be unique within your Model Registry. Defaults to ``grpo-model``; override it with a unique name such as ``qwen3-grpo-tool-call`` for repeated or production runs. |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Version tag for this model revision. Use semantic versioning. Example: ``1.0.0`` |
| `phase_02_train_opt_n_train` | `int` | `200` | Maximum rows ART draws from the dataset per run. This is independent of the total dataset size. Example: ``200`` for quick tests, ``2000`` for a fuller run. |
| `phase_02_train_opt_completion_timeout_seconds` | `int` | `7200` | Maximum number of seconds to wait for the TrainJob to reach Complete or Failed after it starts. Defaults to ``7200`` seconds (2 hours). The wait returns immediately when the TrainJob completes; set a larger value for longer runs or a smaller value for smoke tests. |
| `phase_02_train_opt_learning_rate` | `float` | `1e-05` |  |
| `phase_02_train_opt_gpu_memory_utilization` | `float` | `0.45` |  |
| `phase_02_train_opt_enforce_eager` | `bool` | `True` |  |
| `phase_02_train_opt_env_vars` | `str` | `""` |  |
| `phase_02_train_opt_cpu` | `str` | `4` |  |
| `phase_02_train_opt_gpu` | `int` | `1` |  |
| `phase_02_train_opt_memory` | `str` | `64Gi` |  |
| `phase_02_train_opt_labels` | `str` | `""` |  |
| `phase_02_train_opt_annotations` | `str` | `""` |  |
| `phase_02_train_opt_runtime` | `str` | `training-hub` |  |
| `phase_04_registry_opt_author` | `str` | `pipeline` |  |
| `phase_04_registry_opt_description` | `str` | `""` |  |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` |  |
| `phase_04_registry_opt_format_version` | `str` | `1.0` |  |
| `phase_04_registry_opt_port` | `int` | `8080` |  |
| `phase_05_deploy_opt_gpu_count` | `int` | `1` |  |
| `phase_05_deploy_opt_max_model_len` | `int` | `4096` |  |
| `phase_05_deploy_opt_min_replicas` | `int` | `1` |  |
| `phase_05_deploy_opt_max_replicas` | `int` | `1` |  |
| `phase_05_deploy_opt_cpu_requests` | `str` | `2` |  |
| `phase_05_deploy_opt_memory_requests` | `str` | `8Gi` |  |

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

<!-- custom-content -->

## Storage behavior

The preparation task mounts the shared PVC at ``/mnt/pipeline``. Hugging Face
references are passed directly to Training Hub, while local/PVC, S3, and
HTTP(S) JSON/JSONL sources are normalized to the PVC before training. Training
Hub's Hugging Face, vLLM, Triton, TorchInductor, and temporary-file caches are
redirected to the mounted PVC under ``/mnt/pipeline/.cache`` and
``/mnt/pipeline/.tmp``. The pipeline does not copy the dataset through a
node-local KFP artifact directory.

For a Hugging Face dataset with multiple configs, include the config in the
URI, for example ``hf://Agent-Ark/Toucan-1.5M:Qwen3``, or set
``phase_02_train_opt_data_config`` separately. Leave this parameter empty for
single-config Hugging Face datasets and for local/PVC, S3, and HTTP(S) inputs;
it is used only for Hugging Face dataset configuration selection. For private
S3 input, set
``phase_01_dataset_opt_data_connection_name`` to the RHOAI Connection Secret containing the AWS
credentials. ``phase_01_dataset_opt_max_invalid_row_ratio`` defaults to 10%
for non-Hugging Face sources and can be set to ``0.0`` for strict validation.
``phase_02_train_opt_n_train`` controls how many examples ART draws for
training.

## Training wait timeout

The pipeline waits up to two hours (7200 seconds) for the TrainJob to reach
``Complete`` or ``Failed``. The wait returns immediately when the TrainJob
finishes; it does not force every run to last two hours. Set
``phase_02_train_opt_completion_timeout_seconds`` to a smaller value for
short smoke tests or a larger value for unusually long runs, for example
``10800`` for three hours. The TrainJob startup wait remains unchanged.

The timeout is measured in seconds.

## Model Registry setup

The pipeline registers the trained model in the selected Model Registry and stores the persistent model location under a run-scoped URI such as ``pvc://<pvc-name>/final_model/<pipeline-run-id>``. This prevents concurrent runs sharing a PVC from overwriting one another. The registry does not need to run in the pipeline namespace, but the pipeline ServiceAccount must be authorized to access the registry Service.

For the initial pipeline test, select an available registry. The registry namespace is installation-specific; verify it in your cluster:

```bash
oc get modelregistry -A
```

The Model Registry operator creates a Role named ``registry-user-<registry-name>``. Bind the selected registry Role to the ServiceAccount used by the pipeline. Replace the namespace, ServiceAccount, and registry name if your cluster uses different values:

```bash
REGISTRY_NAMESPACE=<model-registry-namespace>
PIPELINE_NAMESPACE=<pipeline-namespace>
PIPELINE_SERVICE_ACCOUNT=<pipeline-service-account>
REGISTRY_NAME=<registry-name>

oc get svc -n "$REGISTRY_NAMESPACE"

oc get role "registry-user-${REGISTRY_NAME}" -n "$REGISTRY_NAMESPACE"
oc create rolebinding lora-grpo-${REGISTRY_NAME} \
  --role="registry-user-${REGISTRY_NAME}" \
  --serviceaccount="${PIPELINE_NAMESPACE}:${PIPELINE_SERVICE_ACCOUNT}" \
  -n "$REGISTRY_NAMESPACE"
```

Verify access to the named registry Service. The Role is usually restricted to that Service, so check the resource by name:

```bash
oc auth can-i get service/${REGISTRY_NAME} \
  --as="system:serviceaccount:${PIPELINE_NAMESPACE}:${PIPELINE_SERVICE_ACCOUNT}" \
  -n "$REGISTRY_NAMESPACE"
```

The result must be ``yes``. Pass the selected registry Service host or IP in
``phase_04_registry_man_address`` and set ``phase_04_registry_opt_port`` to its
configured port.

Do not paste the dashboard URL path ``/api/model_registry/v1alpha3``. Use a
unique model name and version for each registration, then set
``phase_05_deploy_man_namespace`` to the namespace containing the user PVC.

<!-- custom-content -->

## Dataset format

This pipeline supports Hugging Face, local/PVC JSON or JSONL, S3, and HTTP(S)
dataset sources. Hugging Face references are passed directly to Training Hub;
the other sources are normalized by the preparation component. Use
``hf://org/dataset`` or ``hf://org/dataset:config`` for Hugging Face data. The
``hf://`` prefix is optional when supplying a Hugging Face repository ID.

For private S3 data, ``phase_01_dataset_opt_data_connection_name`` identifies the optional RHOAI
Connection Secret. It defaults to ``s3-secret`` and is not needed for Hugging
Face, HTTP(S), or local/PVC inputs.

For the built-in single-turn tool-call mode, each row must contain
``question``, ``available_tools``, and ``messages``. The first three messages
must be a system message, a user message, and an assistant message containing
a tool call. Tool definitions must use the OpenAI function-tool format, and
the assistant tool call must provide a non-empty function name and
JSON-encoded arguments. For example:

```json
{
  "question": "Look up the value.",
  "available_tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup_value",
        "description": "Look up a value.",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "You are a tool-calling assistant."},
    {"role": "user", "content": "Look up the value."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {"name": "lookup_value", "arguments": "{}"}
        }
      ]
    }
  ]
}
```

See the [Training Hub loader format](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/lora_grpo.py#L381-L387).
