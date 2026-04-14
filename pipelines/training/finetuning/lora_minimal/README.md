# Lora Minimal Pipeline

> **Stability: alpha** — This asset is not yet stable and may change.

## Overview

A minimal 4-stage ML pipeline for fine-tuning language models with LoRA:

1. **Dataset Download** — Prepares training data from HuggingFace, S3, or HTTP
2. **LoRA Training** — Fine-tunes using unsloth backend (low-rank adapters)
3. **Model Serving + EvalHub Evaluation** — Deploys the model via KServe and evaluates via EvalHub
4. **Model Registry** — Registers trained model to Kubeflow Model Registry

## Working configuration (reference)

The following matches a **verified** end-to-end run on OpenShift AI / RHOAI with Data Science Pipelines. Replace namespaces, ServiceAccount names, and URLs with your environment’s values.

### Platform layout

| Item | Example | Notes |
|------|---------|--------|
| Pipeline project / namespace | `test-kfp` | Namespace where the pipeline run executes |
| Pipeline ServiceAccount | `pipeline-runner-dspa` | Confirm with `oc get sa -n <pipeline-namespace>` (names vary by platform) |
| `ClusterTrainingRuntime` | `training-hub` | Must exist on the cluster; must match `phase_02_train_opt_runtime` |
| KServe + EvalHub namespace | `test-kfp` | Often the **same** as the pipeline namespace for a simple setup; must match `phase_03_eval_man_namespace` |
| EvalHub API URL | `https://evalhub.evalhub-test.svc.cluster.local:8443` | Cluster-internal HTTPS URL reachable from pipeline pods; component uses TLS verify skip when configured |
| EvalHub tenant header | `test-kfp` | Set `phase_03_eval_opt_evalhub_tenant` to the **serving** namespace name EvalHub expects |

### Secrets (minimum for a successful full run)

| Secret | Required? | Purpose |
|--------|-------------|--------|
| `evalhub-auth` (`EVALHUB_TOKEN`) | **Yes** for the eval step | Authenticates to EvalHub; must be a token **accepted by your EvalHub server** (see your EvalHub / platform docs—not necessarily the same as `oc login`) |
| `hf-token` (`HF_TOKEN`) | Optional | Gated HuggingFace models; also used by the serving step when pulling models |
| `kubernetes-credentials` | Optional | Training submits `TrainJob` using the **pod’s in-cluster ServiceAccount** by default |
| `s3-secret` | Optional | Dataset / artifact storage when using S3 URIs |

Create secrets (adjust namespace):

```bash
# Required for EvalHub evaluation
oc create secret generic evalhub-auth \
  --from-literal=EVALHUB_TOKEN="<evalhub-api-token>" \
  -n <pipeline-namespace>

# Optional — HuggingFace gated models
oc create secret generic hf-token \
  --from-literal=HF_TOKEN="hf_..." \
  -n <pipeline-namespace>

# Optional — only if you do not rely on in-cluster credentials for training
oc create secret generic kubernetes-credentials \
  --from-literal=KUBERNETES_SERVER_URL="$(oc whoami --show-server)" \
  --from-literal=KUBERNETES_AUTH_TOKEN="$(oc whoami -t)" \
  -n <pipeline-namespace> \
  --dry-run=client -o yaml | oc apply -f -
```

### RBAC (apply before first run)

Apply in this order; use your real `<pipeline-namespace>` and `<pipeline-serviceaccount>`.

**1. Kubeflow Trainer** — required or the training step fails with `403` on `clustertrainingruntimes`:

```bash
sed -e 's/PIPELINE_NAMESPACE/<pipeline-namespace>/g' \
    -e 's/PIPELINE_SA/<pipeline-serviceaccount>/g' \
    pipelines/training/finetuning/lora_minimal/trainer_pipeline_rbac.yaml | oc apply -f -
```

**2. KServe** — required for creating/deleting InferenceServices and ServingRuntimes in the **serving** namespace (use the same namespace as `phase_03_eval_man_namespace`):

```bash
oc apply -n <serving-namespace> -f - <<'EOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: kserve-manager
rules:
  - apiGroups: ["serving.kserve.io"]
    resources: ["inferenceservices", "servingruntimes"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pipeline-runner-kserve
subjects:
  - kind: ServiceAccount
    name: <pipeline-serviceaccount>
    namespace: <pipeline-namespace>
roleRef:
  kind: Role
  name: kserve-manager
  apiGroup: rbac.authorization.k8s.io
EOF
```

**3. EvalHub tenant label** — on the namespace used for serving / tenant (same as `phase_03_eval_opt_evalhub_tenant` in simple setups):

```bash
oc label ns <serving-namespace> evalhub.trustyai.opendatahub.io/tenant=""
```

### Troubleshooting quick reference

| Symptom | Likely cause |
|---------|----------------|
| `403` on `clustertrainingruntimes` | Trainer RBAC not applied or wrong ServiceAccount in `trainer_pipeline_rbac.yaml` |
| `401` on EvalHub `/evaluations/providers` | Missing or invalid `evalhub-auth` secret; `EVALHUB_TOKEN` not accepted by EvalHub |
| KServe create/delete denied | KServe Role not bound in `phase_03_eval_man_namespace` |

## Prerequisites

Use the **Working configuration** section above for the full checklist. Summary:

### Kubernetes Secrets

| Secret | Namespace | Keys | Purpose |
|--------|-----------|------|---------|
| `evalhub-auth` | pipeline namespace | `EVALHUB_TOKEN` | **Required** for EvalHub evaluation |
| `hf-token` | pipeline namespace | `HF_TOKEN` | Optional; gated HuggingFace models |
| `kubernetes-credentials` | pipeline namespace | `KUBERNETES_SERVER_URL`, `KUBERNETES_AUTH_TOKEN` | Optional; training defaults to in-cluster credentials |
| `s3-secret` | pipeline namespace | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Optional; S3 datasets |

### RBAC Permissions

See **Working configuration → RBAC**. Replace `<pipeline-namespace>` and the pipeline ServiceAccount name (e.g. `pipeline-runner-dspa` on OpenShift AI).

#### Kubeflow Trainer (LoRA training step)

The training step lists **cluster-scoped** `ClusterTrainingRuntime` objects and creates **TrainJob** resources in the pipeline namespace.

```bash
sed -e 's/PIPELINE_NAMESPACE/<pipeline-namespace>/g' \
    -e 's/PIPELINE_SA/<your-pipeline-serviceaccount>/g' \
    pipelines/training/finetuning/lora_minimal/trainer_pipeline_rbac.yaml | oc apply -f -
```

#### KServe (model serving + teardown)

See the KServe Role and RoleBinding in **Working configuration**.

### EvalHub Tenant

```bash
oc label ns <serving-namespace> evalhub.trustyai.opendatahub.io/tenant=""
```

## Pipeline Parameters

### Stage 1: Dataset Download

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_01_dataset_man_data_uri` | `str` | *required* | Dataset location (`hf://org/repo`, `s3://bucket/path`, `https://url`) |
| `phase_01_dataset_man_data_split` | `float` | `0.9` | Train/eval split ratio (1.0 = no split) |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |

### Stage 2: LoRA Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID) |
| `phase_02_train_man_train_epochs` | `int` | `2` | Number of training epochs |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size per optimizer step |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker |
| `phase_02_train_man_train_tokens` | `int` | `32000` | Max tokens per GPU (memory cap) |
| `phase_02_train_man_lora_r` | `int` | `16` | LoRA rank (4, 8, 16, 32, 64) |
| `phase_02_train_man_lora_alpha` | `int` | `32` | LoRA scaling factor (typically 2x rank) |
| `phase_02_train_opt_learning_rate` | `float` | `2e-4` | Learning rate |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_use_liger` | `bool` | `True` | Enable Liger kernel optimizations |
| `phase_02_train_opt_lora_dropout` | `float` | `0.0` | Dropout rate for LoRA layers |
| `phase_02_train_opt_lora_target_modules` | `str` | `""` | Modules to apply LoRA (empty = auto-detect) |
| `phase_02_train_opt_lora_load_in_4bit` | `bool` | `True` | Enable 4-bit quantization (QLoRA) |
| `phase_02_train_opt_lora_load_in_8bit` | `bool` | `False` | Enable 8-bit quantization |
| `phase_02_train_opt_dataset_type` | `str` | `""` | Dataset format type (empty = chat template) |
| `phase_02_train_opt_field_messages` | `str` | `""` | Field name for messages column (chat-format) |
| `phase_02_train_opt_field_instruction` | `str` | `""` | Field name for instruction/question column |
| `phase_02_train_opt_field_input` | `str` | `""` | Field name for input/context column |
| `phase_02_train_opt_field_output` | `str` | `""` | Field name for output/answer column |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | ClusterTrainingRuntime name |

### Stage 3: Model Serving + EvalHub Evaluation

**Why evaluation can take several minutes:** EvalHub runs benchmarks (e.g. ARC Easy via lm_evaluation harness) in its own job; runtime depends on model throughput, GPU, and full dataset size. The pipeline step also polls EvalHub every 15s until the job finishes. For a quicker smoke test, extend the pipeline later with `num_examples` on `evalhub_evaluate` or a smaller benchmark list.

**Serving cleanup:** The `evalhub_evaluate` component (defaults) **deletes** the KServe InferenceService (and the managed ServingRuntime when not using an external `existing_runtime`) **after** evaluation, and **recreates** serving if a same-named InferenceService already exists—so a previous run’s `lora-model` is not left behind and you do not evaluate a stale deployment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_03_eval_man_evalhub_url` | `str` | `""` | EvalHub API base URL (e.g. `https://evalhub.evalhub-test.svc.cluster.local:8443`) |
| `phase_03_eval_man_namespace` | `str` | `""` | Kubernetes namespace for KServe model serving |
| `phase_03_eval_opt_evalhub_tenant` | `str` | `""` | EvalHub tenant namespace (must match the serving namespace) |
| `phase_03_eval_opt_tokenizer` | `str` | `""` | HuggingFace tokenizer ID for evaluation (typically the base model ID) |

### Stage 4: Model Registry

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_name` | `str` | `lora-model` | Model name in registry |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port |

## Using Non-Chat Datasets

The pipeline supports datasets that are not in chat template format (e.g. instruction/input/output style datasets like `b-mc2/sql-create-context`). Use the field-mapping parameters to tell the training component how to interpret the columns:

```
phase_02_train_opt_field_instruction = question
phase_02_train_opt_field_input       = context
phase_02_train_opt_field_output      = answer
```

The training component automatically converts these to chat format before passing to the LoRA training backend.

## Example run configuration (verified parameters)

For fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` on `b-mc2/sql-create-context` with serving and EvalHub in namespace `test-kfp`:

| Parameter | Value |
|-----------|-------|
| `phase_01_dataset_man_data_uri` | `hf://b-mc2/sql-create-context` |
| `phase_02_train_man_train_model` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `phase_02_train_man_train_epochs` | `1` |
| `phase_02_train_opt_field_instruction` | `question` |
| `phase_02_train_opt_field_input` | `context` |
| `phase_02_train_opt_field_output` | `answer` |
| `phase_03_eval_man_evalhub_url` | `https://evalhub.evalhub-test.svc.cluster.local:8443` |
| `phase_03_eval_man_namespace` | `test-kfp` |
| `phase_03_eval_opt_evalhub_tenant` | `test-kfp` |
| `phase_03_eval_opt_tokenizer` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `phase_04_registry_man_address` | *(empty to skip registry, or your Model Registry URL)* |

`phase_04_registry_man_reg_name` defaults to `lora-model`; the eval step derives the KServe InferenceService name from this (e.g. `lora-model`). The bundled `evalhub_evaluate` component **replaces** an existing same-named deployment and **deletes** serving resources after evaluation so repeated runs do not accumulate stale InferenceServices.

## Compiling and running the pipeline

### Compile `pipeline.yaml`

From the repository root:

```bash
PYTHONPATH=/path/to/dj-pipelines-components \
  python3 pipelines/training/finetuning/lora_minimal/pipeline.py
```

This writes `pipelines/training/finetuning/lora_minimal/pipeline.yaml`.

### Upload and create a run

1. In **Data Science Pipelines**, upload `pipeline.yaml` (or register it via your usual CI) to create or update a pipeline **version**.
2. Create a **new run** and set parameters (see tables above). Ensure secrets and RBAC from **Working configuration** are already applied in the target namespace.

### When to re-upload `pipeline.yaml`

| Change | Re-upload required? |
|--------|---------------------|
| Cluster secrets, RBAC, EvalHub labels | **No** — apply with `oc`; start a new run |
| Edits to `pipeline.py` or component code | **Yes** — recompile, upload new version, then run |

Operational fixes (Trainer RBAC, EvalHub token, KServe permissions) do **not** require a new `pipeline.yaml` unless the pipeline definition itself changed.

## Metadata

- **Name**: lora_minimal_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow Pipelines >= 2.15.2
  - Kubeflow Trainer >= 0.1.0
  - KServe >= 0.11.0
  - EvalHub SDK v0.1.4
  - HuggingFace Datasets >= 2.14.0
- **Tags**: training, fine_tuning, lora, peft, evalhub, kserve, minimal
- **Owners**: briangallagher, Fiona-Waters, kramaranya, MStokluska, szaher

## Additional Resources

- [Kubeflow Trainer](https://github.com/kubeflow/trainer)
- [KServe](https://kserve.github.io/website/)
- [EvalHub Docs](https://eval-hub.github.io/)
- [vLLM](https://docs.vllm.ai/)
