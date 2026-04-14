# LoRA Pipeline

> **Stability: alpha** — This asset is not yet stable and may change.

## Overview

A 4-stage ML pipeline for parameter-efficient fine-tuning with LoRA:

1. **Dataset Download** — Prepares training data from HuggingFace, S3, or HTTP
2. **LoRA Training** — Fine-tunes using the unsloth backend (Kubeflow Trainer `TrainJob`)
3. **Model Serving + EvalHub Evaluation** — KServe (vLLM) deployment and EvalHub benchmarks
4. **Model Registry** — Registers the trained model in Kubeflow Model Registry

The eval step uses the shared `evalhub_evaluate` component: it can create a `ServingRuntime` + `InferenceService`, run EvalHub jobs, replace an existing same-named InferenceService, and delete serving resources after evaluation (see component defaults).

## Working configuration (reference)

These values match a successful deployment on OpenShift AI / RHOAI with Data Science Pipelines. Adjust namespaces, ServiceAccounts, and URLs for your cluster.

### Platform

| Item | Example | Notes |
|------|---------|--------|
| Pipeline project / namespace | `test-kfp` | Where the pipeline run executes |
| Pipeline ServiceAccount | `pipeline-runner-dspa` | Check with `oc get sa -n <namespace>` |
| `ClusterTrainingRuntime` | `training-hub` | Must exist; matches `phase_02_train_opt_runtime` |
| KServe + EvalHub namespace | `test-kfp` | Set `phase_03_eval_man_namespace` (often same as pipeline project) |
| EvalHub API URL | `https://evalhub.evalhub-test.svc.cluster.local:8443` | Cluster-internal HTTPS; `phase_03_eval_man_evalhub_url` |
| EvalHub tenant | `test-kfp` | Set `phase_03_eval_opt_evalhub_tenant`, **or** leave empty: `evalhub_evaluate` uses `phase_03_eval_man_namespace` as the `X-Tenant` value when tenant is blank (avoids EvalHub **400** on `/evaluations/providers`) |

### Secrets

| Secret | Required? | Purpose |
|--------|-------------|--------|
| `evalhub-auth` (`EVALHUB_TOKEN`) | **Yes** for evaluation | Token accepted by **your EvalHub server** |
| `hf-token` (`HF_TOKEN`) | Optional | Gated HuggingFace models and serving pulls |
| `kubernetes-credentials` | Optional | Training uses the step pod’s in-cluster ServiceAccount by default |
| `s3-secret` | Optional | S3 datasets / artifacts |

```bash
oc create secret generic evalhub-auth \
  --from-literal=EVALHUB_TOKEN="<evalhub-api-token>" \
  -n <pipeline-namespace>
```

### RBAC

Apply **before** the first run (replace namespace and pipeline ServiceAccount):

**1. Kubeflow Trainer** — lists cluster-scoped `ClusterTrainingRuntime` and creates `TrainJob` in the pipeline namespace (avoids **403** on `clustertrainingruntimes`):

```bash
sed -e 's/PIPELINE_NAMESPACE/<pipeline-namespace>/g' \
    -e 's/PIPELINE_SA/<pipeline-serviceaccount>/g' \
    pipelines/training/finetuning/lora_minimal/trainer_pipeline_rbac.yaml | oc apply -f -
```

(Run from the repository root, or use the path to `trainer_pipeline_rbac.yaml` next to the minimal LoRA pipeline.)

**2. KServe** — Role + RoleBinding in **`phase_03_eval_man_namespace`** for InferenceService / ServingRuntime lifecycle:

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

**3. EvalHub tenant label** on the serving / tenant namespace:

```bash
oc label ns <serving-namespace> evalhub.trustyai.opendatahub.io/tenant=""
```

### Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| **403** on `clustertrainingruntimes` | Trainer RBAC not applied or wrong ServiceAccount in binding |
| **401** on EvalHub | Missing/invalid `evalhub-auth` |
| **400** on `/evaluations/providers` with empty tenant | Set `phase_03_eval_opt_evalhub_tenant` **or** ensure `phase_03_eval_man_namespace` is set (tenant falls back to namespace) |
| KServe denied | KServe Role not bound in `phase_03_eval_man_namespace` |

## Pipeline parameters (summary)

### Required / important

| Parameter | Description |
|-----------|-------------|
| `phase_01_dataset_man_data_uri` | Dataset URI (`hf://`, `s3://`, `https://`, …) |
| `phase_03_eval_man_evalhub_url` | EvalHub base URL |
| `phase_03_eval_man_namespace` | Namespace for KServe resources (also used as EvalHub tenant if `phase_03_eval_opt_evalhub_tenant` is empty) |
| `phase_04_registry_man_address` | Model Registry (empty skips registration) |

### Evaluation and serving (optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phase_03_eval_opt_evalhub_tenant` | `""` | EvalHub `X-Tenant`; if empty, namespace is used |
| `phase_03_eval_opt_benchmarks_json` | `""` | Custom benchmarks JSON; empty uses EvalHub defaults (e.g. ARC Easy) |
| `phase_03_eval_opt_tokenizer` | `""` | Tokenizer for benchmarks |
| `phase_03_eval_opt_timeout` | `3600` | EvalHub job wait (seconds) |
| `phase_03_eval_opt_serve_*` | (various) | vLLM / KServe sizing, image, GPU, ports |

See `pipeline.py` docstring for the full list of training and registry parameters.

## Example run (verified pattern)

| Parameter | Value |
|-----------|-------|
| `phase_01_dataset_man_data_uri` | `hf://b-mc2/sql-create-context` |
| `phase_02_train_man_train_model` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `phase_03_eval_man_evalhub_url` | `https://evalhub.evalhub-test.svc.cluster.local:8443` |
| `phase_03_eval_man_namespace` | `test-kfp` |
| `phase_03_eval_opt_evalhub_tenant` | `test-kfp` *(or leave empty to use namespace)* |
| `phase_04_registry_man_address` | *(empty to skip, or your registry URL)* |

`phase_04_registry_man_reg_name` defaults to `lora-model`; the eval step derives the InferenceService name from it (e.g. `lora-model`).

## Compiling and running

### Generate `pipeline.yaml`

```bash
PYTHONPATH=/path/to/dj-pipelines-components \
  python3 pipelines/training/finetuning/lora/pipeline.py
```

Output: `pipelines/training/finetuning/lora/pipeline.yaml`.

### Upload and run

1. Register/upload `pipeline.yaml` in Data Science Pipelines.
2. Create a run with parameters and required secrets/RBAC in place.

Re-upload **only** when `pipeline.py` or embedded components change. Cluster-only changes (secrets, RBAC, labels) do not require a new YAML.

## Metadata

- **Name**: `lora_pipeline` (pipeline display name: `lora-pipeline`)
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow Pipelines >= 2.15.2
  - Kubeflow Trainer (ClusterTrainingRuntime / TrainJob)
  - KServe
  - EvalHub server + EvalHub SDK (see `components/evaluation/evalhub_eval`)
- **Tags**: training, fine_tuning, lora, peft, evalhub, kserve
- **Owners**: see `OWNERS`

## Additional resources

- [Kubeflow Trainer](https://github.com/kubeflow/trainer)
- [KServe](https://kserve.github.io/website/)
- [EvalHub docs](https://eval-hub.github.io/)
- [LoRA minimal README](../lora_minimal/README.md) (narrower pipeline; shared RBAC manifest path)
