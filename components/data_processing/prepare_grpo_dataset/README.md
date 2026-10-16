# Prepare Grpo Dataset ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Resolve any supported source for Training Hub's GRPO loader.

Hugging Face references are returned without downloading or rewriting them. Local/PVC JSON or JSONL files, S3 JSON/JSONL objects, and HTTP(S) JSON/JSONL resources are converted to Training Hub's local canonical format on the shared PVC. The component intentionally does not subset, split, or sample
the data; Training Hub owns ``n_train`` and ``n_val``.

Private S3 sources use ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``. ``AWS_S3_ENDPOINT`` and ``AWS_DEFAULT_REGION`` are optional for S3-compatible services.

### S3 credentials in a pipeline 🔐

RHOAI users should create an object-storage Connection in their Data Science
Project. The Connection creates the Kubernetes Secret used by the pipeline.
At pipeline-authoring time, inject that Secret into the task with
``kfp.kubernetes.use_secret_as_env``:

```python
from kfp import kubernetes

dataset_task = prepare_grpo_dataset(
    dataset_uri=dataset_uri,
    pvc_mount_path="/mnt/pipeline",
)
kubernetes.use_secret_as_env(
    task=dataset_task,
    secret_name=data_connection_name,
    secret_key_to_env={
        "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
        "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
    },
    optional=True,
)
```

Here, ``data_connection_name`` is the name of the Kubernetes Secret created
for the Connection. The component intentionally does not expose this as a
component input: ``use_secret_as_env`` configures the compiled KFP task, while
the component itself only consumes the resulting environment variables. For
an S3 URI, provide the Connection Secret or configure an equivalent workload
identity/default AWS credential chain.

See the Training Hub GRPO loader contract: https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/lora_grpo.py

The local canonical single-turn format is ``question``, ``tools``, ``target_tool_name``, ``target_arguments``, and an optional ``system_prompt``. Multi-turn rows retain ``messages`` and are normalized to modern OpenAI ``tool_calls``/``tool`` messages so Training Hub can decompose them.

Malformed rows that fail expected validation are skipped. The component fails
when more than 10% of the input rows are invalid, or when no valid rows remain.
Unexpected implementation errors are not treated as row validation failures.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `dataset_uri` | `str` | `None` | Hugging Face URI/repository ID, ``s3://`` object, ``http://`` or ``https://`` JSON/JSONL URL, ``pvc://`` path, or local JSON/JSONL path visible in the component container. |
| `pvc_mount_path` | `str` | `None` | Mount point of the shared PVC used for caches and normalized local data. |
| `dataset_config` | `str` | `""` | Optional Hugging Face dataset configuration. An inline ``:config`` suffix on an HF URI takes precedence. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('PrepareDatasetOutputs', [('resolved_data_path', str), ('resolved_data_config', str), ('use_hf_direct', bool)])` | A Training Hub data path, config, and flag indicating whether the path should be passed through the direct Hugging Face loader. |

## Metadata 🗂️

- **Name**: prepare_grpo_dataset
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: AWS S3, Version: >=1.0.0
- **Tags**:
  - data_processing
  - dataset
  - grpo
  - huggingface
  - s3
- **Last Verified**: 2026-10-15 00:00:00+00:00
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

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
