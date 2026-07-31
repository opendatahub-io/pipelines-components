# Kubeflow Model Registry ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Register model to Kubeflow Model Registry with full provenance tracking.

Uses the upstream model artifact (input_model) produced by training, or falls back to PVC path if no artifact is provided.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pvc_mount_path` | `str` | `None` | PVC mount path for workspace storage. |
| `input_model` | `dsl.Input[dsl.Model]` | `None` | Model artifact from training step. |
| `input_metrics` | `dsl.Input[dsl.Metrics]` | `None` | Training metrics. |
| `eval_metrics` | `dsl.Input[dsl.Metrics]` | `None` | Evaluation metrics from lm-eval. |
| `eval_results` | `dsl.Input[dsl.Artifact]` | `None` | Full evaluation results JSON artifact. |
| `registry_address` | `str` | `""` | Model Registry server address (hostname or IP). |
| `registry_port` | `int` | `8080` | Model Registry server port (default: 8080). |
| `model_name` | `str` | `fine-tuned-model` | Name for the registered model. |
| `model_version` | `str` | `1.0.0` | Version string for the model (e.g. "1.0.0"). |
| `model_format_name` | `str` | `pytorch` | Model format name (e.g. "pytorch", "onnx"). |
| `model_format_version` | `str` | `1.0` | Model format version. |
| `model_description` | `str` | `""` | Optional description for the model. |
| `author` | `str` | `pipeline` | Author name for the model registration. |
| `shared_log_file` | `str` | `pipeline_log.txt` | Filename for shared pipeline log. |
| `source_pipeline_name` | `str` | `""` | Name of the source KFP pipeline. |
| `source_pipeline_run_id` | `str` | `""` | Unique ID of the pipeline run. |
| `source_pipeline_run_name` | `str` | `""` | Display name of the pipeline run. |
| `source_namespace` | `str` | `""` | Namespace where pipeline runs (auto-detected if empty). |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | |

## Metadata 🗂️

- **Name**: kubeflow_model_registry
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Model Registry, Version: >=0.3.4
- **Tags**:
  - deployment
  - model_registry
  - registration
  - kubeflow
- **Last Verified**: 2026-01-15 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - briangallagher
    - efazal
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/model-registry](https://github.com/kubeflow/model-registry)
