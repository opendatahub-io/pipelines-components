# Component Stage Map Publisher ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Publish the component-to-stage-to-step map for dashboard consumption.

Reads the static JSON template from the package (``run_status_templates/pipelines/``) and publishes it as a KFP artifact. Dashboards use this map to show expected components, stages, and steps before pipeline execution begins.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pipeline_id` | `str` | `None` | Pipeline identifier matching the template filename (e.g. ``autogluon-tabular-training-pipeline``). |
| `run_id` | `str` | `None` | KFP run ID for tracking (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``). |
| `component_stage_map` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the component-to-stage-to-step map. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `None` |  |

## Metadata 🗂️

- **Name**: component_stage_map_publisher
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - automl
  - run-status
- **Last Verified**: 2026-05-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR

<!-- custom-content -->

### Artifact layout

The task writes a single file under the output artifact path:

```text
component_stage_map/
└── component_stage_map.json
```

**`component_stage_map.json`** contains:

| Field | Description |
| ----- | ----------- |
| `pipeline_id` | Matches the `pipeline_id` input and template filename (e.g. `autogluon-tabular-training-pipeline`). |
| `description` | Human-readable pipeline summary from the template. |
| `components` | Ordered list of `{id, description, stages[{id, description, steps?}]}`. |
| `kfp_run_id` | Run id from `dsl.PIPELINE_JOB_ID_PLACEHOLDER`. |
| `published_at` | UTC ISO timestamp when the map was published. |

Templates live under ``components/training/automl/shared/run_status_templates/pipelines/``. Dashboards should treat this artifact as the **expected** component-to-stage-to-step map; live progress comes from each component's ``component_status`` artifact.

#### Dashboard join keys

| Layer | Naming | Notes |
| ----- | ------ | ----- |
| Template `components[].id` | snake_case | Canonical id for dashboards (e.g. `autogluon_models_training`). |
| Runtime `component_status.json` → `component_id` | snake_case | Must match the template `components[].id` for the same logical step. |
| KFP root DAG task id | kebab-case | Compiled pipeline step name (e.g. `autogluon-models-training`); use only to resolve artifact paths. |
| KFP output parameter | snake_case | Always `component_status` for progress artifacts. |
| This component's output parameter | snake_case | `component_stage_map` → `component_stage_map.json`. |

Stage and step ids inside both JSON files are also snake_case. Canonical component ids live in [`run_status_templates/pipelines/`](../shared/run_status_templates/pipelines/); load them at runtime with ``pipeline_component_ids()`` from [`run_status.py`](../shared/run_status.py).
