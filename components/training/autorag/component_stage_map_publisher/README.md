# Component Stage Map Publisher ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Publish the component-to-stage map for dashboard consumption.

Reads the static JSON template from the embedded artifact (``run_status_templates/pipelines/``) and publishes it as a KFP artifact. Dashboards use this map to show expected components, stages, and steps before pipeline execution begins.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pipeline_id` | `str` | `None` | Pipeline identifier matching the template filename (e.g. ``documents-rag-optimization-pipeline``). |
| `run_id` | `str` | `None` | KFP run ID for tracking (from ``dsl.PIPELINE_JOB_ID_PLACEHOLDER``). |
| `component_stage_map` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the component-to-stage map. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` package with pipeline templates. |

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
  - autorag
  - run-status
- **Last Verified**: 2026-06-10 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR
