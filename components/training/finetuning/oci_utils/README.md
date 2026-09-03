# Oci Utils ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Check if a URI is an OCI reference.

Returns "true" or "false" as a string for use with dsl.If conditions, since dsl.If only supports comparing task outputs (not Python methods).

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `uri` | `str` | `None` |  |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` |  |

## Metadata 🗂️

- **Name**: oci_utils
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
  - fine_tuning
  - oci
  - model_download
  - modelcar
- **Last Verified**: 2026-06-26 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - briangallagher
    - efazal
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher
