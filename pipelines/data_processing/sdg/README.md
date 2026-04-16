# Sdg Hub ✨

> ⚠️ **Stability: beta** — This asset is not yet stable and may change.

## Overview 🧾

Run SDG LLM test flow end-to-end.

Creates sample input data, runs the LLM test flow via the SDG Hub component, and outputs generated data as a KFP artifact.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `model` | `str` | `openai/gpt-4o-mini` | LiteLLM model identifier. |
| `max_concurrency` | `int` | `1` | Max concurrent LLM requests. |
| `temperature` | `float` | `0.7` | LLM sampling temperature. |
| `max_tokens` | `int` | `256` | Max response tokens. |

## Metadata 🗂️

- **Name**: sdg_hub
- **Stability**: beta
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: SDG Hub, Version: >=0.7.0
    - Name: LiteLLM, Version: >=1.0.0
- **Last Verified**: 2026-03-21 00:00:00+00:00
- **Tags**:
  - sdg
  - synthetic_data_generation
  - llm
  - data_processing
- **Owners**:
  - Approvers:
    - beatsmonster
    - shivchander
    - eshwarprasadS
    - abhi1092
  - Reviewers:
    - beatsmonster
    - shivchander
    - eshwarprasadS
    - abhi1092

## Additional Resources 📚

- **Documentation**: [https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub)
