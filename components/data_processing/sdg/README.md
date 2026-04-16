# Sdg Hub ✨

> ⚠️ **Stability: beta** — This asset is not yet stable and may change.

## Overview 🧾

Run an SDG Hub flow to generate synthetic data.

Loads input data, selects and configures a flow, executes it, and writes the output as a JSONL artifact with execution metrics.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `output_artifact` | `dsl.Output[dsl.Dataset]` | `None` | KFP Dataset artifact for downstream components. |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | KFP Metrics artifact with execution stats. |
| `input_artifact` | `dsl.Input[dsl.Dataset]` | `None` | KFP Dataset artifact from upstream component (optional). |
| `input_pvc_path` | `str` | `""` | Path to JSONL input file on a mounted PVC (optional). |
| `flow_id` | `str` | `""` | Built-in flow ID from the SDG Hub registry. |
| `flow_yaml_path` | `str` | `""` | Path to a custom flow YAML file. |
| `model` | `str` | `""` | LiteLLM model identifier (e.g. 'openai/gpt-4o-mini'). |
| `max_concurrency` | `int` | `10` | Maximum concurrent LLM requests. |
| `checkpoint_pvc_path` | `str` | `""` | PVC path for checkpoints (enables resume). |
| `save_freq` | `int` | `100` | Checkpoint save frequency (number of samples). |
| `log_level` | `str` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR). |
| `temperature` | `float` | `-1.0` | LLM sampling temperature (0.0-2.0). Use -1 for flow default. |
| `max_tokens` | `int` | `-1` | Maximum response tokens. Use -1 for flow default. |
| `export_to_pvc` | `bool` | `False` | Whether to export output to PVC (in addition to KFP artifact). |
| `export_path` | `str` | `""` | Base PVC path for exports (required if export_to_pvc is True). |
| `runtime_params` | `dict` | `None` | Per-block parameter overrides as a dict of {block_name: {param: value}}. |

## Metadata 🗂️

- **Name**: sdg_hub
- **Stability**: beta
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: SDG Hub, Version: >=0.7.0
    - Name: LiteLLM, Version: >=1.0.0
- **Tags**:
  - sdg
  - synthetic_data_generation
  - llm
  - data_processing
- **Last Verified**: 2026-03-21 00:00:00+00:00
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
