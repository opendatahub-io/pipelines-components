# Speculator ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Run the upstream Speculators performance sweep against a speculative vLLM server.

The component starts vLLM with the verifier and draft model, then runs the pinned upstream ``scripts/evaluate/evaluate.py sweep`` against its OpenAI endpoint. Upstream results (acceptance.csv, perf_results.csv, raw GuideLLM JSON, and provenance) are returned in ``output_results``.

A local JSONL/PVC dataset may contain prompts, chat messages, or conversation records. Local records are normalized to prompt JSONL and deterministically capped at ``evaluation_max_samples``. A Hugging Face dataset ID is passed directly to the upstream script, using its HumanEval example subset.

Use paths relative to the configured mount without a PVC prefix. With ``pvc://`` syntax, prefix the path with the exact mounted claim name; for example, use ``pvc://shared/models/ultrachat.jsonl`` when the claim is ``shared``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | KFP metrics for acceptance and sweep performance results. |
| `output_results` | `dsl.Output[dsl.Artifact]` | `None` | Directory artifact containing upstream CSVs and raw GuideLLM output. |
| `verifier_model` | `str` | `None` | Verifier directory under the model mount, optionally using pvc://<claim>/<path> syntax. |
| `draft_model_path` | `str` | `None` | Draft directory under the model mount, optionally using pvc://<claim>/<path> syntax. |
| `evaluation_dataset_uri` | `str` | `None` | Hugging Face dataset ID, relative path under the dataset mount, or ``pvc://<claim>/<path>``. |
| `evaluation_speculator_type` | `str` | `None` | Speculator method configured on the vLLM server. |
| `evaluation_dataset_pvc` | `str` | `""` | Mounted PVC claim name. Every ``pvc://`` URI must use this exact claim name; relative paths need no PVC prefix. |
| `evaluation_max_samples` | `int` | `80` | Maximum GuideLLM requests at each sweep point. |
| `evaluation_max_tokens` | `int` | `256` | Maximum generation tokens for each request. |
| `evaluation_temperature` | `float` | `0.0` | Sampling temperature passed through to GuideLLM. |
| `evaluation_num_speculative_tokens` | `int` | `5` | Number of draft tokens proposed by vLLM. |
| `evaluation_gpu_memory_utilization` | `float` | `0.9` | vLLM GPU memory utilization fraction. |
| `evaluation_port` | `int` | `8000` | Local port for the temporary vLLM endpoint. |
| `evaluation_startup_timeout_seconds` | `int` | `900` | Maximum wait for the vLLM health endpoint. |
| `evaluation_timeout_seconds` | `int` | `7200` | Maximum time allowed for the upstream evaluation sweep. |
| `evaluation_sample_seed` | `int` | `42` | Seed used when sampling a local JSONL dataset. |
| `evaluation_model_mount_path` | `str` | `/mnt/persistent` | Root directory for verifier and draft models; resolved paths must stay inside it. |
| `evaluation_dataset_mount_path` | `str` | `/mnt/persistent` | Root directory for local datasets; resolved paths must stay inside it. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `None` |  |

## Metadata 🗂️

- **Name**: speculator
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: vLLM, Version: >=0.9.0
    - Name: GuideLLM, Version: ~=0.7.1
- **Tags**:
  - evaluation
  - llm
  - speculator
  - speculative-decoding
  - metrics
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

- **Documentation**: [https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/evaluating_performance/](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/evaluating_performance/)
