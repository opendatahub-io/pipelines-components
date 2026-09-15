# Grpo Eval ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate GRPO training results from the shared pipeline workspace.

Reads the provisional ``training_results.json`` contract written by the GRPO training component. The file path is supplied by the pipeline, allowing the shared PVC layout to change without changing this component's public interface.

Required JSON fields are ``mean_reward``, ``full_match_rate``, ``reward_history``, and ``timing_history``. The first two fields and every reward-history entry must be finite numbers. ``timing_history`` must be a list; its entry schema intentionally remains producer-defined while ART's final output
contract is being established.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `training_results_path` | `str` | `None` | Mounted path to the training component's ``training_results.json`` file. |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | KFP Metrics artifact receiving the GRPO scalar metrics. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('GrpoEvalOutputs', [('promotion_passed', bool)])` | Named output ``promotion_passed``. It is true only when the final reward is strictly greater than the initial reward. A single reward value does not demonstrate improvement and returns false. |

## Metadata 🗂️

- **Name**: grpo_eval
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - evaluation
  - grpo
  - reinforcement_learning
  - metrics
  - finetuning
- **Last Verified**: 2026-09-15 00:00:00+00:00
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
