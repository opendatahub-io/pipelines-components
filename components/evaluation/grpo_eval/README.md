# Grpo Eval ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate GRPO training results from the shared pipeline workspace.

Reads ART's ``training_results.json`` from the caller-provided shared-PVC path. The GRPO pipeline supplies ``{workspace_path}/checkpoints/training_results.json``, while the complete path input keeps the component independent of the PVC mount location.

Required JSON fields are ``final_mean_reward``, ``reward_history``, ``full_match_history``, and ``timing_history``. The final reward and every history entry must be a finite number. Reward and full-match histories must have the same number of iterations. The logged ``mean_reward`` is ART's reported
final aggregate reward, while ``final_reward`` is the last entry in ``reward_history`` used for the promotion comparison.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `training_results_path` | `str` | `None` | Mounted path to ART's ``checkpoints/training_results.json`` file. |
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
    - Name: Pipelines, Version: >=2.16.1
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
