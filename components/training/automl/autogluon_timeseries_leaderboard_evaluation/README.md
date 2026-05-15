# Timeseries Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate refitted AutoGluon timeseries models and generate a leaderboard.

This component aggregates metrics from a list of model artifacts produced by ``autogluon_timeseries_models_full_refit`` (collected via ``dsl.Collected``) and generates an HTML-formatted leaderboard ranking the models by their performance metric.

Each artifact in ``models`` must have been produced by ``autogluon_timeseries_models_full_refit``, which writes the following layout under the artifact path::

{artifact.path}/{model_name_full}/metrics/metrics.json {artifact.path}/{model_name_full}/predictor/ {artifact.path}/{model_name_full}/notebooks/

Note: KFP does not propagate artifact metadata through executor inputs for collected artifact lists, so metrics are read from ``metrics/metrics.json`` on the filesystem rather than from ``artifact.metadata``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `models` | `List[dsl.Model]` | `None` | List of model artifacts from ``autogluon_timeseries_models_full_refit`` collected via ``dsl.Collected``. Each artifact provides metrics and location metadata for one refitted model. |
| `eval_metric` | `str` | `None` | Metric name for ranking (e.g. ``"MASE"``, ``"WAPE"``); leaderboard is sorted descending (AutoGluon uses higher-is-better convention so metrics like MASE are negated - higher value means better model). |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output artifact for the HTML-formatted leaderboard. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Artifact]` | `None` | Embedded shared files injected by the KFP runtime; provides ``leaderboard_html_template.html``. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', best_model=str)` | NamedTuple with ``best_model`` (str): display name of the top-ranked model. |

## Metadata 🗂️

- **Name**: timeseries_leaderboard_evaluation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - leaderboard
  - automl
  - timeseries
- **Last Verified**: 2026-03-31 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR

<!-- custom-content -->
## Usage Examples 💡

### Basic usage with collected refit model artifacts

Typical use after refitting multiple timeseries models: collect model artifacts from ``autogluon_timeseries_models_full_refit`` and pass the evaluation metric from the selection stage:

```python
from kfp import dsl
from kfp_components.components.training.automl.autogluon_timeseries_leaderboard_evaluation import timeseries_leaderboard_evaluation

@dsl.pipeline(name="automl-timeseries-leaderboard-pipeline")
def my_pipeline(refit_tasks, eval_metric):
    leaderboard_task = timeseries_leaderboard_evaluation(
        models=dsl.Collected(refit_tasks.outputs["model_artifact"]),
        eval_metric=eval_metric,
    )
```

### MASE metric (default for timeseries)

AutoGluon negates error metrics so that higher is always better. ``MASE`` is the standard default:

```python
timeseries_leaderboard_evaluation(
    models=dsl.Collected(refit_full_task.outputs["model_artifact"]),
    eval_metric="MASE",
)
```

### WAPE metric

```python
timeseries_leaderboard_evaluation(
    models=dsl.Collected(refit_full_task.outputs["model_artifact"]),
    eval_metric="WAPE",
)
```
