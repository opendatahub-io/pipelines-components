# Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate refitted AutoGluon models and generate a leaderboard.

This component reads pre-computed metrics from a combined models artifact produced by ``autogluon_models_training`` and generates an HTML-formatted leaderboard ranking the models by their performance metric.

The artifact layout expected under ``models_artifact.path``::

models_artifact.path / <model_name>_FULL / metrics / metrics.json predictor / predictor.pkl notebooks / automl_predictor_notebook.ipynb

``models_artifact.metadata["model_names"]`` must contain the list of refitted model display names (e.g. ``["LightGBM_BAG_L1_FULL", ...]``).

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `models_artifact` | `dsl.Input[dsl.Model]` | `None` | Combined Model artifact from ``autogluon_models_training`` with ``metadata["model_names"]`` and per-model subdirectories. |
| `eval_metric` | `str` | `None` | Metric name for ranking (e.g. ``"accuracy"``, ``"root_mean_squared_error"``); leaderboard sorted descending (AutoGluon uses higher-is-better convention). |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output artifact for the HTML-formatted leaderboard. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Artifact]` | `None` | Embedded component files injected by the KFP runtime; provides ``leaderboard_html_template.html``. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', best_model=str)` |  |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of leaderboard_evaluation."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import (
    leaderboard_evaluation,
)


@dsl.pipeline(name="autogluon-leaderboard-evaluation-example")
def example_pipeline(
    eval_metric: str = "root_mean_squared_error",
):
    """Example pipeline using leaderboard_evaluation.

    Args:
        eval_metric: Evaluation metric name.
    """
    models_artifact = dsl.importer(
        artifact_uri="gs://placeholder/models_artifact",
        artifact_class=dsl.Model,
    )
    leaderboard_evaluation(
        models_artifact=models_artifact.output,
        eval_metric=eval_metric,
    )

```

## Metadata 🗂️

- **Name**: leaderboard_evaluation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - leaderboard
  - automl
- **Last Verified**: 2026-03-06 11:05:29+00:00
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

Typical use after refitting multiple models: collect model artifacts and pass the evaluation metric from the selection stage:

```python
from kfp import dsl
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import leaderboard_evaluation

@dsl.pipeline(name="automl-leaderboard-pipeline")
def my_pipeline(refit_tasks, eval_metric):
    leaderboard_evaluation(
        models=dsl.Collected(refit_tasks.outputs["model_artifact"]),
        eval_metric=eval_metric,
    )
```

### Classification (accuracy)

```python
leaderboard_evaluation(
    models=dsl.Collected(refit_full_task.outputs["model_artifact"]),
    eval_metric="accuracy",
)
```

### Regression (RMSE)

```python
leaderboard_evaluation(
    models=dsl.Collected(refit_full_task.outputs["model_artifact"]),
    eval_metric="root_mean_squared_error",
)
```
