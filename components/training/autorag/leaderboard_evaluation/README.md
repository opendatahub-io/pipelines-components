# Leaderboard Evaluation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Build an HTML leaderboard artifact from RAG pattern evaluation results.

Reads pattern.json from each subdirectory of rag_patterns (produced by rag_templates_optimization) and generates a single HTML table: Pattern_Name, mean_* metrics (e.g. mean_answer_correctness, mean_faithfulness), then config columns (chunking.*, embeddings.model_id, retrieval.* including
search_mode ("hybrid" | "vector" per ai4rag), ranker_strategy, generation.model_id). Writes the HTML to html_artifact.path.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `rag_patterns` | `dsl.InputPath(dsl.Artifact)` | `None` | Path to the directory of RAG patterns; each subdir contains pattern.json (pattern_name, indexing_params, rag_params, scores, execution_time, final_score). |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output HTML artifact; the leaderboard table is written to html_artifact.path (single file). |
| `optimization_metric` | `str` | `faithfulness` | Name of the metric used to rank patterns (e.g. faithfulness, answer_correctness, context_correctness). Shown in the leaderboard subtitle. Defaults to "faithfulness". |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of leaderboard_evaluation."""

from kfp import dsl
from kfp_components.components.training.autorag.leaderboard_evaluation import leaderboard_evaluation


@dsl.pipeline(name="autorag-leaderboard-evaluation-example")
def example_pipeline(
    optimization_metric: str = "faithfulness",
):
    """Example pipeline using leaderboard_evaluation.

    Args:
        optimization_metric: Metric to optimize for.
    """
    rag_patterns = dsl.importer(
        artifact_uri="gs://placeholder/rag_patterns",
        artifact_class=dsl.Artifact,
    )
    leaderboard_evaluation(
        rag_patterns=rag_patterns.output,
        optimization_metric=optimization_metric,
    )

```

## Metadata 🗂️

- **Name**: leaderboard_evaluation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
  - autorag
  - html-artifact
  - leaderboard
  - evaluation
- **Last Verified**: 2026-02-10 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
