# Search Space Preparation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Search space preparation for AutoRAG experiments.

Thin wrapper that delegates to ``ai4rag.components.optimization.search_space_preparation.prepare_search_space_report``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `test_data` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact with benchmark questions and expected answers. |
| `extracted_text` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact with extracted text documents. |
| `search_space_prep_report` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact for the JSON search space report. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `embedding_models` | `Optional[List]` | `None` | List of embedding model identifiers to try. |
| `generation_models` | `Optional[List]` | `None` | List of generation model identifiers to try. |
| `metric` | `str` | `None` | Quality metric for evaluation (e.g. "faithfulness"). |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) uses recursive chunking without contextual enrichment. "balanced" uses hybrid chunking with LLM contextual enrichment in the search space. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of search_space_preparation."""

from kfp import dsl
from kfp_components.components.training.autorag.search_space_preparation import search_space_preparation


@dsl.pipeline(name="search-space-preparation-example")
def example_pipeline(
    metric: str = "faithfulness",
):
    """Example pipeline using search_space_preparation.

    Args:
        metric: Evaluation metric name.
    """
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    search_space_preparation(
        test_data=test_data.output,
        extracted_text=extracted_text.output,
        metric=metric,
    )

```

## Metadata 🗂️

- **Name**: search_space_preparation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: ~=0.10.0
    - Name: pyYaml, Version: >=6.0.0
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - search-space
  - optimization
- **Last Verified**: 2026-07-10 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko
