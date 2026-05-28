# Search Space Preparation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Runs an AutoRAG experiment's first phase which includes:

- AutoRAG search space creation given the user's constraints, - embedding and foundation models number limitation and initial selection,

Generates a .yml-formatted report including results of this experiment's phase. For its exact content please refer to the `search_space_prep_report_schema.yml` file.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `test_data` | `dsl.Input[dsl.Artifact]` | `None` | A path to a .json file containing questions and expected answers that can be retrieved from input documents. Necessary baseline for calculating quality metrics of RAG pipeline. |
| `extracted_text` | `dsl.Input[dsl.Artifact]` | `None` | A path to either a single file or a folder of files. The document(s) will be sampled and used during the models selection process. |
| `search_space_prep_report` | `dsl.Output[dsl.Artifact]` | `None` | kfp-enforced argument specifying an output artifact. Provided by kfp backend automatically. |
| `embedding_models` | `Optional[List]` | `None` | List of embedding model identifiers to try out in the experiment process. This list, if too long, will undergo models preselection (limiting). |
| `generation_models` | `Optional[List]` | `None` | List of generation model identifiers to try out in the experiment process. This list, if too long, will undergo models preselection (limiting). |
| `metric` | `str` | `None` | Quality metric to evaluate the intermediate RAG patterns. |

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
    - Name: ai4rag, Version: ~=0.6.1
    - Name: pyYaml, Version: >=6.0.0
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - search-space
  - optimization
- **Last Verified**: 2026-05-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
