# Search Space Preparation ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Search space preparation and validation for AutoRAG experiments.

Resolves and validates the requested MaaS models, builds the AutoRAG search space, and writes it as a JSON report. This step runs *before* text extraction so that unresponsive or misconfigured models fail the experiment fast, before any heavy document processing is performed.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `test_data` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact with benchmark questions and expected answers. Used for language detection during search-space preparation. |
| `search_space_report` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact for the JSON search space report. |
| `embedding_models` | `List[str]` | `None` | List of embedding model identifiers to try. |
| `generation_models` | `List[str]` | `None` | List of generation model identifiers to try. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) uses recursive chunking without contextual enrichment. "balanced" uses hybrid chunking with LLM contextual enrichment in the search space. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of search_space_preparation."""

from kfp import dsl
from kfp_components.components.training.autorag.search_space_preparation import search_space_preparation


@dsl.pipeline(name="search-space-preparation-example")
def example_pipeline():
    """Example pipeline using search_space_preparation."""
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    search_space_preparation(
        test_data=test_data.output,
        embedding_models=["ibm-granite/granite-embedding-278m-multilingual"],
        generation_models=["ibm-granite/granite-3.3-8b-instruct"],
    )

```

## Metadata 🗂️

- **Name**: search_space_preparation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: ~=0.12.0
    - Name: MaaS, Version: >=1.0.0
    - Name: pyYaml, Version: >=6.0.0
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - search-space
  - optimization
- **Last Verified**: 2026-08-24 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko
