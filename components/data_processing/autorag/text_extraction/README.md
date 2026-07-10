# Text Extraction ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Text Extraction component.

Thin wrapper that delegates to ``ai4rag.components.data.text_extraction.extract_text``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `documents_descriptor` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact containing documents_descriptor.json with bucket, prefix, and documents list. |
| `extracted_text` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact directory where DoclingDocument JSON files will be written. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `error_tolerance` | `Optional[float]` | `None` | Fraction of documents (0.0-1.0) allowed to fail without raising an error. None (the default) means zero tolerance. |
| `max_extraction_workers` | `Optional[int]` | `None` | Number of parallel worker processes used for text extraction. Defaults to 4. Set to None to use all available CPU cores. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) disables Docling table structure parsing. "balanced" enables TableFormer table reconstruction. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of text_extraction."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.text_extraction import text_extraction


@dsl.pipeline(name="text-extraction-example")
def example_pipeline():
    """Example pipeline using text_extraction."""
    documents_descriptor = dsl.importer(
        artifact_uri="gs://placeholder/documents_descriptor",
        artifact_class=dsl.Artifact,
    )
    text_extraction(documents_descriptor=documents_descriptor.output)

```

## Metadata 🗂️

- **Name**: text_extraction
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: docling, Version: >=1.0.0
- **Tags**:
  - data-processing
  - autorag
  - text-extraction
- **Last Verified**: 2026-06-17 12:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko
