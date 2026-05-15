# Text Extraction ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Text Extraction component.

Reads the documents_descriptor JSON (from documents_discovery), fetches the listed documents from S3, and extracts text using the docling library.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `documents_descriptor` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact containing documents_descriptor.json with bucket, prefix, and documents list. |
| `extracted_text` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact where the extracted text content will be stored. |
| `error_tolerance` | `Optional[float]` | `None` | Fraction of documents (0.0–1.0) allowed to fail without raising an error. None (the default) means zero tolerance — any failure raises immediately after all documents are processed. 0.1 means up to 10 % of documents may fail. Exceeding the threshold raises RuntimeError with a summary of up to 10 failing documents. |
| `max_extraction_workers` | `Optional[int]` | `None` | Number of parallel worker processes used for text extraction. Each worker loads a full docling DocumentConverter into memory (ONNX models, layout detection, etc.), so this should be kept low to avoid out-of-memory issues. Defaults to 4. Set to None to use all available CPU cores. Set to 1 to disable parallelism. |

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
- **Last Verified**: 2026-04-21 12:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
