# Rag Templates Optimization ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

RAG Templates Optimization component.

Carries out the iterative RAG optimization process.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `extracted_text` | `dsl.InputPath(dsl.Artifact)` | `None` | A path pointing to a folder containg extracted texts from input documents. |
| `test_data` | `dsl.InputPath(dsl.Artifact)` | `None` | A path pointing to test data used for evaluating RAG pattern quality. |
| `search_space_prep_report` | `dsl.InputPath(dsl.Artifact)` | `None` | A path pointing to a .yml file containig short report on the experiment's first phase (search space preparation). |
| `rag_patterns` | `dsl.Output[dsl.Artifact]` | `None` | kfp-enforced argument specifying an output artifact. Provided by kfp backend automatically. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | kfp-enforced argument to allow access of base64 encoded dir with notebook templates. |
| `test_data_key` | `Optional[str]` | `None` | Path to the benchmark JSON file in object storage used by generated notebooks. |
| `vector_io_provider_id` | `str` | `None` | Vector I/O provider identifier as registered in OGX. |
| `optimization_settings` | `Optional[dict]` | `None` | Additional settings customising the experiment. |
| `input_data_key` | `Optional[str]` | `""` | A path to documents dir within a bucket used as an input to AI4RAG experiment. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of rag_templates_optimization."""

from kfp import dsl
from kfp_components.components.training.autorag.rag_templates_optimization import (
    rag_templates_optimization,
)


@dsl.pipeline(name="rag-templates-optimization-example")
def example_pipeline(
    test_data_key: str = "questions",
    vector_io_provider_id: str = "milvus",
    input_data_key: str = "",
):
    """Example pipeline using rag_templates_optimization.

    Args:
        test_data_key: Key for the test data.
        vector_io_provider_id: Vector I/O provider identifier.
        input_data_key: Key for the input data.
    """
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    search_space_prep_report = dsl.importer(
        artifact_uri="gs://placeholder/search_space_prep_report",
        artifact_class=dsl.Artifact,
    )
    rag_templates_optimization(
        extracted_text=extracted_text.output,
        test_data=test_data.output,
        search_space_prep_report=search_space_prep_report.output,
        test_data_key=test_data_key,
        vector_io_provider_id=vector_io_provider_id,
        input_data_key=input_data_key,
    )

```

## Metadata 🗂️

- **Name**: rag_templates_optimization
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: ~=0.6.1
    - Name: OGX API, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0
    - Name: Milvus Lite, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - optimization
  - rag-patterns
- **Last Verified**: 2026-05-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
