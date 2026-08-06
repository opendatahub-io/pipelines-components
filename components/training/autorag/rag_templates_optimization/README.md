# Rag Templates Optimization ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

RAG Templates Optimization component.

Thin wrapper that delegates to ``ai4rag.components.optimization.rag_templates_optimization.run_rag_optimization``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `extracted_text` | `dsl.InputPath(dsl.Artifact)` | `None` | Path to extracted text documents. |
| `test_data` | `dsl.InputPath(dsl.Artifact)` | `None` | Path to benchmark test data JSON. |
| `search_space_prep_report` | `dsl.InputPath(dsl.Artifact)` | `None` | Path to the YAML search space report. |
| `rag_patterns` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact for generated RAG patterns. |
| `test_data_key` | `Optional[str]` | `None` | Path to benchmark JSON in object storage. |
| `vector_io_provider_id` | `str` | `None` | Vector I/O provider identifier in OGX. |
| `ogx_secret_name` | `str` | `None` | Name of the K8s secret with OGX credentials. |
| `input_data_secret_name` | `str` | `None` | Name of the K8s secret with S3 credentials for input data. |
| `input_data_bucket_name` | `str` | `None` | S3 bucket containing input documents. |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output HTML artifact; the leaderboard table is written to html_artifact.path (single file). |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `optimization_settings` | `Optional[dict]` | `None` | Additional experiment settings. |
| `input_data_key` | `Optional[str]` | `""` | Path to documents dir within bucket. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) uses 10 benchmark query threads. "balanced" uses 4 threads (reduced due to larger per-request context). |

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
    ogx_secret_name: str = "ogx-connection",
    input_data_secret_name: str = "s3-input-connection",
    input_data_bucket_name: str = "my-bucket",
    input_data_key: str = "",
):
    """Example pipeline using rag_templates_optimization.

    Args:
        test_data_key: Key for the test data.
        vector_io_provider_id: Vector I/O provider identifier.
        ogx_secret_name: Name of the K8s secret with OGX credentials.
        input_data_secret_name: Name of the K8s secret with S3 credentials.
        input_data_bucket_name: S3 bucket containing input documents.
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
        ogx_secret_name=ogx_secret_name,
        input_data_secret_name=input_data_secret_name,
        input_data_bucket_name=input_data_bucket_name,
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
    - Name: ai4rag, Version: ~=0.11.0
    - Name: OGX API, Version: ~=1.1.0
    - Name: Milvus, Version: >=2.0.0
    - Name: Milvus Lite, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - optimization
  - rag-patterns
- **Last Verified**: 2026-07-20 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko
