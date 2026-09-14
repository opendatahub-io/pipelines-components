# Rag Templates Optimization ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

RAG Templates Optimization component.

Runs search-space construction, evaluator setup, and the optimization experiment directly against ``ai4rag`` primitives (``AI4RAGSearchSpace``, ``AI4RAGExperiment``, and the ``ragas``/``unitxt`` evaluators), since ``ai4rag`` no longer ships a single orchestration entry point for this flow.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `extracted_text` | `dsl.InputPath(dsl.Artifact)` | `None` | Path to extracted text documents. |
| `test_data` | `dsl.InputPath(dsl.Artifact)` | `None` | Path to benchmark test data JSON. |
| `search_space_mps_report` | `dsl.InputPath(dsl.Artifact)` | `None` | Path to the JSON search space report. |
| `rag_patterns` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact for generated RAG patterns. |
| `test_data_key` | `str` | `None` | Path to benchmark JSON in object storage. |
| `maas_secret_name` | `str` | `None` | Name of the K8s secret with MaaS inference credentials ("MAAS_BASE_URL", "MAAS_API_KEY"). Propagated into each generated ``pattern.json`` indexing spec for downstream deployment. |
| `vector_db_secret_name` | `str` | `None` | Name of the K8s secret holding the vector database configuration. Its keys select the backend: ``MILVUS_*`` keys use Milvus, ``PGVECTOR_*`` keys use PGVector. Propagated into each generated ``pattern.json`` indexing spec. |
| `input_data_secret_name` | `str` | `None` | Name of the K8s secret with S3 credentials for input data. |
| `input_data_bucket_name` | `str` | `None` | S3 bucket containing input documents. |
| `leaderboard` | `dsl.Output[dsl.HTML]` | `None` | Output HTML artifact; the leaderboard table is written to leaderboard_html.path (single file). |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `optimization_settings` | `Optional[dict]` | `None` | Additional experiment settings. |
| `input_data_keys` | `Optional[list[str]]` | `None` | Paths to documents dirs within bucket. Only the first entry is used for the generated indexing notebook; the full list is propagated to the indexing pipeline blueprint. |
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
    maas_secret_name: str = "maas-connection",
    vector_db_secret_name: str = "vector-db-connection",
    input_data_secret_name: str = "s3-input-connection",
    input_data_bucket_name: str = "my-bucket",
    input_data_keys: list[str] = [],
):
    """Example pipeline using rag_templates_optimization.

    Args:
        test_data_key: Key for the test data.
        maas_secret_name: Name of the K8s secret with MaaS inference credentials.
        vector_db_secret_name: Name of the K8s secret with the vector database
            configuration (MILVUS_* selects Milvus, PGVECTOR_* selects PGVector).
        input_data_secret_name: Name of the K8s secret with S3 credentials.
        input_data_bucket_name: S3 bucket containing input documents.
        input_data_keys: Keys for the input data; only the first one is used for discovery.
    """
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    search_space_mps_report = dsl.importer(
        artifact_uri="gs://placeholder/search_space_mps_report",
        artifact_class=dsl.Artifact,
    )
    rag_templates_optimization(
        extracted_text=extracted_text.output,
        test_data=test_data.output,
        search_space_mps_report=search_space_mps_report.output,
        test_data_key=test_data_key,
        maas_secret_name=maas_secret_name,
        vector_db_secret_name=vector_db_secret_name,
        input_data_secret_name=input_data_secret_name,
        input_data_bucket_name=input_data_bucket_name,
        input_data_keys=input_data_keys,
    )

```

## Metadata 🗂️

- **Name**: rag_templates_optimization
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: ~=0.16.0
    - Name: MaaS, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0
    - Name: PGVector, Version: >=0.5.0
- **Tags**:
  - training
  - autorag
  - optimization
  - rag-patterns
- **Last Verified**: 2026-09-08 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
    - Mateusz-Switala
    - filip-komarzyniec
    - jakub-walaszczyk
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko
