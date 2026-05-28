# Documents Indexing ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Index extracted text into a vector store with optional batch processing.

Reads markdown files from extracted_text, chunks them, embeds via OGX, and adds them to the vector store. When batch_size > 0, processes documents in batches to limit memory use and allow progress on large inputs.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `embedding_model_id` | `str` | `None` | Embedding model ID used for the vector store. |
| `extracted_text` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact (folder) containing .md files from text extraction. |
| `vector_io_provider_id` | `str` | `None` | OGX provider ID for the vector database. |
| `embedding_params` | `Optional[dict]` | `None` | Optional embedding parameters. |
| `distance_metric` | `str` | `cosine` | Vector distance metric (e.g. "cosine"). |
| `chunking_method` | `str` | `recursive` | Chunking method. |
| `chunk_size` | `int` | `1024` | Chunk size in characters. |
| `chunk_overlap` | `int` | `0` | Chunk overlap in characters. |
| `batch_size` | `int` | `20` | Number of documents per batch; 0 means process all in one batch. |
| `collection_name` | `Optional[str]` | `None` | Optional name of the collection to reuse; omit to create a new one. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of documents_indexing."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.documents_indexing import documents_indexing


@dsl.pipeline(name="documents-indexing-example")
def example_pipeline(
    embedding_model_id: str = "all-MiniLM-L6-v2",
    vector_io_provider_id: str = "milvus",
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
):
    """Example pipeline using documents_indexing.

    Args:
        embedding_model_id: ID of the embedding model.
        vector_io_provider_id: OGX provider ID for the vector database.
        distance_metric: Distance metric for similarity search.
        chunking_method: Method for text chunking.
        chunk_size: Size of each text chunk.
        chunk_overlap: Overlap between chunks.
        batch_size: Number of documents per batch.
    """
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    documents_indexing(
        embedding_model_id=embedding_model_id,
        extracted_text=extracted_text.output,
        vector_io_provider_id=vector_io_provider_id,
        distance_metric=distance_metric,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
    )

```

## Metadata 🗂️

- **Name**: documents_indexing
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: ai4rag, Version: ~=0.6.1
- **Tags**:
  - data-indexing
  - autorag
- **Last Verified**: 2026-05-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
