# Documents Indexing ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Chunk, embed, and index extracted documents into a vector store.

Reads DoclingDocument JSON files from the *extracted_text* artifact, splits them into chunks, computes embeddings via MaaS, and inserts the resulting vectors into the configured vector store. Documents are processed in batches to bound memory consumption.

The vector store backend (Milvus or PGVector) is resolved at runtime from the environment injected by the vector-database secret: ``MILVUS_*`` keys select Milvus, ``PGVECTOR_*`` keys select PGVector.

Individual document failures (corrupt JSON, chunking errors) are recorded in the indexing report and skipped — they do not abort the pipeline. Systemic failures (MaaS API unreachable, vector database unreachable, embedding model errors) propagate normally.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `embedding_model_id` | `str` | `None` | Embedding model ID served by MaaS. |
| `extracted_text` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact (directory) containing DoclingDocument JSON files from text extraction. |
| `indexing_report` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing ``indexing_report.json`` with per-document indexing status and pipeline settings. |
| `indexing_report_html` | `dsl.Output[dsl.HTML]` | `None` | Output HTML artifact containing a styled rendering of the indexing results (summary stats, settings, per-document table). |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded HTML report template injected by KFP at runtime from ``indexing_report_template.html``. |
| `embedding_params` | `Optional[dict]` | `None` | Optional parameters forwarded to :class:`OpenAIEmbeddingParams` (e.g. ``embedding_dimension``, ``context_length``). |
| `chunking_method` | `str` | `recursive` | Chunking strategy: ``"recursive"`` (LangChain) or ``"hybrid"`` (Docling structure-aware). |
| `chunk_size` | `int` | `1024` | Maximum chunk size in tokens (128--2048). |
| `chunk_overlap` | `int` | `0` | Token overlap between consecutive chunks (recursive method only). |
| `batch_size` | `int` | `20` | Number of documents loaded and processed per batch. Controls peak memory usage, not API payload sizes. Defaults to ``20``; ``0`` processes all documents in a single batch. |
| `collection_name` | `Optional[str]` | `None` | Vector store collection to reuse (matches ``pattern.json`` ``settings.vector_store_binding.collection_name``). Omit to create a new collection. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of documents_indexing."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.documents_indexing import documents_indexing


@dsl.pipeline(name="documents-indexing-example")
def example_pipeline(
    embedding_model_id: str = "all-MiniLM-L6-v2",
    collection_name: str = "",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
):
    """Example pipeline using documents_indexing.

    Args:
        embedding_model_id: ID of the embedding model.
        collection_name: Vector store collection to reuse (empty creates a new one).
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
        collection_name=collection_name,
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
    - Name: ai4rag, Version: ~=0.12.0
    - Name: MaaS, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0
    - Name: PGVector, Version: >=0.5.0
- **Tags**:
  - data-indexing
  - autorag
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
