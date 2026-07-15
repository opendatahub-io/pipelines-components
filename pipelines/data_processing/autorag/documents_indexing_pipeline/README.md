# Documents Indexing Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Build a production vector index from documents for AutoRAG.

Discovers documents from object storage, extracts text, and indexes chunks into OGX. Intended for post-optimization deploy when applying an optimized RAG pattern's indexing settings to a production corpus.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `ogx_secret_name` | `str` | `None` | Name of the secret with OGX credentials ("OGX_CLIENT_BASE_URL", "OGX_CLIENT_API_KEY"). |
| `embedding_model_id` | `str` | `None` | Embedding model ID for the vector store. |
| `vector_io_provider_id` | `str` | `None` | Optional OGX provider ID. |
| `input_data_secret_name` | `str` | `None` | Name of the secret with S3 credentials for input data ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION"). |
| `input_data_bucket_name` | `str` | `None` | Name of the S3 bucket containing input data. |
| `input_data_key` | `Optional[str]` | `None` | Path to folder with input documents within bucket. |
| `vector_store_id` | `str` | `None` | OGX vector store / collection id to reuse (aligned with ``pattern.json`` ``settings.vector_store_binding.vector_store_id``). Omit to create a new collection. |
| `embedding_params` | `Optional[dict]` | `None` | Dict passed to OGXEmbeddingParams (default: {}). |
| `distance_metric` | `str` | `cosine` | Vector distance metric (e.g. "cosine"). |
| `chunking_method` | `str` | `recursive` | Chunking method (e.g. "recursive"). |
| `chunk_size` | `int` | `1024` | Chunk size in characters. |
| `chunk_overlap` | `int` | `0` | Chunk overlap in characters. |
| `batch_size` | `int` | `20` | Number of documents per batch. Defaults to ``20``; ``0`` processes all documents in a single batch. |

## Metadata 🗂️

- **Name**: documents-indexing-pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: docling, Version: >=2.72.0
    - Name: boto3, Version: >=1.42.34
    - Name: ai4rag, Version: ~=0.10.0
    - Name: RHOAI Connections API, Version: >=1.0.0
- **Tags**:
  - data_processing
  - text_extraction
  - documents_discovery
  - data_indexing
  - autorag
- **Last Verified**: 2026-05-14 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - LukaszCmielowski
    - DorotaDR
