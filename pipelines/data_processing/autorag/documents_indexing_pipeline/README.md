# Documents Indexing Pipeline ✨

> ⚠️ **Stability: beta** — This asset is not yet stable and may change.

## Overview 🧾

Build a production vector index from documents for AutoRAG.

Discovers documents from object storage, extracts text, and indexes chunks into a vector database. Intended for post-optimization deploy when applying an optimized RAG pattern's indexing settings to a production corpus.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `maas_secret_name` | `str` | `None` | Name of the secret with MaaS inference credentials ("MAAS_BASE_URL", "MAAS_API_KEY"). |
| `vector_db_secret_name` | `str` | `None` | Name of the secret carrying the vector database configuration. The env-var prefix selects the backend: ``MILVUS_*`` keys (at least ``MILVUS_URI``) select Milvus, ``PGVECTOR_*`` keys select PGVector. |
| `embedding_model_id` | `str` | `None` | Embedding model ID served by MaaS. |
| `input_data_secret_name` | `str` | `None` | Name of the secret with S3 credentials for input data ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION"). |
| `input_data_bucket_name` | `str` | `None` | Name of the S3 bucket containing input data. |
| `input_data_key` | `Optional[str]` | `None` | Path to folder with input documents within bucket. |
| `collection_name` | `Optional[str]` | `None` | Vector store collection to reuse (aligned with ``pattern.json`` ``settings.vector_store_binding.collection_name``). Omit to create a new collection. |
| `embedding_params` | `Optional[dict]` | `None` | Dict passed to OpenAIEmbeddingParams (default: {}). |
| `chunking_method` | `str` | `recursive` | Chunking method (e.g. "recursive"). |
| `chunk_size` | `int` | `1024` | Maximum chunk size in tokens (128--2048). |
| `chunk_overlap` | `int` | `0` | Token overlap between consecutive chunks (recursive method only). |
| `batch_size` | `int` | `20` | Number of documents per batch. Defaults to ``20``; ``0`` processes all documents in a single batch. |

## Metadata 🗂️

- **Name**: documents-indexing-pipeline
- **Stability**: beta
- **Managed**: Yes
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: boto3, Version: >=1.42.34
    - Name: ai4rag, Version: ~=0.12.0
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: MaaS, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0
    - Name: PGVector, Version: >=0.5.0
- **Tags**:
  - data_processing
  - text_extraction
  - documents_discovery
  - data_indexing
  - autorag
- **Last Verified**: 2026-08-24 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - LukaszCmielowski
    - DorotaDR
