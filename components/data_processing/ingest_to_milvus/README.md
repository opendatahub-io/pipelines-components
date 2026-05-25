# Ingest To Milvus ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Read chunks from S3, embed, and insert into Milvus.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `s3_endpoint` | `str` | `None` | S3-compatible endpoint URL (e.g. MinIO). |
| `s3_bucket` | `str` | `None` | S3 bucket containing chunk files. |
| `milvus_host` | `str` | `None` | Milvus service hostname. |
| `s3_prefix` | `str` | `chunks` | Key prefix for chunk files in S3. |
| `embedding_endpoint` | `str` | `""` | Optional embedding service URL. If empty, uses a local sentence-transformers model. |
| `embedding_model` | `str` | `ibm-granite/granite-embedding-125m-english` | Embedding model name (for API or local). |
| `embedding_dim` | `int` | `768` | Dimension of the embedding vectors. |
| `milvus_port` | `int` | `19530` | Milvus gRPC port. |
| `milvus_db` | `str` | `default` | Milvus database name. |
| `milvus_token` | `str` | `""` | Milvus authentication token. Empty string for unauthenticated connections. |
| `collection_name` | `str` | `rag_documents` | Milvus collection name. |
| `drop_existing` | `bool` | `True` | If True, drop and recreate the collection. If False, append to it. |
| `embed_batch_size` | `int` | `64` | Batch size for embedding requests. |
| `milvus_batch_size` | `int` | `256` | Batch size for Milvus inserts. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The Milvus collection name and total vectors inserted. |

## Metadata 🗂️

- **Name**: ingest_to_milvus
- **Description**: Read chunked JSONL files, generate embeddings via an embedding service endpoint or local model, and insert into Milvus.

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Milvus, Version: >=2.4.0
- **Tags**:
  - milvus
  - embeddings
  - vector_db
  - ingestion
  - rag
- **Last Verified**: 2026-04-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - szaher
  - Reviewers:
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
