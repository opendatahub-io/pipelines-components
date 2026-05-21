# Autorag Documents Indexing ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Defines a pipeline to load, sample, extract text, and index documents for AutoRAG.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `ogx_secret_name` | `str` | `None` | Name of the secret with OGX credentials ("OGX_CLIENT_BASE_URL", "OGX_CLIENT_API_KEY"). |
| `embedding_model_id` | `str` | `None` | Embedding model ID for the vector store. |
| `vector_io_provider_id` | `str` | `None` | Optional OGX provider ID. |
| `input_data_secret_name` | `str` | `None` | Name of the secret with S3 credentials for input data (required when input_data_source="s3"). Must contain: "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION". |
| `input_data_bucket_name` | `str` | `None` | S3 bucket containing input data (required when input_data_source="s3"). |
| `input_data_key` | `Optional[str]` | `None` | S3 prefix / directory path within bucket (required when input_data_source="s3"). |
| `input_data_source` | `str` | `s3` | Data source type ("s3" or "pvc"). Default is "s3". |
| `pvc_input_data_path` | `str` | `""` | Directory path on PVC relative to pvc_mount_path containing documents (required when input_data_source="pvc"). Example: "documents/" will be accessed at {pvc_mount_path}/documents/. |
| `pvc_name` | `str` | `""` | Name of existing PVC to mount (required when input_data_source="pvc"). This PVC must already exist in the same namespace as the pipeline. |
| `pvc_mount_path` | `str` | `/mnt/data` | Path where the PVC will be mounted in the container (default: "/mnt/data"). The pvc_input_data_path will be relative to this mount path. |
| `collection_name` | `str` | `None` | Optional name of the collection to reuse; omit to create a new one. |
| `embedding_params` | `Optional[dict]` | `None` | Dict passed to OGXEmbeddingParams (default: {}). |
| `distance_metric` | `str` | `cosine` | Vector distance metric (e.g. "cosine"). |
| `chunking_method` | `str` | `recursive` | Chunking method (e.g. "recursive"). |
| `chunk_size` | `int` | `1024` | Chunk size in characters. |
| `chunk_overlap` | `int` | `0` | Chunk overlap in characters. |
| `batch_size` | `int` | `20` | Number of documents per batch (0 = process all at once). |

## Metadata 🗂️

- **Name**: autorag-documents-indexing
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: docling, Version: >=2.72.0
    - Name: boto3, Version: >=1.42.34
    - Name: ai4rag, Version: ~=0.6.1
    - Name: RHOAI Connections API, Version: >=1.0.0
- **Tags**:
  - data_processing
  - text_extraction
  - documents_discovery
  - data_indexing
  - autorag
- **Last Verified**: 2026-05-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - LukaszCmielowski
    - DorotaDR
