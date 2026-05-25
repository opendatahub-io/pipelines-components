# Pdf Documents Processing Rag ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Multi-step RAG pipeline: parse PDFs, ingest into Milvus, deploy LLM.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pvc_name` | `str` | `data-pvc` | PVC containing input PDF documents. |
| `pvc_mount_path` | `str` | `/mnt/data` | Mount path for the data PVC inside pods. |
| `namespace` | `str` | `ray-docling` | OpenShift namespace for all resources. |
| `s3_endpoint` | `str` | `http://minio-service.default.svc.cluster.local:9000` | S3-compatible endpoint URL (e.g. MinIO). |
| `s3_bucket` | `str` | `rag-chunks` | S3 bucket for intermediate chunk storage. |
| `s3_prefix` | `str` | `chunks` | Key prefix for chunk files in S3. |
| `s3_secret_name` | `str` | `minio-secret` | Kubernetes Secret with S3 credentials. |
| `input_path` | `str` | `input/pdfs` | Path to PDF files on the PVC. |
| `ray_image` | `str` | `quay.io/rhoai-szaher/docling-ray:latest` | Container image with Ray and Docling pre-installed. |
| `num_workers` | `int` | `2` | Number of Ray worker pods. |
| `worker_cpus` | `int` | `8` | CPUs per Ray worker pod. |
| `worker_memory_gb` | `int` | `16` | Memory (GB) per Ray worker pod. |
| `head_cpus` | `int` | `2` | CPUs for the Ray head pod. |
| `head_memory_gb` | `int` | `8` | Memory (GB) for the Ray head pod. |
| `cpus_per_actor` | `int` | `4` | CPUs per Docling processing actor. |
| `min_actors` | `int` | `2` | Minimum actor pool size. |
| `max_actors` | `int` | `4` | Maximum actor pool size. |
| `batch_size` | `int` | `4` | Files per batch sent to each actor. |
| `chunk_max_tokens` | `int` | `256` | Maximum tokens per chunk. |
| `num_files` | `int` | `1000` | Number of PDFs to process (0 = all). |
| `timeout_seconds` | `int` | `600` | Per-file processing timeout in seconds. |
| `enable_profiling` | `bool` | `False` | Enable cProfile profiling output. |
| `verbose` | `bool` | `True` | Enable verbose logging. |
| `bypass_kueue` | `bool` | `False` | If True, bypass Kueue quota management for the RayJob. |
| `deploy_embedding` | `bool` | `False` | If True, deploy embedding model as InferenceService. |
| `embedding_endpoint` | `str` | `""` | Embedding service URL (empty = local model). |
| `embedding_model` | `str` | `ibm-granite/granite-embedding-125m-english` | Embedding model name. |
| `embedding_dim` | `int` | `768` | Embedding vector dimension. |
| `embedding_runtime_image` | `str` | `_DEFAULT_EMBEDDING_RUNTIME_IMAGE` | Container image for the embedding server. |
| `embedding_serving_runtime_name` | `str` | `embedding-runtime` | Name of the embedding ServingRuntime CR. |
| `embedding_gpu_count` | `int` | `1` | GPUs for the embedding service. |
| `embedding_min_replicas` | `int` | `1` | Minimum replicas for the embedding service. |
| `embedding_max_replicas` | `int` | `1` | Maximum replicas for the embedding service. |
| `embedding_cpu_requests` | `str` | `2` | CPU requests for the embedding service. |
| `embedding_cpu_limits` | `str` | `4` | CPU limits for the embedding service. |
| `embedding_memory_requests` | `str` | `4Gi` | Memory requests for the embedding service. |
| `embedding_memory_limits` | `str` | `8Gi` | Memory limits for the embedding service. |
| `embedding_max_model_len` | `int` | `512` | Maximum sequence length for the embedding model. |
| `milvus_host` | `str` | `milvus-milvus.milvus.svc.cluster.local` | Milvus service hostname. |
| `milvus_port` | `int` | `19530` | Milvus gRPC port. |
| `milvus_db` | `str` | `default` | Milvus database name. |
| `milvus_token` | `str` | `""` | Milvus authentication token. Empty string for unauthenticated connections. |
| `collection_name` | `str` | `rag_documents` | Milvus collection name. |
| `drop_existing` | `bool` | `True` | If True, drop and recreate the Milvus collection. If False, append. |
| `embed_batch_size` | `int` | `64` | Batch size for embedding requests. |
| `milvus_batch_size` | `int` | `256` | Batch size for Milvus inserts. |
| `hf_secret_name` | `str` | `hf-token-secret` | Kubernetes Secret with HuggingFace token (key: 'token'). |
| `llm_model_name` | `str` | `mistralai/Mistral-7B-Instruct-v0.3` | HuggingFace LLM model ID for inference. |
| `model_cache_pvc` | `str` | `model-cache-pvc` | PVC for cached model weights. |
| `model_cache_mount` | `str` | `/mnt/models` | Mount path for the model cache PVC. |
| `max_model_len` | `int` | `4096` | Maximum context length for the LLM. |
| `gpu_count` | `int` | `1` | GPUs for LLM serving. |
| `llm_hardware_profile_name` | `str` | `gpu-profile` | Hardware profile name for LLM deployment. |
| `llm_hardware_profile_namespace` | `str` | `redhat-ods-applications` | Namespace of the hardware profile. |
| `llm_min_replicas` | `int` | `1` | Minimum replicas for the LLM service. |
| `llm_max_replicas` | `int` | `1` | Maximum replicas for the LLM service. |
| `llm_cpu_requests` | `str` | `2` | CPU requests for the LLM service. |
| `llm_cpu_limits` | `str` | `2` | CPU limits for the LLM service. |
| `llm_memory_requests` | `str` | `8Gi` | Memory requests for the LLM service. |
| `llm_memory_limits` | `str` | `8Gi` | Memory limits for the LLM service. |
| `llm_force_recreate` | `bool` | `False` | If True, delete and recreate the LLM InferenceService (causes downtime). If False (default), patch in place. |

## Metadata 🗂️

- **Name**: pdf_documents_processing_rag
- **Description**: Multi-step RAG pipeline: parse and chunk PDFs via Docling RayJob, ingest embeddings into Milvus, deploy LLM with vLLM on OpenShift AI. Requires secrets: hf-token-secret (key: token) and minio-secret (keys: access_key, secret_key).

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KubeRay Operator, Version: >=1.4.0
    - Name: Milvus, Version: >=2.4.0
    - Name: KServe, Version: >=0.11.0
- **Tags**:
  - data_processing
  - pdf
  - docling
  - ray
  - chunking
  - rag
  - milvus
- **Last Verified**: 2026-04-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - szaher
  - Reviewers:
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
