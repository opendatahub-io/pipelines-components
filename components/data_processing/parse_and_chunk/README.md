# Parse And Chunk ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Parse PDFs and write chunked JSONL files to S3.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `pvc_name` | `str` | `None` | Name of the PVC with input PDFs. |
| `pvc_mount_path` | `str` | `None` | Mount path for the PVC inside pods. |
| `input_path` | `str` | `None` | Relative path to input PDFs under the PVC. |
| `ray_image` | `str` | `None` | Container image with Ray + Docling pre-installed. |
| `namespace` | `str` | `None` | OpenShift namespace for the RayJob. |
| `s3_endpoint` | `str` | `None` | S3-compatible endpoint URL (e.g. MinIO). |
| `s3_bucket` | `str` | `None` | S3 bucket for output chunks. |
| `s3_prefix` | `str` | `chunks` | Key prefix for chunk files in S3. |
| `s3_secret_name` | `str` | `minio-secret` | Kubernetes Secret with S3 credentials (keys: access_key, secret_key). |
| `tokenizer` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | Tokenizer for HybridChunker (usually the embedding model name). |
| `chunk_max_tokens` | `int` | `256` | Max tokens per chunk (HybridChunker). |
| `num_workers` | `int` | `2` | Number of Ray worker pods. |
| `worker_cpus` | `int` | `8` | CPUs per worker pod. |
| `worker_memory_gb` | `int` | `16` | Memory (GB) per worker pod. |
| `head_cpus` | `int` | `2` | CPUs for the Ray head pod. |
| `head_memory_gb` | `int` | `8` | Memory (GB) for the Ray head pod. |
| `cpus_per_actor` | `int` | `4` | CPUs per Docling processing actor. |
| `min_actors` | `int` | `2` | Minimum actor pool size. |
| `max_actors` | `int` | `4` | Maximum actor pool size. |
| `batch_size` | `int` | `4` | Files per batch sent to each actor. |
| `num_files` | `int` | `0` | Number of PDFs to process (0 = all). |
| `timeout_seconds` | `int` | `600` | Per-file processing timeout. |
| `enable_profiling` | `bool` | `False` | Enable cProfile profiling (outputs profile stats). |
| `verbose` | `bool` | `True` | Enable verbose logging for debugging. |
| `bypass_kueue` | `bool` | `False` | If True, remove the Kueue queue-name label and manually unsuspend the RayJob, bypassing cluster quota management. Only use on clusters without Kueue or with sufficient free resources. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The S3 URI where JSONL chunk files were written. |

## Metadata 🗂️

- **Name**: parse_and_chunk
- **Description**: Parse PDFs with Docling and chunk with HybridChunker using a Ray Data actor pool. Writes JSONL chunk files to S3-compatible bucket (MinIO).

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KubeRay Operator, Version: >=1.4.0
- **Tags**:
  - data_processing
  - pdf
  - docling
  - ray
  - chunking
  - rag
- **Last Verified**: 2026-04-28 00:00:00+00:00
- **Owners**:
  - Approvers:
    - szaher
  - Reviewers:
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)
