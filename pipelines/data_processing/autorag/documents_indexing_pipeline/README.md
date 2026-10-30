# Documents Indexing Pipeline ✨

## Overview 🧾

Build a production vector index from documents for AutoRAG.

Discovers documents from object storage, extracts text, and indexes chunks into a vector database. Intended for post-optimization deploy when applying an optimized RAG pattern's indexing settings to a production corpus.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `maas_secret_name` | `str` | `None` | Name of the secret with MaaS inference credentials ("MAAS_BASE_URL", "MAAS_API_KEY"). |
| `db_secret_name` | `str` | `None` | Name of the secret carrying the database configuration. The env-var prefix selects the backend: ``MILVUS_*`` keys (at least ``MILVUS_URI``) select Milvus, ``PGVECTOR_*`` keys select PGVector, and ``NEO4J_*`` keys select Neo4j. ``NEO4J_CA_CERT`` optionally supplies a private CA certificate for Neo4j TLS. |
| `embedding_model_id` | `str` | `None` | Embedding model ID served by MaaS. |
| `input_data_secret_name` | `str` | `None` | Name of the secret with S3 credentials for input data ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION"). |
| `input_data_bucket_name` | `str` | `None` | Name of the S3 bucket containing input data. |
| `input_data_keys` | `Optional[list[str]]` | `None` | Paths to folders with input documents within bucket, 1-10 of them. Every one is discovered and merged into a single corpus deduplicated by object key; leave empty to use the whole bucket. More than 10 fails the run. |
| `collection_name` | `Optional[str]` | `None` | Vector store collection to reuse (aligned with ``pattern.json`` ``settings.store_binding.collection_name``). Omit to create a new collection. |
| `embedding_params` | `Optional[dict]` | `None` | Dict passed to OpenAIEmbeddingParams (default: {}). |
| `foundation_model_id` | `Optional[str]` | `None` | Generation model used to rebuild Neo4j graph entities. It is supplied by an optimized graph-mode pattern; ignored by non-Neo4j stores. |
| `foundation_model_params` | `Optional[dict]` | `None` | Generation parameters (for example, ``temperature`` and ``max_completion_tokens``) for Neo4j graph extraction. Supplied by an optimized graph-mode pattern; ignored by non-Neo4j stores. |
| `chunking_method` | `str` | `recursive` | Chunking method (e.g. "recursive"). |
| `chunk_size` | `int` | `1024` | Maximum chunk size in tokens (128--1024). |
| `chunk_overlap` | `int` | `0` | Token overlap between consecutive chunks (recursive method only). |
| `batch_size` | `int` | `20` | Number of documents per batch. Defaults to ``20``; ``0`` processes all documents in a single batch. |
| `ocr_lang` | `Optional[str]` | `None` | Language used to pick the RapidOCR model bundle. Pass ``pattern.json`` ``settings.generation.language.code`` from the optimization run so indexing OCRs the corpus the same way the experiment did. Note that AutoRAG derives that code from the benchmark questions, not from the documents themselves, so override it when the corpus is in a different language. Chinese selects the Chinese bundle; omitting it selects the English bundle, which covers all Latin-script languages. |
| `graph_extraction_config` | `Optional[dict]` | `None` | Neo4j graph-extraction settings from an optimized pattern. Ignored by other vector stores. |
| `preset` | `str` | `speed` | Text extraction quality tier from the optimization run. "speed" (default) disables table structure parsing; "balanced" enables TableFormer table reconstruction. |

## Metadata 🗂️

- **Name**: documents-indexing-pipeline
- **Stability**: stable
- **Managed**: Yes
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: boto3, Version: >=1.42.34
    - Name: ai4rag, Version: ~=0.25.3
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
- **Last Verified**: 2026-10-29 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - Mateusz-Switala
    - jakub-walaszczyk
  - Reviewers:
    - LukaszCmielowski
    - Wojciech-Rebisz
    - Mateusz-Switala
