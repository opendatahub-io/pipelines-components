# Documents Rag Optimization Pipeline ✨

> ⚠️ **Stability: beta** — This asset is not yet stable and may change.

## Overview 🧾

Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications.

The Documents RAG Optimization Pipeline is an automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications within Red Hat OpenShift AI. It leverages Kubeflow Pipelines to orchestrate the optimization workflow, using the ai4rag optimization engine to systematically
explore RAG configurations and identify the best performing parameter settings based on an upfront-specified quality metric.

The system integrates with OGX API for inference and vector database operations, producing optimized RAG patterns as artifacts that can be deployed and used for production RAG applications. Each optimized pattern contains a ``pattern.json`` with deployment settings (including
``settings.responses_template`` for OGX ``/v1/responses``), executable notebooks, and evaluation results.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `test_data_secret_name` | `str` | `None` | Name of the Kubernetes secret holding S3-compatible credentials for test data access. The following environment variables are required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT. AWS_DEFAULT_REGION is optional. |
| `test_data_bucket_name` | `str` | `None` | S3 (or compatible) bucket name for the test data file. |
| `test_data_key` | `str` | `None` | Object key (path) of the test data JSON file in the test data bucket. |
| `input_data_secret_name` | `str` | `None` | Name of the Kubernetes secret holding S3-compatible credentials for input document data access. The following environment variables are required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT. AWS_DEFAULT_REGION is optional. |
| `input_data_bucket_name` | `str` | `None` | S3 (or compatible) bucket name for the input documents. |
| `ogx_secret_name` | `str` | `None` | Name of the Kubernetes secret for OGX API connection. The secret must define: OGX_CLIENT_API_KEY, OGX_CLIENT_BASE_URL. |
| `vector_io_provider_id` | `str` | `None` | Vector I/O provider id (e.g., registered in OGX Milvus). |
| `input_data_key` | `str` | `""` | Object key (path) of the input documents in the input data bucket. |
| `embedding_models` | `Optional[List]` | `None` | Optional list of embedding model identifiers to use in the search space. |
| `generation_models` | `Optional[List]` | `None` | Optional list of foundation/generation model identifiers to use in the search space. |
| `optimization_metric` | `str` | `overall_score` | Quality metric used to rank RAG patterns. Supported values: "faithfulness", "answer_correctness", "context_correctness", "answer_relevance", and "overall_score" (default). "faithfulness", "answer_correctness", and "context_correctness" are deterministic Unitxt metrics; choosing one as the optimization metric keeps the experiment deterministic. The LLM-judge metric "answer_relevance" is always computed but only drives optimization when selected (or via "overall_score", which aggregates all metrics). |
| `optimization_max_rag_patterns` | `int` | `8` | Maximum number of RAG patterns to generate. Passed to ai4rag (max_number_of_rag_patterns). Defaults to 8. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) uses recursive chunking, no table structure parsing, and no contextual enrichment. "balanced" enables Docling table layout parsing, hybrid chunking, and LLM contextual enrichment. Both presets use the same resource tier. |

## Metadata 🗂️

- **Name**: documents-rag-optimization-pipeline
- **Stability**: beta
- **Managed**: Yes
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: 2.16.1
  - External Services:
    - Name: ai4rag, Version: ~=0.9.1
    - Name: OGX API, Version: ~=1.1.0
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: Milvus, Version: >=2.0.0
    - Name: Milvus Lite, Version: >=2.0.0
    - Name: MLFlow, Version: >=2.0.0
    - Name: docling, Version: >=1.0.0
- **Tags**:
  - training
  - pipeline
  - autorag
  - rag-optimization
- **Last Verified**: 2026-07-10 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko

<!-- custom-content -->

### Progress and dashboard artifacts

Besides RAG pattern and data artifacts below, each run publishes:

| KFP task | Output | File | Purpose |
| -------- | ------ | ---- | ------- |
| `publish-component-stage-map` | `component_stage_map` | `component_stage_map.json` | Static component-to-stage-to-step catalog for the RAG pipeline (published once at run start). |
| `test-data-loader` | `component_status` | `component_status.json` | Stage progress for benchmark test data download and sampling. |
| `documents-discovery` | `component_status` | `component_status.json` | Stage progress for listing and sampling source documents. |
| `text-extraction` | `component_status` | `component_status.json` | Stage progress for docling text extraction. |
| `search-space-preparation` | `component_status` | `component_status.json` | Stage progress for search-space preparation and model pre-selection. |
| `rag-templates-optimization` | `component_status` | `component_status.json` | Stage progress for RAG template optimization (including sub-steps). |
| `rag-templates-optimization` | `html_artifact` | `*.html` | Leaderboard HTML comparing optimized RAG patterns. |

Example artifact-store layout (task folder names are kebab-case):

```text
<pipeline_name>/<run_id>/
├── publish-component-stage-map/<task_id>/component_stage_map/component_stage_map.json
├── test-data-loader/<task_id>/component_status/component_status.json
├── documents-discovery/<task_id>/component_status/component_status.json
├── text-extraction/<task_id>/component_status/component_status.json
├── search-space-preparation/<task_id>/component_status/component_status.json
├── rag-templates-optimization/<task_id>/component_status/component_status.json
└── rag-templates-optimization/<task_id>/html_artifact/<leaderboard>.html
```

See [AutoRAG training components README](../../../components/training/autorag/README.md) for JSON field details.

#### Dashboard join keys

Dashboards join the static map (`component_stage_map.json`) to live progress (`component_status.json`) using **snake_case component ids**, not KFP task names:

| Layer | Naming | Test data loader example |
| ----- | ------ | ------------------------ |
| Template `components[].id` | snake_case | `test_data_loader` |
| Runtime `component_status.json` → `component_id` | snake_case | `test_data_loader` |
| KFP root DAG task id (compiled YAML) | kebab-case | `test-data-loader` |
| KFP output parameter | snake_case | `component_status` |
| Artifact file | snake_case | `component_status.json` |

Use `component_id` (and stage `id` fields inside each file) to correlate artifacts. KFP task names are only for locating artifact paths in the store.

Canonical component ids are defined in the pipeline JSON templates under
[`run_status_templates/pipelines/`](../../../components/training/autorag/shared/run_status_templates/pipelines/)
(e.g. `documents-rag-optimization-pipeline.json`).

## Breaking Changes (ai4rag 0.8.0)

This version introduces breaking changes to the pipeline's artifact format and
task topology. Existing pipeline runs cached against the previous version are
incompatible and must be re-run.

| Change | Before | After | Action required |
| ------ | ------ | ----- | --------------- |
| Extracted text format | Flat text / markdown files | DoclingDocument JSON files | Invalidate KFP step caches; re-run text extraction. |
| Leaderboard task | Separate `leaderboard-evaluation` task | Built-in stage of `rag-templates-optimization` | Update dashboard artifact path expectations. |
| Responses API payload | Separate `v1_responses_body.json` via `prepare_responses_api_requests` | `settings.responses_template` inside `pattern.json` | Update downstream consumers to read from `pattern.json`. |

## Optimization Engine: ai4rag 🚀

The pipeline uses [ai4rag](https://github.com/IBM/ai4rag), a RAG Templates Optimization Engine that
provides an automated approach to optimizing Retrieval-Augmented Generation (RAG) systems. The
engine is designed to be LLM and Vector Database provider agnostic, making it flexible and
adaptable to various RAG implementations.

ai4rag accepts a variety of RAG templates and search space definitions, then systematically
explores different parameter configurations to find optimal settings. The engine returns initialized
RAG templates with optimal parameter values, which are referred to as RAG Patterns.

## Supported Features ✨

**Status**: Tech Preview - MVP (May 2026)

### RAG Configuration

- **RAG Type**: Documents (documents provided as input)
- **Supported Languages**: English
- **Supported Document Types**: PDF, DOCX, PPTX, Markdown, HTML, Plain text
- **Document Data Sources**: S3 (Amazon S3), Local filesystem (FS)

### Infrastructure Components

- **Vector Databases**: Milvus, Milvus Lite, ChromaDB
- **LLM Provider**: OGX-supported models and vendors
- **Experiment Tracking**: MLFlow (optional) - For experiment tracking, metrics logging, and
  artifact management

### Processing Methods

- **Chunking Method**: Recursive
- **Retrieval Methods**: Simple, Simple with hybrid ranker

### Interfaces

- **API**: Programmatic access to AutoRAG functionality
- **UI**: User interface for interacting with AutoRAG

## Glossary 📚

### RAG Configuration Definition

A **RAG Configuration** is a specific set of parameter values that define how a
Retrieval-Augmented Generation system operates. It includes settings for:

- **Chunking**: Method and parameters for splitting documents (e.g., recursive method with
  chunk_size=2048, chunk_overlap=256)
- **Embeddings**: The embedding model used (e.g., `intfloat/multilingual-e5-large`)
- **Generation**: The language model used (e.g., `ibm/granite-13b-instruct-v2`) along with its
  parameters
- **Retrieval**: The method for retrieving relevant document chunks (e.g., simple retrieval or
  hybrid ranker)

### RAG Pattern

A **RAG Pattern** is an optimized RAG configuration that has been evaluated and ranked by
AutoRAG. It represents a complete, deployable RAG system with:

- Validated parameter settings that have been tested and evaluated
- Performance metrics (e.g., answer_correctness, faithfulness, context_correctness)
- Executable notebooks for indexing and inference operations
- A position in the leaderboard based on performance

### RAG Template

A **RAG Template** is a reusable blueprint that defines the structure and workflow of a RAG
system. Templates are parameterized and AutoRAG uses templates as the foundation, optimizing the
parameter values to create RAG Patterns.

## Additional Resources 📚

- **ai4rag Documentation**: [ai4rag GitHub](https://github.com/IBM/ai4rag)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
