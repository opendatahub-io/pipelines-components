# Documents Discovery ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Documents discovery component with optional benchmark test data loading.

Discovers input documents in S3 and optionally downloads benchmark test data for document-prioritised sampling. When ``test_data_bucket_name`` is provided, the component first downloads and samples the benchmark JSON, then uses the referenced document IDs to prioritise discovery.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `input_data_bucket_name` | `str` | `None` | S3 (or compatible) bucket containing input documents. |
| `input_data_path` | `str` | `""` | Path to folder with input documents within the bucket. |
| `test_data_bucket_name` | `str` | `""` | S3 bucket containing the test data file.  Leave empty to skip test data loading (e.g. for the indexing pipeline). |
| `test_data_path_key` | `str` | `""` | S3 object key to the JSON test data file. |
| `benchmark_sample_size` | `int` | `25` | Maximum number of benchmark records to keep. When the dataset exceeds this limit, a reproducible random sample is drawn (seed 42). Set to 0 to disable sampling. |
| `sampling_enabled` | `bool` | `True` | Whether to enable document size-based sampling. |
| `sampling_max_size` | `float` | `1` | Maximum size of sampled documents (in gigabytes). |
| `discovered_documents` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the documents descriptor JSON file. |
| `test_data` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the (possibly sampled) test data JSON. Empty when test data loading is skipped. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of documents_discovery."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.documents_discovery import documents_discovery


@dsl.pipeline(name="documents-discovery-example")
def example_pipeline(
    input_data_bucket_name: str = "my-bucket",
    input_data_path: str = "documents/",
    sampling_enabled: bool = True,
    sampling_max_size: float = 1,
):
    """Example pipeline using documents_discovery.

    Args:
        input_data_bucket_name: S3 bucket containing input documents.
        input_data_path: Path prefix within the bucket.
        sampling_enabled: Whether to enable sampling.
        sampling_max_size: Maximum sample size in GB.
    """
    documents_discovery(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_path,
        sampling_enabled=sampling_enabled,
        sampling_max_size=sampling_max_size,
    )

```

## Metadata 🗂️

- **Name**: documents_discovery
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: ai4rag, Version: ~=0.12.0
- **Tags**:
  - data-processing
  - autorag
  - documents-sampling
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
