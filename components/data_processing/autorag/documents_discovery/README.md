# Documents Discovery ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Documents discovery component.

Lists available documents from S3 or PVC, performs sampling if applied and writes a JSON manifest (documents_descriptor.json) with metadata. Does not download document contents.

**Data Source Configuration:** - When ``data_source="s3"``: lists documents from S3 bucket using AWS credentials - When ``data_source="pvc"``: discovers documents in PVC workspace directory

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `input_data_bucket_name` | `str` | `None` | S3 bucket containing input data (required when data_source="s3"). |
| `input_data_path` | `str` | `""` | For S3: prefix; for PVC: directory path within workspace. |
| `test_data` | `dsl.Input[dsl.Artifact]` | `None` | Optional input artifact containing test data for sampling. |
| `sampling_enabled` | `bool` | `True` | Whether to enable sampling or not. |
| `sampling_max_size` | `float` | `1` | Maximum size of sampled documents (in gigabytes). |
| `data_source` | `str` | `s3` | Data source type ("s3" or "pvc"). Default is "s3". |
| `pvc_data_path` | `str` | `""` | Directory path on PVC containing documents (required when data_source="pvc"). Can be absolute or relative to current directory. |
| `discovered_documents` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the documents descriptor JSON file. |

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
    - Name: ai4rag, Version: ~=0.6.1
- **Tags**:
  - data-processing
  - autorag
  - documents-sampling
- **Last Verified**: 2026-05-14 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
