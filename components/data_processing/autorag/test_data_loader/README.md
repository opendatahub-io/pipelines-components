# Test Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Download test data json file from S3 or PVC into a KFP artifact.

**Data Source Configuration:** - When ``data_source="s3"``: downloads JSON from S3 using AWS credentials - When ``data_source="pvc"``: loads JSON from PVC workspace filesystem

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `test_data_bucket_name` | `str` | `None` | S3 bucket containing the test data file (required when data_source="s3"). |
| `test_data_path` | `str` | `None` | S3 object key to the JSON test data file (required when data_source="s3"). |
| `test_data` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact that receives the downloaded file. |
| `data_source` | `str` | `s3` | Data source type ("s3" or "pvc"). Default is "s3". |
| `pvc_data_path` | `str` | `""` | Path to JSON file on PVC (required when data_source="pvc"). Can be absolute or relative to current directory. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of test_data_loader."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.test_data_loader import test_data_loader


@dsl.pipeline(name="test-data-loader-example")
def example_pipeline(
    test_data_bucket_name: str = "my-bucket",
    test_data_path: str = "test_data/questions.json",
):
    """Example pipeline using test_data_loader.

    Args:
        test_data_bucket_name: S3 bucket containing test data.
        test_data_path: Path to the test data file within the bucket.
    """
    test_data_loader(
        test_data_bucket_name=test_data_bucket_name,
        test_data_path=test_data_path,
    )

```

## Metadata 🗂️

- **Name**: test_data_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - data-processing
  - autorag
  - test-data
- **Last Verified**: 2026-05-13 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
