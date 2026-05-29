# Test Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Download test data JSON from S3 and sample it for benchmarking.

The component reads S3-compatible credentials from environment variables (injected by the pipeline from a Kubernetes secret), downloads a JSON test data file, and randomly samples up to ``benchmark_sample_size`` records to limit evaluation cost in downstream components.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `test_data_bucket_name` | `str` | `None` | S3 (or compatible) bucket that contains the test data file. |
| `test_data_path` | `str` | `None` | S3 object key to the JSON test data file. |
| `benchmark_sample_size` | `int` | `25` | Maximum number of records to keep from the test data. When the dataset exceeds this limit, a reproducible random sample is drawn (seed 42). Set to 0 to disable sampling and keep all records. |
| `test_data` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact that receives the (possibly sampled) file. |

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
- **Last Verified**: 2026-01-23 10:29:45+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski
