# Linear Model ✨

## Overview 🧾

Trains a simple linear model on input data.

This component trains a linear regression model used for testing subcategory README generation.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `features` | `str` | `None` | Path to the features dataset. |
| `target` | `str` | `None` | Name of the target column. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | Path to the trained model artifact. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of linear_model."""

from kfp import dsl
from kfp_components.components.grouped.ml_models.linear_model import linear_model


@dsl.pipeline(name="linear-model-example")
def example_pipeline(data_path: str = "/data/train.csv", target_col: str = "price"):
    """Example pipeline using linear_model.

    Args:
        data_path: Path to training data.
        target_col: Target column name.
    """
    linear_model(features=data_path, target=target_col)

```

## Metadata 🗂️

- **Name**: Linear Model
- **Description**: A linear regression training component
- **Documentation**: https://example.com/linear-model
- **Tags**:
  - testing
  - subcategory
- **Owners**:
  - Approvers:
    - nsingla
    - hbelmiro
  - Reviewers:
    - nsingla
