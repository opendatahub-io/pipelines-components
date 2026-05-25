"""Example pipelines demonstrating usage of automl_data_loader."""

from kfp import dsl
from kfp_components.components.data_processing.automl.tabular_data_loader import automl_data_loader


@dsl.pipeline(name="tabular-data-loader-example")
def example_pipeline(
    file_key: str = "data/train.csv",
    bucket_name: str = "my-bucket",
    workspace_path: str = "/tmp/workspace",
    label_column: str = "target",
    task_type: str = "regression",
    selection_train_size: float = 0.3,
):
    """Example pipeline using automl_data_loader.

    Args:
        file_key: S3 key of the data file.
        bucket_name: S3 bucket name.
        workspace_path: Path to the workspace directory.
        label_column: Name of the label column.
        task_type: Type of ML task.
        selection_train_size: Fraction of data for training.
    """
    automl_data_loader(
        file_key=file_key,
        bucket_name=bucket_name,
        workspace_path=workspace_path,
        label_column=label_column,
        task_type=task_type,
        selection_train_size=selection_train_size,
        pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
