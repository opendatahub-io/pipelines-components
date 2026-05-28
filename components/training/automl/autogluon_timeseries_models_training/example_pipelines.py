"""Example pipelines demonstrating usage of autogluon_timeseries_models_training."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_timeseries_models_training import (
    autogluon_timeseries_models_training,
)


@dsl.pipeline(name="autogluon-timeseries-models-training-example")
def example_pipeline(
    target: str = "value",
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    train_data_path: str = "/tmp/train_data",
    extra_train_data_path: str = "/tmp/extra_train_data",
    top_n: int = 3,
    workspace_path: str = "/tmp/workspace",
    prediction_length: int = 1,
    pipeline_name: str = "autogluon-timeseries-models-training-example",
    run_id: str = "run-001",
):
    """Example pipeline using autogluon_timeseries_models_training.

    Args:
        target: Name of the target column.
        id_column: Name of the ID column.
        timestamp_column: Name of the timestamp column.
        train_data_path: Path to the training data.
        extra_train_data_path: Path to extra training data for full refit.
        top_n: Number of top models to select.
        workspace_path: Path to the workspace directory.
        prediction_length: Number of time steps to predict.
        pipeline_name: Pipeline name used in generated notebook placeholders.
        run_id: Run id used in generated notebook placeholders.
    """
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Dataset,
    )
    autogluon_timeseries_models_training(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        train_data_path=train_data_path,
        test_data=test_data.output,
        top_n=top_n,
        workspace_path=workspace_path,
        pipeline_name=pipeline_name,
        run_id=run_id,
        extra_train_data_path=extra_train_data_path,
        prediction_length=prediction_length,
    )
