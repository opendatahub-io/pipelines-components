from typing import List, Optional

from kfp import dsl
from kfp_components.components.data_processing.automl.timeseries_data_loader import timeseries_data_loader
from kfp_components.components.training.automl.autogluon_timeseries_leaderboard_evaluation import (
    timeseries_leaderboard_evaluation,
)
from kfp_components.components.training.automl.autogluon_timeseries_models_full_refit import (
    autogluon_timeseries_models_full_refit,
)
from kfp_components.components.training.automl.autogluon_timeseries_models_selection import (
    autogluon_timeseries_models_selection,
)


@dsl.pipeline(
    name="autogluon-timeseries-training-pipeline",
    description=(
        "End-to-end AutoGluon time series forecasting pipeline. Loads time series data from S3 in "
        "TimeSeriesDataFrame format (item_id, timestamp, target), trains multiple AutoGluon TimeSeries models "
        "(local statistical and global deep learning), ranks them by the chosen eval metric, and produces a "
        "leaderboard and the top N trained predictors for deployment. Supports optional known covariates and "
        "configurable prediction_length."
    ),
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size="12Gi",  # TODO: change to recommended size
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "accessModes": ["ReadWriteOnce"],
                }
            ),
        ),
    ),
)
def autogluon_timeseries_training_pipeline(
    train_data_secret_name: str,
    train_data_bucket_name: str,
    train_data_file_key: str,
    target: str,
    id_column: str,
    timestamp_column: str,
    known_covariates_names: Optional[List[str]] = None,
    prediction_length: int = 1,
    top_n: int = 3,
):
    """AutoGluon time series training pipeline.

    Trains AutoGluon TimeSeries models on data loaded from S3, scores candidates on a per-series
    temporal holdout, refits the top models on the full train portion (selection + extra splits),
    and aggregates metrics into a leaderboard.

    Storage strategy:

    Train and test CSV splits are produced on the PVC workspace (``PipelineConfig.workspace``) so
    steps can read shared paths without re-downloading. The per-series test split is also exposed as a
    dataset artifact. S3 credentials for the initial load are supplied via the Kubernetes secret
    ``train_data_secret_name``.

    Pipeline stages:

    1. **Data loading & splitting** (``timeseries_data_loader``): Loads CSV from S3 (up to 1GB), then
       applies a two-stage **per-series temporal** split on ``id_column`` / ``timestamp_column``:
       default **80/20** train vs test per series, then **30/70** of each series' train rows into
       ``models_selection_train_dataset.csv`` and ``extra_train_dataset.csv`` under
       ``{workspace_path}/datasets/``. The test split is written to the ``sampled_test_dataset`` artifact.

    2. **Model selection** (``autogluon_timeseries_models_selection``): Trains multiple AutoGluon TimeSeries
       models on the selection split, scores them on the test split, and emits the top ``top_n`` models
       plus predictor path and configuration.

    3. **Full refit** (``autogluon_timeseries_models_full_refit``, ``ParallelFor`` with parallelism 1): For
       each top model, fits a new predictor on **selection + extra** train data (full train portion per
       series), evaluates on the test split, and writes a ``_FULL`` model artifact (predictor, metrics,
       notebook). Parallelism is **one** concurrent refit pod because the workspace PVC is
       **ReadWriteOnce**; higher parallelism would mount the same RWO volume from multiple pods and cause
       **Multi-Attach** errors on typical block storage.

    4. **Leaderboard** (``leaderboard_evaluation``): Builds an HTML leaderboard from the refitted model
       metrics using the selection stage's evaluation metric.

    Args:
        train_data_secret_name: Kubernetes secret name containing S3 credentials
            (e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION).
        train_data_bucket_name: S3-compatible bucket name containing the time series data file.
        train_data_file_key: S3 object key of the data file (CSV or Parquet). File must include
            columns for item_id, timestamp, and target; optional columns for known covariates.
        target: Name of the column containing the numeric values to forecast. Corresponds to
            :attr:`~autogluon.timeseries.TimeSeriesDataFrame` target column.
        id_column: Name of the column that identifies each time series (e.g. product_id, store_id).
            Passed as ``id_column`` when constructing TimeSeriesDataFrame; result uses ``item_id``.
        timestamp_column: Name of the column containing the timestamp/datetime for each observation.
            Passed as ``timestamp_column`` when constructing TimeSeriesDataFrame; result uses
            ``timestamp`` as the second index level.
        known_covariates_names: Optional list of column names known in advance for the forecast
            horizon (e.g. holidays, promotions). See
            :attr:`~autogluon.timeseries.TimeSeriesPredictor.known_covariates_names`.
        prediction_length: Number of time steps to forecast (horizon length). Positive integer
            (default: 1).
        top_n: Number of top models to select for the leaderboard and output (default: 3).

    Returns:
        This pipeline wires task outputs between components; compiled runs expose artifacts from the
        refit tasks (Model artifacts with predictor, metrics, notebook paths) and the leaderboard
        evaluation task (HTML leaderboard and aggregated metrics), subject to Kubeflow Pipelines UI
        and artifact configuration.

    Raises:
        Component and runtime failures propagate from the underlying steps (for example: S3 access or
        empty data from the loader, invalid inputs, AutoGluon training or evaluation errors, or
        resource limits in the cluster).

    Example:
        pipeline = autogluon_timeseries_training_pipeline(
            train_data_secret_name="my-s3-secret",
            train_data_bucket_name="my-bucket",
            train_data_file_key="ts/sales.csv",
            target="sales",
            id_column="product_id",
            timestamp_column="date",
            known_covariates_names=["is_holiday", "promo"],
            prediction_length=14,
            top_n=3,
        )
    """
    # Stage 1: Data Loading & Splitting
    data_loader_task = timeseries_data_loader(
        bucket_name=train_data_bucket_name,
        file_key=train_data_file_key,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    data_loader_task.set_caching_options(False)
    data_loader_task.set_cpu_request("2").set_memory_request("8Gi")

    # Configure S3 secret for data loader
    from kfp.kubernetes import use_secret_as_env

    use_secret_as_env(
        data_loader_task,
        secret_name=train_data_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
        optional=True,
    )

    # Stage 2: Model Selection
    # Train multiple models on selection data and select top N performers
    selection_task = autogluon_timeseries_models_selection(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        train_data_path=data_loader_task.outputs["models_selection_train_data_path"],
        test_data=data_loader_task.outputs["sampled_test_dataset"],
        top_n=top_n,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names,
    )
    selection_task.set_caching_options(False)
    selection_task.set_cpu_request("4").set_memory_request("16Gi")

    # Stage 3: Model Refitting (parallelism=1: RWO workspace allows only one pod on the volume at a time).
    with dsl.ParallelFor(items=selection_task.outputs["top_models"], parallelism=1) as model_name:
        refit_task = autogluon_timeseries_models_full_refit(
            model_name=model_name,
            test_dataset=data_loader_task.outputs["sampled_test_dataset"],
            predictor_path=selection_task.outputs["predictor_path"],
            sampling_config=data_loader_task.outputs["sample_config"],
            split_config=data_loader_task.outputs["split_config"],
            model_config=selection_task.outputs["model_config"],
            pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            models_selection_train_data_path=data_loader_task.outputs["models_selection_train_data_path"],
            extra_train_data_path=data_loader_task.outputs["extra_train_data_path"],
            sample_rows=data_loader_task.outputs["sample_rows"],
        )
        refit_task.set_caching_options(False)
        refit_task.set_cpu_request("2").set_memory_request("8Gi")

    # Stage 4: Leaderboard Evaluation
    # Generate leaderboard from all refitted models
    leaderboard_task = timeseries_leaderboard_evaluation(
        models=dsl.Collected(refit_task.outputs["model_artifact"]),
        eval_metric=selection_task.outputs["eval_metric_name"],
    )
    leaderboard_task.set_caching_options(False)
    leaderboard_task.set_cpu_request("1").set_memory_request("4Gi")


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_training_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
