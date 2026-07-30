from typing import List

from kfp import dsl
from kfp_components.components.data_processing.automl.timeseries_data_loader import timeseries_data_loader
from kfp_components.components.training.automl.autogluon_timeseries_models_training import (
    autogluon_timeseries_models_training,
)
from kfp_components.components.training.automl.component_stage_map_publisher import publish_component_stage_map

MAX_CPUS = "32"
MAX_MEMORY = "64Gi"

# Must match run_status_templates/pipelines/<name>.json
PIPELINE_NAME = "autogluon-timeseries-training-pipeline"


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "AutoML time series forecasting pipeline for building accurate, deployment-ready forecasters with "
        "minimal tuning. Powered by AutoGluon TimeSeries, it compares statistical and deep learning "
        "approaches, refits the best models, and delivers top predictors, back-testing insights, and a "
        "ranked leaderboard for model selection."
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
    known_covariates_names: List[str] = [],
    prediction_length: int = 1,
    top_n: int = 3,
    eval_metric: str = "mean_absolute_scaled_error",
    preset: str = "speed",
    test_data_bucket_name: str = "",
    test_data_file_key: str = "",
):
    """AutoGluon time series training pipeline.

    Trains AutoGluon TimeSeries models on data loaded from S3, scores candidates on a per-series
    temporal holdout, refits the top models on the full train portion (selection + extra splits),
    and aggregates metrics into a leaderboard.

    **Compiled pipeline encoding:** Keep this module ASCII-only (no Unicode in docstrings or
    string literals). Some deployments persist compiled pipeline YAML in MySQL ``utf8`` columns,
    which reject multi-byte characters.

    Storage strategy:

    Train and test CSV splits are produced on the PVC workspace (``PipelineConfig.workspace``) so
    steps can read shared paths without re-downloading. The per-series test split is also exposed as a
    dataset artifact. S3 credentials for the initial load are supplied via the Kubernetes secret
    ``train_data_secret_name``.

    Pipeline stages:

    0. **Component stage map**: Publishes the static component-to-stage-to-step map as a KFP
       artifact for dashboards before data loading.

    1. **Data loading & splitting** (``timeseries_data_loader``): Loads CSV from S3 (up to 100 MB),
       replaces ``+/-inf`` with NaN (missing targets stay for AutoGluon), requires parseable timestamps
       and non-null ids, deduplicates ``(id_column, timestamp_column)``, then applies a two-stage
       **per-series temporal** split on ``id_column`` / ``timestamp_column``:
       default **80/20** train vs test per series, then **30/70** of each series' train rows into
       ``models_selection_train_dataset.csv`` and ``extra_train_dataset.csv`` under
       ``{workspace_path}/datasets/``. The test split is written to the ``sampled_test_dataset`` artifact.

    2. **Model generation + full refit** (``autogluon_timeseries_models_training``): Trains multiple
       AutoGluon TimeSeries models on the selection split, picks top ``top_n``, and refits each selected model
       on the full train portion (**selection + extra** splits). The component writes all refitted models to a
       single combined ``models_artifact``.

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
        known_covariates_names: Column names known in advance for the forecast horizon
            (e.g. holidays, promotions). Defaults to ``[]`` (no known covariates). See
            :attr:`~autogluon.timeseries.TimeSeriesPredictor.known_covariates_names`.
        prediction_length: Number of time steps to forecast (horizon length). Positive integer
            (default: 1).
        top_n: Number of top models to select for the leaderboard and output (default: 3).
        eval_metric: Metric for model ranking in snake_case (e.g. ``"mean_absolute_scaled_error"``,
            ``"weighted_quantile_loss"``) or legacy uppercase acronym form. Defaults to
            ``"mean_absolute_scaled_error"``.
        preset: Training quality tier. ``"speed"`` (default, 4 vCPU / 16 GiB) or
            ``"balanced"`` (may run more than 2x longer, 8 vCPU / 32 GiB).
        test_data_bucket_name: Optional S3-compatible bucket name containing user-provided test dataset.
            If provided, test_data_file_key must also be specified.
        test_data_file_key: Optional S3 object key of the test CSV file (features and target column).
            If provided, test_data_bucket_name must also be specified.

    Returns:
        This pipeline wires task outputs between components; compiled runs expose the combined models artifact
        (per-model predictor, metrics, notebook paths) and leaderboard evaluation artifact (HTML + aggregated
        metrics), subject to Kubeflow Pipelines UI and artifact configuration.

    Raises:
        FileNotFoundError: If the S3 file cannot be found or accessed.
        ValueError: If required columns are missing, temporal splits fail, or inputs are invalid.
        RuntimeError: If AutoGluon training or evaluation fails, or cluster resource limits are exceeded.

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
    # Publish component-to-stage-to-step map first so dashboards know expected structure
    component_stage_map_task = publish_component_stage_map(
        pipeline_id=PIPELINE_NAME,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    component_stage_map_task.set_caching_options(False)
    component_stage_map_task.set_cpu_request("0.5").set_memory_request("512Mi").set_cpu_limit("1").set_memory_limit(
        "1Gi"
    )

    # Stage 1: Data Loading & Splitting
    data_loader_task = timeseries_data_loader(
        bucket_name=train_data_bucket_name,
        file_key=train_data_file_key,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        test_data_bucket_name=test_data_bucket_name,
        test_data_file_key=test_data_file_key,
    )
    data_loader_task.after(component_stage_map_task)
    data_loader_task.set_caching_options(False)
    data_loader_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(MAX_MEMORY)

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

    # Stage 2: Combined model generation + full refit.
    # Resource limits differ by preset: medium_quality needs more CPU/memory.
    _training_kwargs = dict(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        train_data_path=data_loader_task.outputs["models_selection_train_data_path"],
        test_data=data_loader_task.outputs["sampled_test_dataset"],
        top_n=top_n,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names,
        pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        sample_rows=data_loader_task.outputs["sample_rows"],
        sampling_config=data_loader_task.outputs["sample_config"],
        split_config=data_loader_task.outputs["split_config"],
        extra_train_data_path=data_loader_task.outputs["extra_train_data_path"],
        preset=preset,
        eval_metric=eval_metric,
        evaluation_mode=data_loader_task.outputs["evaluation_mode"],
    )
    with dsl.If(preset == "balanced"):
        training_task_bl = autogluon_timeseries_models_training(**_training_kwargs)
        training_task_bl.set_caching_options(False)
        training_task_bl.set_cpu_request("8").set_memory_request("32Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
            MAX_MEMORY
        )

    with dsl.Else():
        training_task_sp = autogluon_timeseries_models_training(**_training_kwargs)
        training_task_sp.set_caching_options(False)
        training_task_sp.set_cpu_request("4").set_memory_request("16Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
            MAX_MEMORY
        )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_training_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
