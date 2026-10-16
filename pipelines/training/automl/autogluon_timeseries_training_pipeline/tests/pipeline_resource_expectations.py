"""Expected Kubernetes CPU/memory tiers for the time series training pipeline."""

from kfp_components.utils.pipeline_task_resources import ExecutorResources

STAGE_MAP_RESOURCES = ExecutorResources("0.5", "512Mi", "1", "1Gi")
LOADER_SPEED_RESOURCES = ExecutorResources("2", "16Gi", "32", "64Gi")
LOADER_BALANCED_RESOURCES = ExecutorResources("2", "32Gi", "32", "64Gi")
LOADER_QUALITY_RESOURCES = ExecutorResources("2", "64Gi", "32", "64Gi")
TRAINING_SPEED_RESOURCES = ExecutorResources("4", "16Gi", "32", "64Gi")
TRAINING_BALANCED_RESOURCES = ExecutorResources("8", "32Gi", "32", "64Gi")
TRAINING_QUALITY_RESOURCES = ExecutorResources("16", "64Gi", "32", "128Gi")

AUTOML_TIMESERIES_EXECUTOR_RESOURCES = {
    "publish-component-stage-map": STAGE_MAP_RESOURCES,
    "timeseries-data-loader": LOADER_BALANCED_RESOURCES,
    "timeseries-data-loader-2": LOADER_QUALITY_RESOURCES,
    "timeseries-data-loader-3": LOADER_SPEED_RESOURCES,
    "autogluon-timeseries-models-training": TRAINING_BALANCED_RESOURCES,
    "autogluon-timeseries-models-training-2": TRAINING_QUALITY_RESOURCES,
    "autogluon-timeseries-models-training-3": TRAINING_SPEED_RESOURCES,
}
