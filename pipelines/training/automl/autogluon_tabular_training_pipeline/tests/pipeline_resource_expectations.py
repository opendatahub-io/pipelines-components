"""Expected Kubernetes CPU/memory tiers for the tabular training pipeline."""

from kfp_components.utils.pipeline_task_resources import ExecutorResources

STAGE_MAP_RESOURCES = ExecutorResources("0.5", "512Mi", "1", "1Gi")
WORKLOAD_RESOURCES = ExecutorResources("4", "8Gi", "32", "64Gi")
TRAINING_SPEED_RESOURCES = ExecutorResources("4", "16Gi", "32", "64Gi")
TRAINING_BALANCED_RESOURCES = ExecutorResources("8", "32Gi", "32", "64Gi")

AUTOML_TABULAR_EXECUTOR_RESOURCES = {
    "publish-component-stage-map": STAGE_MAP_RESOURCES,
    "automl-data-loader": WORKLOAD_RESOURCES,
    "autogluon-models-training": TRAINING_BALANCED_RESOURCES,
    "autogluon-models-training-2": TRAINING_SPEED_RESOURCES,
}
