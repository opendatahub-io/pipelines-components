"""Expected Kubernetes CPU/memory tiers for the documents indexing pipeline."""

from kfp_components.utils.pipeline_task_resources import ExecutorResources

WORKLOAD_RESOURCES = ExecutorResources("2", "8Gi", "32", "64Gi")

AUTORAG_INDEXING_EXECUTOR_RESOURCES = {
    "documents-discovery": WORKLOAD_RESOURCES,
    "text-extraction": WORKLOAD_RESOURCES,
    "documents-indexing": WORKLOAD_RESOURCES,
}
