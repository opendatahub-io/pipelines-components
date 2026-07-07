"""Expected Kubernetes CPU/memory tiers for the RAG optimization pipeline."""

from kfp_components.utils.pipeline_task_resources import ExecutorResources

STAGE_MAP_RESOURCES = ExecutorResources("0.5", "512Mi", "1", "1Gi")
WORKLOAD_RESOURCES = ExecutorResources("2", "8Gi", "32", "64Gi")

AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES = {
    "publish-component-stage-map": STAGE_MAP_RESOURCES,
    "documents-discovery": WORKLOAD_RESOURCES,
    "text-extraction": WORKLOAD_RESOURCES,
    "search-space-preparation": WORKLOAD_RESOURCES,
    "rag-templates-optimization": WORKLOAD_RESOURCES,
}
