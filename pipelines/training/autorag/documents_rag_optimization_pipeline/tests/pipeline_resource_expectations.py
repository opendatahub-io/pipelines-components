"""Expected Kubernetes CPU/memory tiers for the RAG optimization pipeline."""

from kfp_components.utils.pipeline_task_resources import ExecutorResources

STAGE_MAP_RESOURCES = ExecutorResources("0.5", "512Mi", "1", "1Gi")
STANDARD_RESOURCES = ExecutorResources("2", "8Gi", "32", "64Gi")
HEAVY_RESOURCES = ExecutorResources("4", "16Gi", "32", "64Gi")

AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES = {
    "publish-component-stage-map": STAGE_MAP_RESOURCES,
    "test-data-loader": STANDARD_RESOURCES,
    "documents-discovery": STANDARD_RESOURCES,
    "text-extraction": HEAVY_RESOURCES,
    "search-space-preparation": STANDARD_RESOURCES,
    "rag-templates-optimization": HEAVY_RESOURCES,
}
