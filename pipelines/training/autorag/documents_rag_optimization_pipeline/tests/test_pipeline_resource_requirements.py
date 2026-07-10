"""Unit tests for RAG optimization pipeline executor resource tiers."""

from kfp_components.pipelines.data_processing.autorag.documents_indexing_pipeline.pipeline import (
    documents_indexing_pipeline,
)
from kfp_components.utils.pipeline_task_resources import (
    assert_executor_resources,
    compile_executor_resources,
    normalize_executor_name,
)

from ..pipeline import documents_rag_optimization_pipeline
from .pipeline_resource_expectations import AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES, STANDARD_RESOURCES


class TestDocumentsRagOptimizationPipelineResourceRequirements:
    """RAG optimization pipeline assigns workload tiers to data/HPO steps."""

    def test_rag_optimization_pipeline_executor_resources(self):
        """RAG optimization pipeline sets resources on every component step."""
        assert_executor_resources(
            compile_executor_resources(documents_rag_optimization_pipeline),
            AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES,
            pipeline_name="documents_rag_optimization_pipeline",
        )

    def test_indexing_pipeline_shared_components_match_rag_optimization(self):
        """documents-discovery uses the same tier in both AutoRAG pipelines."""
        rag_resources = compile_executor_resources(documents_rag_optimization_pipeline)
        indexing_resources = compile_executor_resources(documents_indexing_pipeline)

        rag_by_task = {normalize_executor_name(name): res for name, res in rag_resources.items()}
        indexing_by_task = {normalize_executor_name(name): res for name, res in indexing_resources.items()}

        task_name = "documents-discovery"
        assert task_name in rag_by_task, f"{task_name} missing from RAG optimization pipeline"
        assert task_name in indexing_by_task, f"{task_name} missing from indexing pipeline"
        assert rag_by_task[task_name] == indexing_by_task[task_name] == STANDARD_RESOURCES, (
            f"{task_name} resources differ between pipelines: "
            f"rag={rag_by_task[task_name]}, indexing={indexing_by_task[task_name]}"
        )
