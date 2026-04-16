"""Unit tests for the documents_rag_optimization_pipeline pipeline."""

from kfp_components.utils.pipeline_dag_tasks import (
    assert_compiled_pipeline_root_dag_task_ids,
)

from ..pipeline import documents_rag_optimization_pipeline

_EXPECTED_ROOT_DAG_TASK_IDS = (
    "documents-discovery",
    "leaderboard-evaluation",
    "prepare-responses-api-requests",
    "rag-templates-optimization",
    "search-space-preparation",
    "test-data-loader",
    "text-extraction",
)


class TestDocumentsRagOptimizationPipelineUnit:
    """Unit tests for pipeline structure and interface."""

    def test_pipeline_is_callable(self):
        """Pipeline is a GraphComponent (callable with _component_inputs)."""
        assert callable(documents_rag_optimization_pipeline)
        assert hasattr(documents_rag_optimization_pipeline, "_component_inputs")

    def test_pipeline_required_parameters(self):
        """Pipeline declares expected required parameters."""
        inputs = getattr(documents_rag_optimization_pipeline, "_component_inputs", set())
        assert "test_data_secret_name" in inputs
        assert "test_data_bucket_name" in inputs
        assert "test_data_key" in inputs
        assert "input_data_secret_name" in inputs
        assert "input_data_bucket_name" in inputs
        assert "input_data_key" in inputs
        assert "llama_stack_secret_name" in inputs
        assert "responses_request_default_question" not in inputs

    def test_compiled_pipeline_root_dag_task_ids(self):
        """Root-level step IDs are stable; renames or add/remove steps require updating expectations."""
        assert_compiled_pipeline_root_dag_task_ids(
            pipeline_func=documents_rag_optimization_pipeline,
            expected_task_ids=_EXPECTED_ROOT_DAG_TASK_IDS,
        )
