"""Unit tests for the documents_rag_optimization_pipeline pipeline."""

import tempfile
from pathlib import Path

from kfp import compiler
from kfp_components.utils.pipeline_dag_tasks import (
    assert_compiled_pipeline_root_dag_task_ids,
    load_pipeline_spec_document,
)

from ..pipeline import documents_rag_optimization_pipeline

_EXPECTED_ROOT_DAG_TASK_IDS = (
    "publish-component-stage-map",
    "documents-discovery",
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
        assert "ogx_secret_name" in inputs
        assert "optimization_max_rag_patterns" in inputs
        assert "evaluator" not in inputs
        assert "judge_model_id" not in inputs
        assert "responses_request_default_question" not in inputs

    def test_compiled_pipeline_root_dag_task_ids(self):
        """Root-level step IDs are stable; renames or add/remove steps require updating expectations."""
        assert_compiled_pipeline_root_dag_task_ids(
            pipeline_func=documents_rag_optimization_pipeline,
            expected_task_ids=_EXPECTED_ROOT_DAG_TASK_IDS,
        )

    def test_test_data_loader_runs_after_stage_map_publisher(self):
        """Stage map publisher must complete before downstream components start."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            compiler.Compiler().compile(
                pipeline_func=documents_rag_optimization_pipeline,
                package_path=tmp_path,
            )
            spec = load_pipeline_spec_document(Path(tmp_path))
            loader_task = spec["root"]["dag"]["tasks"]["test-data-loader"]
            assert "publish-component-stage-map" in loader_task["dependentTasks"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_compiled_pipeline_declares_component_resource_tiers(self):
        """Every AutoRAG optimization step declares the expected CPU/memory tier."""
        from kfp_components.utils.pipeline_task_resources import (
            assert_executor_resources,
            compile_executor_resources,
        )

        from .pipeline_resource_expectations import AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES

        assert_executor_resources(
            compile_executor_resources(documents_rag_optimization_pipeline),
            AUTORAG_OPTIMIZATION_EXECUTOR_RESOURCES,
            pipeline_name="documents_rag_optimization_pipeline",
        )
