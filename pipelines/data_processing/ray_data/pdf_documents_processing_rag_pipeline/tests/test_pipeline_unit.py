"""Unit tests for the RAG multi-step pipeline.

These tests compile the pipeline to YAML and inspect its structure.
No mocks or cluster access needed -- everything runs locally against the
KFP compiler.
"""

import tempfile
from pathlib import Path

from kfp import compiler
from kfp_components.pipelines.data_processing.ray_data.pdf_documents_processing_rag_pipeline import (
    rag_multistep_pipeline,
)


def test_pipeline_compiles():
    """Pipeline compiles to YAML without error."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        compiler.Compiler().compile(rag_multistep_pipeline, f.name)
        assert Path(f.name).stat().st_size > 0


def test_pipeline_signature():
    """All expected parameters present in pipeline spec."""
    spec = rag_multistep_pipeline.component_spec
    input_names = set(spec.inputs)
    # Spot check key params from each section
    assert "pvc_name" in input_names
    assert "s3_endpoint" in input_names
    assert "milvus_host" in input_names
    assert "llm_model_name" in input_names
    assert "deploy_embedding" in input_names
    assert "drop_existing" in input_names
    assert "bypass_kueue" in input_names
    assert "llm_force_recreate" in input_names
    assert "embedding_model" in input_names


def test_root_dag_task_ids():
    """Root DAG has expected task IDs (catches unintended graph changes)."""
    from utils.pipeline_dag_tasks import assert_compiled_pipeline_root_dag_task_ids

    assert_compiled_pipeline_root_dag_task_ids(
        pipeline_func=rag_multistep_pipeline,
        expected_task_ids=(
            "condition-branches-1",
            "download-model",
            "model-deployment",
            "parse-and-chunk",
        ),
    )


def test_compiled_pipeline_has_expected_inputs():
    """Spot-check that compiled YAML contains key input names."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        compiler.Compiler().compile(rag_multistep_pipeline, f.name)
        content = Path(f.name).read_text()
    for param in ["s3_endpoint", "milvus_host", "llm_model_name", "namespace"]:
        assert param in content, f"Expected input '{param}' not found in compiled YAML"
