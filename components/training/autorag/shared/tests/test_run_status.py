"""Tests for AutoRAG run status manifest loading."""

import pytest
from kfp_components.components.training.autorag.shared.run_status import (
    PIPELINE_DOCUMENTS_RAG_OPTIMIZATION,
    load_pipeline_run_status_manifest,
    pipeline_component_ids,
    shared_autorag_dir,
)


def test_notebook_templates_packaged_in_shared():
    """Indexing and inference notebook templates ship with autorag.shared."""
    templates_dir = shared_autorag_dir() / "notebook_templates"
    assert (templates_dir / "ogx_indexing_template.ipynb").is_file()
    assert (templates_dir / "ogx_inference_template.ipynb").is_file()


def test_load_documents_rag_optimization_manifest():
    """Stage map JSON lists all documents RAG optimization pipeline components."""
    manifest = load_pipeline_run_status_manifest(PIPELINE_DOCUMENTS_RAG_OPTIMIZATION)
    assert manifest["pipeline_id"] == PIPELINE_DOCUMENTS_RAG_OPTIMIZATION
    component_ids = [component["id"] for component in manifest["components"]]
    assert component_ids == [
        "test_data_loader",
        "documents_discovery",
        "text_extraction",
        "search_space_preparation",
        "rag_templates_optimization",
        "leaderboard_evaluation",
    ]


def test_pipeline_component_ids():
    """pipeline_component_ids returns component ids in manifest order."""
    assert pipeline_component_ids(PIPELINE_DOCUMENTS_RAG_OPTIMIZATION)[0] == "test_data_loader"


def test_load_manifest_rejects_path_traversal():
    """load_pipeline_run_status_manifest rejects pipeline_id with path separators."""
    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("../../../etc/passwd")

    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("foo/bar")

    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("foo\\bar")

    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("..hidden")


def test_load_manifest_rejects_empty_pipeline_id():
    """load_pipeline_run_status_manifest rejects empty pipeline_id."""
    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("")

    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("   ")
