"""Tests for AutoRAG run status manifest loading."""

import pytest
from kfp_components.components.training.autorag.shared.run_status import (
    PIPELINE_DOCUMENTS_RAG_OPTIMIZATION,
    load_pipeline_run_status_manifest,
    pipeline_component_ids,
)


def test_load_documents_rag_optimization_manifest():
    """Stage map JSON lists all documents RAG optimization pipeline components."""
    manifest = load_pipeline_run_status_manifest(PIPELINE_DOCUMENTS_RAG_OPTIMIZATION)
    assert manifest["pipeline_id"] == PIPELINE_DOCUMENTS_RAG_OPTIMIZATION
    component_ids = [component["id"] for component in manifest["components"]]
    assert component_ids == [
        "documents_discovery",
        "search_space_preparation",
        "text_extraction",
        "models_pre_selector",
        "rag_templates_optimization",
    ]


def test_pipeline_component_ids():
    """pipeline_component_ids returns component ids in manifest order."""
    ids = pipeline_component_ids(PIPELINE_DOCUMENTS_RAG_OPTIMIZATION)
    assert ids == [
        "documents_discovery",
        "search_space_preparation",
        "text_extraction",
        "models_pre_selector",
        "rag_templates_optimization",
    ]


def test_load_manifest_rejects_path_traversal():
    """load_pipeline_run_status_manifest rejects pipeline_id with path separators."""
    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("../../../etc/passwd")

    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("foo/bar")

    with pytest.raises(ValueError, match="must be a simple identifier"):
        load_pipeline_run_status_manifest("foo\\bar")


def test_load_manifest_rejects_empty_pipeline_id():
    """load_pipeline_run_status_manifest rejects empty pipeline_id."""
    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        load_pipeline_run_status_manifest("")

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        load_pipeline_run_status_manifest("   ")


def test_load_manifest_missing_raises_file_not_found(tmp_path):
    """load_pipeline_run_status_manifest raises when the manifest file is absent."""
    with pytest.raises(FileNotFoundError, match="nonexistent-pipeline"):
        load_pipeline_run_status_manifest("nonexistent-pipeline", templates_root=str(tmp_path))
