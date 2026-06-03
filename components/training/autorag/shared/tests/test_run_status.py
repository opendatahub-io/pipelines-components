"""Tests for AutoRAG run status manifest loading."""

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
        "test_data_loader",
        "documents_discovery",
        "text_extraction",
        "search_space_preparation",
        "rag_templates_optimization",
        "prepare_responses_api_requests",
        "leaderboard_evaluation",
    ]


def test_pipeline_component_ids():
    """pipeline_component_ids returns component ids in manifest order."""
    assert pipeline_component_ids(PIPELINE_DOCUMENTS_RAG_OPTIMIZATION)[0] == "test_data_loader"
