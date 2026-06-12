"""Tests for AutoRAG component-local status tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kfp_components.components.training.autorag.shared.component_status import (
    COMPONENT_STATUS_FILENAME,
    ComponentStatusTracker,
    component_status_tracker,
    load_component_status,
)


class TestComponentStatusTracker:
    """Tests for ComponentStatusTracker."""

    def test_record_and_save(self, tmp_path: Path) -> None:
        """save() writes stages and component_id to component_status.json."""
        tracker = ComponentStatusTracker(str(tmp_path), "test_data_loader")
        tracker.record("validate_inputs", "started", rows=5)
        tracker.record("validate_inputs", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["component_id"] == "test_data_loader"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["status"] == "completed"
        assert data["stages"][0]["rows"] == 5

    def test_disabled_tracker_skips_save(self, tmp_path: Path) -> None:
        """When artifact_path is None, save() is a no-op."""
        tracker = ComponentStatusTracker(None, "test_data_loader")
        tracker.record("validate_inputs", "completed")
        tracker.save()
        assert not (tmp_path / COMPONENT_STATUS_FILENAME).exists()

    def test_component_status_tracker_from_none(self, tmp_path: Path) -> None:
        """component_status_tracker() accepts a missing artifact for unit tests."""
        tracker = component_status_tracker(None, "documents_discovery")
        tracker.record("validate_inputs", "completed")
        tracker.save()
        assert not (tmp_path / COMPONENT_STATUS_FILENAME).exists()

    def test_context_manager_marks_failed_and_saves(self, tmp_path: Path) -> None:
        """Context manager marks active stage failed and persists status on exception."""
        with pytest.raises(RuntimeError, match="boom"):
            with ComponentStatusTracker(str(tmp_path), "text_extraction") as status:
                status.record("extract_documents", "started")
                raise RuntimeError("boom")

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["status"] == "failed"
        assert "boom" in data["stages"][-1]["error"]

    def test_record_completed_with_steps(self, tmp_path: Path) -> None:
        """Completed stages can include manifest step ids for dashboard display."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        steps = ["chunking", "embedding", "retrieval", "generation", "evaluation"]
        tracker.record("run_optimization", "started")
        tracker.record("run_optimization", "completed", steps=steps)
        tracker.save()

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["steps"] == steps
