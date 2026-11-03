"""Tests for AutoRAG component-local status tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kfp_components.components.training.autorag.shared.component_status import (
    COMPONENT_STATUS_FILENAME,
    ComponentStatusEncoder,
    ComponentStatusTracker,
    NullComponentStatusTracker,
    bootstrap_status_tracker,
    load_component_status,
    load_embedded_component_status_module,
    utc_now_z,
)

_CANONICAL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["component_id", "started_at", "stages", "metadata"],
    "additionalProperties": False,
    "properties": {
        "component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "started_at": {"type": "string", "format": "date-time"},
        "completed_at": {"type": "string", "format": "date-time"},
        "metadata": {
            "type": "object",
            "required": ["display_name"],
            "additionalProperties": True,
            "properties": {
                "display_name": {"type": "string", "minLength": 1},
            },
        },
        "stages": {
            "type": "array",
            "items": {"$ref": "#/$defs/stage"},
        },
    },
    "$defs": {
        "statusMessage": {
            "type": "object",
            "required": ["level", "text"],
            "additionalProperties": False,
            "properties": {
                "level": {"type": "string", "enum": ["info", "warning", "error"]},
                "text": {"type": "string", "minLength": 1},
            },
        },
        "stageStatus": {
            "type": "object",
            "required": ["state"],
            "additionalProperties": False,
            "properties": {
                "state": {"type": "string", "enum": ["started", "running", "completed", "failed"]},
                "step": {"type": "string"},
                "message": {"$ref": "#/$defs/statusMessage"},
                "running_at": {"type": "string", "format": "date-time"},
            },
        },
        "stage": {
            "type": "object",
            "required": ["id", "status"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                "status": {"$ref": "#/$defs/stageStatus"},
                "metrics": {
                    "type": "object",
                    "additionalProperties": {
                        "type": ["string", "number", "boolean", "array", "null"],
                        "items": {"type": ["string", "number", "boolean"]},
                    },
                },
                "error": {"type": "string"},
            },
        },
    },
}


@pytest.fixture()
def canonical_schema() -> dict:
    """Return the canonical component_status JSON Schema."""
    return _CANONICAL_SCHEMA


def _save_and_load(tracker: ComponentStatusTracker) -> dict:
    """Save tracker to disk and return the parsed JSON."""
    tracker.save()
    return json.loads((Path(tracker.artifact_path) / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))


def _assert_schema_valid(instance: dict, schema: dict) -> None:
    """Validate instance against schema (lazy-import jsonschema for import guard)."""
    from jsonschema import validate

    validate(instance=instance, schema=schema)


class TestComponentStatusTracker:
    """Tests for ComponentStatusTracker."""

    def test_record_and_save(self, tmp_path: Path) -> None:
        """save() writes stages with nested status and metrics to component_status.json."""
        tracker = ComponentStatusTracker(str(tmp_path), "test_data_loader")
        tracker.record("load_benchmark", "started", metrics={"rows": 5})
        tracker.record("load_benchmark", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["component_id"] == "test_data_loader"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["status"]["state"] == "completed"
        assert data["stages"][0]["metrics"]["rows"] == 5

    def test_context_manager_marks_failed_and_saves(self, tmp_path: Path) -> None:
        """Context manager marks active stage failed and persists status on exception."""
        with pytest.raises(RuntimeError, match="boom"):
            with ComponentStatusTracker(str(tmp_path), "text_extraction") as status:
                status.record("extract_documents", "started")
                raise RuntimeError("boom")

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["status"]["state"] == "failed"
        assert "boom" in data["stages"][-1]["error"]

    def test_stage_skips_auto_complete_when_completed_inside_block(self, tmp_path: Path) -> None:
        """stage() does not overwrite a completed record written inside the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        with tracker.stage("optimize_templates"):
            tracker.record(
                "optimize_templates",
                "completed",
                metrics={"max_rag_patterns": 8},
            )
        tracker.save()

        data = load_component_status(str(tmp_path))
        run_stage = next(stage for stage in data["stages"] if stage["id"] == "optimize_templates")
        assert run_stage["status"]["state"] == "completed"
        assert run_stage["metrics"]["max_rag_patterns"] == 8

    def test_utc_now_z_ends_with_z(self) -> None:
        """Timestamps use UTC ISO-8601 with Z suffix."""
        assert utc_now_z().endswith("Z")

    def test_mark_active_failed_marks_started_stage(self, tmp_path: Path) -> None:
        """mark_active_failed() updates the in-progress stage to failed."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        tracker.record("optimize_templates", "started")
        tracker.mark_active_failed("optimization timeout")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["status"]["state"] == "failed"
        assert data["stages"][-1]["error"] == "optimization timeout"

    def test_save_best_effort_swallows_io_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_best_effort() logs I/O errors instead of raising."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        tracker.record("build_leaderboard", "started")

        def _raise_save() -> None:
            raise OSError("disk full")

        monkeypatch.setattr(tracker, "save", _raise_save)
        tracker.save_best_effort()

    def test_context_manager_saves_on_success(self, tmp_path: Path) -> None:
        """Context manager persists status when the block completes normally."""
        with ComponentStatusTracker(str(tmp_path), "documents_discovery") as status:
            status.record("discover_documents", "completed")

        assert (tmp_path / COMPONENT_STATUS_FILENAME).exists()

    def test_stage_context_manager_records_completed(self, tmp_path: Path) -> None:
        """stage() records started then completed when no exception is raised."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        with tracker.stage("discover_documents"):
            pass
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["id"] == "discover_documents"
        assert data["stages"][-1]["status"]["state"] == "completed"

    def test_stage_context_manager_records_failed(self, tmp_path: Path) -> None:
        """stage() records failed when an exception escapes the block."""
        tracker = ComponentStatusTracker(str(tmp_path), "text_extraction")
        with pytest.raises(ValueError, match="bad format"):
            with tracker.stage("extract_documents"):
                raise ValueError("bad format")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][-1]["status"]["state"] == "failed"
        assert "bad format" in data["stages"][-1]["error"]

    def test_running_state_sets_running_at(self, tmp_path: Path) -> None:
        """Recording state=running auto-sets running_at timestamp."""
        tracker = ComponentStatusTracker(str(tmp_path), "search_space_preparation")
        tracker.record("prepare_search_space", "running")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["state"] == "running"
        assert "running_at" in data["stages"][0]["status"]
        assert data["stages"][0]["status"]["running_at"].endswith("Z")

    def test_step_field_on_status(self, tmp_path: Path) -> None:
        """Step is placed inside the nested status object."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        tracker.record("optimize_templates", "running", step="embedding")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["step"] == "embedding"

    def test_message_field_on_status(self, tmp_path: Path) -> None:
        """Message is placed inside the nested status object."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.record("load_benchmark", "running", message={"level": "info", "text": "Loading data..."})
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["message"] == {"level": "info", "text": "Loading data..."}

    def test_metrics_accumulate_across_records(self, tmp_path: Path) -> None:
        """Metrics from multiple record() calls on the same stage merge together."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.record("load_benchmark", "started", metrics={"rows": 100})
        tracker.record("load_benchmark", "completed", metrics={"duplicates_dropped": 5})
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["metrics"]["rows"] == 100
        assert data["stages"][0]["metrics"]["duplicates_dropped"] == 5

    def test_completed_at_set_when_finished(self, tmp_path: Path) -> None:
        """completed_at is set only when all stages are completed."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.record("load_benchmark", "completed")
        tracker.record("discover_documents", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert "completed_at" in data

    def test_completed_at_not_set_while_running(self, tmp_path: Path) -> None:
        """completed_at is not set while stages are still in progress."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.record("load_benchmark", "completed")
        tracker.record("discover_documents", "started")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert "completed_at" not in data

    def test_completed_at_preserved_across_repeated_saves(self, tmp_path: Path) -> None:
        """First terminal completed_at is kept when save() is called again."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.record("load_benchmark", "completed")
        tracker.save()
        first = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))["completed_at"]

        tracker.save()
        second = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))["completed_at"]
        assert second == first

    def test_failed_to_completed_clears_stale_error(self, tmp_path: Path) -> None:
        """Updating a failed stage to completed removes the previous error."""
        tracker = ComponentStatusTracker(str(tmp_path), "text_extraction")
        tracker.record("extract_documents", "failed", error="ValueError: boom")
        tracker.record("extract_documents", "completed")
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        assert data["stages"][0]["status"]["state"] == "completed"
        assert "error" not in data["stages"][0]

    def test_invalid_state_raises(self, tmp_path: Path) -> None:
        """record() rejects invalid state values."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        with pytest.raises(ValueError, match="state must be one of"):
            tracker.record("load_benchmark", "unknown_state")

    def test_no_flat_keys_on_stage(self, tmp_path: Path) -> None:
        """Stage objects only contain id, status, metrics, and error — no flat keys."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.record("load_benchmark", "completed", metrics={"rows": 10})
        tracker.save()

        data = json.loads((tmp_path / COMPONENT_STATUS_FILENAME).read_text(encoding="utf-8"))
        allowed_keys = {"id", "status", "metrics", "error"}
        actual_keys = set(data["stages"][0].keys())
        assert actual_keys <= allowed_keys, f"Unexpected stage keys: {actual_keys - allowed_keys}"


class TestComponentStatusEncoder:
    """Tests for JSON encoding of status metadata values."""

    def test_encodes_datetime_path_bytes_and_set(self) -> None:
        """Known non-JSON types are converted for serialization."""
        from datetime import UTC, datetime

        encoder = ComponentStatusEncoder()
        assert encoder.default(Path("/tmp/out")) == "/tmp/out"
        assert encoder.default(b"abc") == "YWJj"
        assert encoder.default({1, 2}) == [1, 2]
        encoded = encoder.default(datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC))
        assert encoded.endswith("Z")

    def test_unknown_type_raises_type_error(self) -> None:
        """Unsupported metadata types fail fast instead of being stringified."""
        encoder = ComponentStatusEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())


class TestEmbeddedStatusBootstrap:
    """Tests for embedded-artifact loader helpers."""

    def test_bootstrap_status_tracker_from_shared_dir(self, tmp_path: Path) -> None:
        """bootstrap_status_tracker loads from a directory embedded artifact path."""
        embed_dir = Path(__file__).resolve().parents[1] / "runtime_embed"
        embedded = type("Embedded", (), {"path": str(embed_dir)})()
        status = bootstrap_status_tracker(embedded, type("Status", (), {"path": str(tmp_path)})(), "test_data_loader")
        status.record("load_benchmark", "completed")
        status.save()
        assert (tmp_path / COMPONENT_STATUS_FILENAME).is_file()

    def test_load_embedded_module_from_file_path(self) -> None:
        """load_embedded_component_status_module accepts a file embedded artifact path."""
        module_path = Path(__file__).resolve().parents[1] / "runtime_embed" / "component_status.py"
        embedded = type("Embedded", (), {"path": str(module_path)})()
        module = load_embedded_component_status_module(embedded)
        assert hasattr(module, "bootstrap_status_tracker")

    def test_bootstrap_status_tracker_returns_noop_when_component_status_is_none(self) -> None:
        """Notebook-style invocations without component_status use a no-op tracker."""
        embed_dir = Path(__file__).resolve().parents[1] / "runtime_embed"
        embedded = type("Embedded", (), {"path": str(embed_dir)})()
        status = bootstrap_status_tracker(embedded, None, "documents_discovery")
        assert isinstance(status, NullComponentStatusTracker)
        with status:
            with status.stage("discover_documents"):
                pass


class TestNullComponentStatusTracker:
    """Tests for no-op status tracker used in notebook execution."""

    def test_null_tracker_propagates_exceptions(self) -> None:
        """Exceptions escape the context manager (not suppressed)."""
        from kfp_components.components.training.autorag.shared.component_status import (
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
        with pytest.raises(ValueError, match="test error"):
            with status:
                raise ValueError("test error")

    def test_null_tracker_creates_no_files(self, tmp_path: Path) -> None:
        """No artifacts written when using null tracker."""
        status = NullComponentStatusTracker()
        with status:
            status.record("stage1", "started")
            status.set_metadata(key="value")
            with status.stage("stage2"):
                pass

        assert list(tmp_path.iterdir()) == []

    def test_null_tracker_stage_propagates_exceptions(self) -> None:
        """stage() context manager doesn't suppress exceptions."""
        from kfp_components.components.training.autorag.shared.component_status import (
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
        with pytest.raises(RuntimeError, match="boom"):
            with status.stage("test_stage"):
                raise RuntimeError("boom")


class TestComponentStatusTrackerStage:
    """Additional stage() behaviour tests."""

    def test_stage_marks_failed_on_base_exception_subclass(self, tmp_path: Path) -> None:
        """stage() records failed when a BaseException subclass escapes the block."""
        with pytest.raises(KeyboardInterrupt):
            with ComponentStatusTracker(str(tmp_path), "text_extraction") as tracker:
                with tracker.stage("extract_documents"):
                    raise KeyboardInterrupt

        data = load_component_status(str(tmp_path))
        assert data["stages"][-1]["status"]["state"] == "failed"
        assert data["stages"][-1]["error"] == "KeyboardInterrupt"


class TestLoadComponentStatus:
    """Tests for load_component_status edge cases."""

    def test_corrupt_json_returns_empty_dict(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt status files return {} and log a warning."""
        status_file = tmp_path / COMPONENT_STATUS_FILENAME
        status_file.write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert load_component_status(str(tmp_path)) == {}
        assert "Failed to load status" in caplog.text


class TestSchemaCompliance:
    """Validate tracker output against the canonical JSON Schema."""

    def test_completed_documents_discovery(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Completed documents_discovery output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.set_metadata(display_name="Documents Discovery Status")
        tracker.record("load_benchmark", "started")
        tracker.record("load_benchmark", "completed")
        tracker.record("discover_documents", "started")
        tracker.record("discover_documents", "completed")

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_completed_text_extraction(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Completed text_extraction output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "text_extraction")
        tracker.set_metadata(display_name="Text Extraction Status")
        tracker.record("extract_documents", "started")
        tracker.record("extract_documents", "completed")

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_completed_search_space_preparation(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Completed search_space_preparation output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "search_space_preparation")
        tracker.set_metadata(display_name="Search Space Preparation Status")
        tracker.record("prepare_search_space", "started")
        tracker.record("prepare_search_space", "completed")

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_completed_models_pre_selector(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Completed models_pre_selector output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "models_pre_selector")
        tracker.set_metadata(display_name="Model Pre-Selection Status")
        tracker.record("model_pre_selection", "started")
        tracker.record("model_pre_selection", "completed")

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_completed_rag_templates_optimization(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Completed rag_templates_optimization output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        tracker.set_metadata(display_name="RAG Templates Optimization Status")
        tracker.record("optimize_templates", "started")
        tracker.record(
            "optimize_templates",
            "completed",
            metrics={
                "max_rag_patterns": 5,
                "selected_patterns": ["pattern_a", "pattern_b"],
            },
        )
        tracker.record("build_leaderboard", "started")
        tracker.record("build_leaderboard", "completed")

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_running_with_step(self, tmp_path: Path, canonical_schema: dict) -> None:
        """In-progress stage with step and message validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "rag_templates_optimization")
        tracker.set_metadata(display_name="RAG Templates Optimization Status")
        tracker.record(
            "optimize_templates",
            "running",
            step="embedding",
            message={"level": "info", "text": "Embedding documents..."},
        )

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_failed_stage(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Failed component output validates against the schema."""
        tracker = ComponentStatusTracker(str(tmp_path), "text_extraction")
        tracker.set_metadata(display_name="Text Extraction Status")
        tracker.record("extract_documents", "started")
        tracker.record("extract_documents", "failed", error="ValueError: unsupported format")

        _assert_schema_valid(_save_and_load(tracker), canonical_schema)

    def test_context_manager_failure(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Output from context manager exception path validates against the schema."""
        with pytest.raises(RuntimeError):
            with ComponentStatusTracker(str(tmp_path), "documents_discovery") as status:
                status.set_metadata(display_name="Documents Discovery Status")
                status.record("load_benchmark", "started")
                raise RuntimeError("connection lost")

        data = load_component_status(str(tmp_path))
        _assert_schema_valid(data, canonical_schema)

    def test_flat_keys_rejected_by_schema(self, tmp_path: Path, canonical_schema: dict) -> None:
        """Manually injected flat keys on a stage fail schema validation."""
        from jsonschema import ValidationError, validate

        tracker = ComponentStatusTracker(str(tmp_path), "documents_discovery")
        tracker.set_metadata(display_name="Documents Discovery Status")
        tracker.record("load_benchmark", "completed")

        data = _save_and_load(tracker)
        data["stages"][0]["rows"] = 100

        with pytest.raises(ValidationError, match="Additional properties are not allowed"):
            validate(instance=data, schema=canonical_schema)
