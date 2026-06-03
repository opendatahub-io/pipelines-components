"""Tests for workspace run status helpers."""

import json
from pathlib import Path

import pytest
from kfp_components.components.training.automl.shared.run_status import (
    COMPONENT_DATA_LOADER,
    COMPONENT_LEADERBOARD,
    COMPONENT_MODELS_TRAINING,
    COMPONENT_TIMESERIES_DATA_LOADER,
    COMPONENT_TIMESERIES_MODELS_TRAINING,
    DOCUMENT_PIPELINE_ID_FIELD,
    PIPELINE_TABULAR_TRAINING,
    PIPELINE_TIMESERIES_TRAINING,
    RUN_STATUS_ARTIFACT_FILENAME,
    STATUS_COMPLETED,
    STATUS_PENDING,
    RunStatusRecorder,
    _get_component_entry,
    begin_component,
    complete_component,
    ensure_pipeline_plan,
    init_run_status,
    load_component_stage_catalog,
    load_pipeline_run_status_manifest,
    load_run_status,
    pipeline_component_ids,
    publish_run_status_artifact,
    record_stage,
    resolve_templates_dir,
    run_status_file_path,
    validate_component_stages,
)


def _component_by_id(document: dict, component_id: str) -> dict:
    entry = _get_component_entry(document, component_id)
    assert entry is not None, f"component {component_id} not found"
    return entry


def test_pipeline_manifest_json_exists():
    """Test that pipeline manifest JSON files exist."""
    templates = resolve_templates_dir() / "pipelines"
    assert (templates / f"{PIPELINE_TABULAR_TRAINING}.json").is_file()
    assert (templates / f"{PIPELINE_TIMESERIES_TRAINING}.json").is_file()


def test_tabular_pipeline_manifest_covers_all_components():
    """Test that tabular pipeline manifest covers all components."""
    manifest = load_pipeline_run_status_manifest(PIPELINE_TABULAR_TRAINING)
    assert manifest["pipeline_id"] == PIPELINE_TABULAR_TRAINING
    component_ids = pipeline_component_ids(PIPELINE_TABULAR_TRAINING)
    assert component_ids == [
        COMPONENT_DATA_LOADER,
        COMPONENT_MODELS_TRAINING,
        COMPONENT_LEADERBOARD,
    ]
    for component in (
        COMPONENT_DATA_LOADER,
        COMPONENT_MODELS_TRAINING,
        COMPONENT_LEADERBOARD,
    ):
        catalog = load_component_stage_catalog(component, pipeline_id=PIPELINE_TABULAR_TRAINING)
        assert catalog["id"] == component
        assert len(catalog["stages"]) >= 1


def test_timeseries_pipeline_manifest_covers_all_components():
    """Test that timeseries pipeline manifest covers all components."""
    manifest = load_pipeline_run_status_manifest(PIPELINE_TIMESERIES_TRAINING)
    assert manifest["pipeline_id"] == PIPELINE_TIMESERIES_TRAINING
    component_ids = pipeline_component_ids(PIPELINE_TIMESERIES_TRAINING)
    assert component_ids == [
        COMPONENT_TIMESERIES_DATA_LOADER,
        COMPONENT_TIMESERIES_MODELS_TRAINING,
        COMPONENT_LEADERBOARD,
    ]


def test_init_seeds_full_pipeline_as_pending(tmp_path):
    """Test that init seeds full pipeline as pending."""
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
    )
    doc = load_run_status(ws)
    assert [component["id"] for component in doc["components"]] == [
        COMPONENT_DATA_LOADER,
        COMPONENT_MODELS_TRAINING,
        COMPONENT_LEADERBOARD,
    ]
    assert _component_by_id(doc, COMPONENT_DATA_LOADER)["state"] == STATUS_PENDING
    assert _component_by_id(doc, COMPONENT_MODELS_TRAINING)["state"] == STATUS_PENDING
    loader_stages = {s["id"]: s["status"] for s in _component_by_id(doc, COMPONENT_DATA_LOADER)["stages"]}
    assert loader_stages == {
        "validate_inputs": STATUS_PENDING,
        "read_and_sample": STATUS_PENDING,
        "cleanse": STATUS_PENDING,
        "split": STATUS_PENDING,
        "write_outputs": STATUS_PENDING,
    }


def test_init_copies_catalog_metadata_from_manifest(tmp_path):
    """Init should expose list order, descriptions, and stage steps for dashboards."""
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
    )
    doc = load_run_status(ws)
    loader = _component_by_id(doc, COMPONENT_DATA_LOADER)
    assert "order" not in loader
    assert "description" in loader
    validate_inputs = next(s for s in loader["stages"] if s["id"] == "validate_inputs")
    assert validate_inputs["status"] == STATUS_PENDING
    assert "description" in validate_inputs
    assert "steps" not in validate_inputs

    training = _component_by_id(doc, COMPONENT_MODELS_TRAINING)
    model_selection = next(s for s in training["stages"] if s["id"] == "model_selection")
    assert model_selection["steps"] == [
        "feature_engineering",
        "model_training",
        "stacking",
        "model_evaluation",
    ]


def test_ensure_pipeline_plan_preserves_progress(tmp_path):
    """Test that ensure_pipeline_plan preserves progress."""
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
    )
    record_stage(ws, COMPONENT_DATA_LOADER, "validate_inputs", STATUS_COMPLETED)
    ensure_pipeline_plan(ws)
    doc = load_run_status(ws)
    assert _component_by_id(doc, COMPONENT_MODELS_TRAINING)["state"] == STATUS_PENDING
    assert _component_by_id(doc, COMPONENT_DATA_LOADER)["stages"][0]["status"] == STATUS_COMPLETED


def test_record_stage_autofills_steps_from_manifest_on_completed(tmp_path):
    """``completed`` stages copy ``steps`` from the pipeline manifest when defined."""
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
    )
    begin_component(ws, COMPONENT_MODELS_TRAINING)
    record_stage(
        ws,
        COMPONENT_MODELS_TRAINING,
        "model_selection",
        STATUS_COMPLETED,
        top_n=2,
    )
    training = _component_by_id(load_run_status(ws), COMPONENT_MODELS_TRAINING)
    model_selection = next(s for s in training["stages"] if s["id"] == "model_selection")
    assert model_selection["steps"] == [
        "feature_engineering",
        "model_training",
        "stacking",
        "model_evaluation",
    ]
    assert model_selection["top_n"] == 2
    assert "description" in model_selection
    record_stage(ws, COMPONENT_DATA_LOADER, "validate_inputs", STATUS_COMPLETED)
    loader_stage = next(
        s
        for s in _component_by_id(load_run_status(ws), COMPONENT_DATA_LOADER)["stages"]
        if s["id"] == "validate_inputs"
    )
    assert "steps" not in loader_stage


def test_init_and_stages(tmp_path):
    """Test init and stage operations."""
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="tabular-job-abc",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
    )
    begin_component(ws, COMPONENT_DATA_LOADER)
    record_stage(ws, COMPONENT_DATA_LOADER, "read_and_sample", "completed", rows=100)
    complete_component(ws, COMPONENT_DATA_LOADER)

    doc = json.loads(run_status_file_path(ws).read_text())
    assert doc["kfp_run_id"] == "run-1"
    assert doc[DOCUMENT_PIPELINE_ID_FIELD] == PIPELINE_TABULAR_TRAINING
    assert _component_by_id(doc, COMPONENT_DATA_LOADER)["state"] == STATUS_COMPLETED
    assert _component_by_id(doc, COMPONENT_MODELS_TRAINING)["state"] == STATUS_PENDING
    read_stage = next(s for s in _component_by_id(doc, COMPONENT_DATA_LOADER)["stages"] if s["id"] == "read_and_sample")
    assert read_stage["rows"] == 100


def test_run_status_recorder(tmp_path):
    """Test RunStatusRecorder class."""
    ws = str(tmp_path)
    RunStatusRecorder.init_pipeline_run(
        ws,
        kfp_run_id="run-2",
        pipeline_name="p2",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
    )
    recorder = RunStatusRecorder(ws, COMPONENT_DATA_LOADER)
    recorder.begin()
    recorder.record("validate_inputs", "completed")
    recorder.complete()
    doc = recorder.publish_artifact(str(tmp_path / "artifact"))
    assert _component_by_id(doc, COMPONENT_DATA_LOADER)["state"] == STATUS_COMPLETED
    assert _component_by_id(doc, COMPONENT_LEADERBOARD)["state"] == STATUS_PENDING


def test_load_empty_returns_empty_dict(tmp_path):
    """Test that load_run_status returns empty dict when no file exists."""
    assert load_run_status(str(tmp_path)) == {}


def test_publish_run_status_artifact(tmp_path):
    """Test that run_status artifact is published correctly."""
    ws = str(tmp_path / "ws")
    artifact_dir = str(tmp_path / "artifact")
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
    )
    begin_component(ws, COMPONENT_DATA_LOADER)
    complete_component(ws, COMPONENT_DATA_LOADER)

    doc = publish_run_status_artifact(artifact_dir, ws)
    assert doc["kfp_run_id"] == "run-1"
    assert (Path(artifact_dir) / RUN_STATUS_ARTIFACT_FILENAME).exists()


def test_validate_component_stages_warns_on_missing(caplog):
    """Test that validate_component_stages warns on missing stages."""
    document = {
        DOCUMENT_PIPELINE_ID_FIELD: PIPELINE_TABULAR_TRAINING,
        "components": [
            {
                "id": COMPONENT_DATA_LOADER,
                "stages": [{"id": "validate_inputs", "status": "completed"}],
            }
        ],
    }
    with caplog.at_level("WARNING"):
        validate_component_stages(document, COMPONENT_DATA_LOADER)
    assert "missing manifest stages" in caplog.text


def test_validate_component_stages_warns_on_unknown(caplog):
    """Test that validate_component_stages warns on unknown stages."""
    document = {
        DOCUMENT_PIPELINE_ID_FIELD: PIPELINE_TABULAR_TRAINING,
        "components": [
            {
                "id": COMPONENT_DATA_LOADER,
                "stages": [{"id": "not_in_catalog", "status": "completed"}],
            }
        ],
    }
    with caplog.at_level("WARNING"):
        validate_component_stages(document, COMPONENT_DATA_LOADER)
    assert "stages not in manifest" in caplog.text


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
