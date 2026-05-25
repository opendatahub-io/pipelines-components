"""Tests for workspace run status helpers."""

import json
from pathlib import Path

from run_status import (
    COMPONENT_DATA_LOADER,
    COMPONENT_LEADERBOARD,
    COMPONENT_MODELS_TRAINING,
    DOCUMENT_PIPELINE_ID_FIELD,
    PIPELINE_TABULAR_TRAINING,
    RUN_STATUS_ARTIFACT_FILENAME,
    RUN_STATUS_REL_PATH,
    STATUS_COMPLETED,
    STATUS_PENDING,
    STATUS_RUNNING,
    RunStatusRecorder,
    begin_component,
    complete_component,
    ensure_pipeline_plan,
    expected_stage_steps,
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

_SHARED_ROOT = str(Path(__file__).resolve().parents[1])


def test_pipeline_manifest_json_exists():
    manifest_path = resolve_templates_dir(_SHARED_ROOT) / "pipelines" / f"{PIPELINE_TABULAR_TRAINING}.json"
    assert manifest_path.is_file()


def test_tabular_pipeline_manifest_covers_all_components():
    manifest = load_pipeline_run_status_manifest(PIPELINE_TABULAR_TRAINING, templates_root=_SHARED_ROOT)
    assert manifest["pipeline_id"] == PIPELINE_TABULAR_TRAINING
    component_ids = pipeline_component_ids(PIPELINE_TABULAR_TRAINING, templates_root=_SHARED_ROOT)
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
        catalog = load_component_stage_catalog(
            component, pipeline_id=PIPELINE_TABULAR_TRAINING, templates_root=_SHARED_ROOT
        )
        assert catalog["id"] == component
        assert len(catalog["stages"]) >= 1


def test_init_seeds_full_pipeline_as_pending(tmp_path):
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
        templates_root=_SHARED_ROOT,
    )
    doc = load_run_status(ws)
    assert set(doc["components"]) == {
        COMPONENT_DATA_LOADER,
        COMPONENT_MODELS_TRAINING,
        COMPONENT_LEADERBOARD,
    }
    assert doc["components"][COMPONENT_DATA_LOADER]["state"] == STATUS_PENDING
    assert doc["components"][COMPONENT_MODELS_TRAINING]["state"] == STATUS_PENDING
    loader_stages = {s["id"]: s["status"] for s in doc["components"][COMPONENT_DATA_LOADER]["stages"]}
    assert loader_stages == {
        "validate_inputs": STATUS_PENDING,
        "read_and_sample": STATUS_PENDING,
        "cleanse": STATUS_PENDING,
        "split": STATUS_PENDING,
        "write_outputs": STATUS_PENDING,
    }


def test_ensure_pipeline_plan_preserves_progress(tmp_path):
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
        templates_root=_SHARED_ROOT,
    )
    record_stage(
        ws,
        COMPONENT_DATA_LOADER,
        "validate_inputs",
        STATUS_COMPLETED,
        templates_root=_SHARED_ROOT,
    )
    ensure_pipeline_plan(ws, templates_root=_SHARED_ROOT)
    doc = load_run_status(ws)
    assert doc["components"][COMPONENT_MODELS_TRAINING]["state"] == STATUS_PENDING
    assert doc["components"][COMPONENT_DATA_LOADER]["stages"][0]["status"] == STATUS_COMPLETED


def test_record_stage_with_optional_steps(tmp_path):
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
        templates_root=_SHARED_ROOT,
    )
    begin_component(ws, COMPONENT_MODELS_TRAINING, templates_root=_SHARED_ROOT)
    record_stage(
        ws,
        COMPONENT_MODELS_TRAINING,
        "model_selection",
        "completed",
        steps=["feature_engineering", "model_training", "stacking", "model_evaluation"],
        top_n=2,
    )
    training = load_run_status(ws)["components"][COMPONENT_MODELS_TRAINING]
    model_selection = next(s for s in training["stages"] if s["id"] == "model_selection")
    assert len(training["stages"]) == 4
    assert model_selection["steps"] == [
        "feature_engineering",
        "model_training",
        "stacking",
        "model_evaluation",
    ]
    assert model_selection["top_n"] == 2


def test_expected_stage_steps_from_manifest():
    steps = expected_stage_steps(
        COMPONENT_MODELS_TRAINING,
        "model_selection",
        pipeline_id=PIPELINE_TABULAR_TRAINING,
        templates_root=_SHARED_ROOT,
    )
    assert steps == [
        "feature_engineering",
        "model_training",
        "stacking",
        "model_evaluation",
    ]
    assert (
        expected_stage_steps(
            COMPONENT_DATA_LOADER,
            "validate_inputs",
            pipeline_id=PIPELINE_TABULAR_TRAINING,
            templates_root=_SHARED_ROOT,
        )
        is None
    )


def test_validate_component_stages_warns_on_missing_steps(caplog):
    document = {
        DOCUMENT_PIPELINE_ID_FIELD: PIPELINE_TABULAR_TRAINING,
        "components": {
            COMPONENT_MODELS_TRAINING: {
                "stages": [
                    {
                        "id": "model_selection",
                        "status": "completed",
                        "timestamp": "2026-01-01T00:00:00Z",
                    }
                ],
            }
        },
    }
    with caplog.at_level("WARNING"):
        validate_component_stages(document, COMPONENT_MODELS_TRAINING, templates_root=_SHARED_ROOT)
    assert "missing steps" in caplog.text


def test_init_and_stages(tmp_path):
    ws = str(tmp_path)
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="tabular-job-abc",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
        templates_root=_SHARED_ROOT,
    )
    begin_component(ws, COMPONENT_DATA_LOADER, templates_root=_SHARED_ROOT)
    record_stage(ws, COMPONENT_DATA_LOADER, "read_and_sample", "completed", rows=100)
    complete_component(ws, COMPONENT_DATA_LOADER)

    doc = json.loads(run_status_file_path(ws).read_text())
    assert doc["kfp_run_id"] == "run-1"
    assert doc[DOCUMENT_PIPELINE_ID_FIELD] == PIPELINE_TABULAR_TRAINING
    assert doc["components"][COMPONENT_DATA_LOADER]["state"] == STATUS_COMPLETED
    assert doc["components"][COMPONENT_MODELS_TRAINING]["state"] == STATUS_PENDING
    read_stage = next(s for s in doc["components"][COMPONENT_DATA_LOADER]["stages"] if s["id"] == "read_and_sample")
    assert read_stage["rows"] == 100


def test_run_status_recorder(tmp_path):
    ws = str(tmp_path)
    RunStatusRecorder.init_pipeline_run(
        ws,
        kfp_run_id="run-2",
        pipeline_name="p2",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
        templates_root=_SHARED_ROOT,
    )
    recorder = RunStatusRecorder(ws, COMPONENT_DATA_LOADER, templates_root=_SHARED_ROOT)
    recorder.begin()
    recorder.record("validate_inputs", "completed")
    recorder.complete()
    doc = recorder.publish_artifact(str(tmp_path / "artifact"))
    assert doc["components"][COMPONENT_DATA_LOADER]["state"] == STATUS_COMPLETED
    assert doc["components"][COMPONENT_LEADERBOARD]["state"] == STATUS_PENDING


def test_load_empty_returns_empty_dict(tmp_path):
    assert load_run_status(str(tmp_path)) == {}


def test_publish_run_status_artifact(tmp_path):
    ws = str(tmp_path / "ws")
    artifact_dir = str(tmp_path / "artifact")
    init_run_status(
        ws,
        kfp_run_id="run-1",
        pipeline_name="p1",
        run_status_pipeline_id=PIPELINE_TABULAR_TRAINING,
        templates_root=_SHARED_ROOT,
    )
    begin_component(ws, COMPONENT_DATA_LOADER, templates_root=_SHARED_ROOT)
    complete_component(ws, COMPONENT_DATA_LOADER)

    doc = publish_run_status_artifact(artifact_dir, ws)
    assert doc["kfp_run_id"] == "run-1"
    assert (Path(artifact_dir) / RUN_STATUS_ARTIFACT_FILENAME).exists()


def test_validate_component_stages_warns_on_missing(caplog):
    document = {
        DOCUMENT_PIPELINE_ID_FIELD: PIPELINE_TABULAR_TRAINING,
        "components": {
            COMPONENT_DATA_LOADER: {
                "stages": [{"id": "validate_inputs", "status": "completed"}],
            }
        },
    }
    with caplog.at_level("WARNING"):
        validate_component_stages(document, COMPONENT_DATA_LOADER, templates_root=_SHARED_ROOT)
    assert "missing manifest stages" in caplog.text


def test_validate_component_stages_warns_on_unknown(caplog):
    document = {
        DOCUMENT_PIPELINE_ID_FIELD: PIPELINE_TABULAR_TRAINING,
        "components": {
            COMPONENT_DATA_LOADER: {
                "stages": [{"id": "not_in_catalog", "status": "completed"}],
            }
        },
    }
    with caplog.at_level("WARNING"):
        validate_component_stages(document, COMPONENT_DATA_LOADER, templates_root=_SHARED_ROOT)
    assert "stages not in manifest" in caplog.text
