"""Tests for workspace run status helpers."""

import json
from pathlib import Path

from ..run_status import (
    COMPONENT_DATA_LOADER,
    COMPONENT_LEADERBOARD,
    COMPONENT_MODELS_TRAINING,
    DOCUMENT_PIPELINE_ID_FIELD,
    INITIAL_DOCUMENT_TEMPLATE,
    PIPELINE_TABULAR_TRAINING,
    RUN_STATUS_ARTIFACT_FILENAME,
    RUN_STATUS_REL_PATH,
    RunStatusRecorder,
    begin_component,
    complete_component,
    expected_stage_ids,
    init_run_status,
    load_component_stage_catalog,
    load_pipeline_run_status_manifest,
    load_run_status,
    load_run_status_template,
    pipeline_component_ids,
    publish_run_status_artifact,
    record_stage,
    run_status_file_path,
    templates_dir,
    validate_component_stages,
)


def test_templates_dir_and_pipeline_manifest():
    root = templates_dir()
    assert root.is_dir()
    assert (root / INITIAL_DOCUMENT_TEMPLATE).is_file()
    manifest_path = root / "pipelines" / f"{PIPELINE_TABULAR_TRAINING}.json"
    assert manifest_path.is_file()


def test_tabular_pipeline_manifest_covers_all_components():
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
        assert expected_stage_ids(component, pipeline_id=PIPELINE_TABULAR_TRAINING) == [
            s["id"] for s in catalog["stages"]
        ]


def test_load_document_initial_template():
    doc = load_run_status_template(INITIAL_DOCUMENT_TEMPLATE)
    assert doc["components"] == {}
    assert doc["run_status_rel_path"] == RUN_STATUS_REL_PATH


def test_init_and_stages(tmp_path):
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

    path = run_status_file_path(ws)
    assert path.exists()
    doc = json.loads(path.read_text())
    assert doc["kfp_run_id"] == "run-1"
    assert doc["pipeline_name"] == "tabular-job-abc"
    assert doc[DOCUMENT_PIPELINE_ID_FIELD] == PIPELINE_TABULAR_TRAINING
    assert doc["run_status_rel_path"] == RUN_STATUS_REL_PATH
    loader = doc["components"][COMPONENT_DATA_LOADER]
    assert loader["state"] == "completed"
    assert loader["stages"][0]["id"] == "read_and_sample"
    assert loader["stages"][0]["rows"] == 100


def test_run_status_recorder(tmp_path):
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
    assert doc["kfp_run_id"] == "run-2"
    assert doc["components"][COMPONENT_DATA_LOADER]["state"] == "completed"


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
    )
    begin_component(ws, COMPONENT_DATA_LOADER)
    complete_component(ws, COMPONENT_DATA_LOADER)

    doc = publish_run_status_artifact(artifact_dir, ws)
    assert doc["kfp_run_id"] == "run-1"
    artifact_file = Path(artifact_dir) / RUN_STATUS_ARTIFACT_FILENAME
    assert artifact_file.exists()
    assert json.loads(artifact_file.read_text())["components"][COMPONENT_DATA_LOADER]["state"] == "completed"


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
        validate_component_stages(document, COMPONENT_DATA_LOADER)
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
        validate_component_stages(document, COMPONENT_DATA_LOADER)
    assert "stages not in manifest" in caplog.text
