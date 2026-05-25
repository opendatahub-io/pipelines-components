"""Tests for workspace run status helpers."""

import importlib
import json
from pathlib import Path

# Load via importlib (not top-level ``from run_status import``) for import-guard compliance.
# ``shared/tests/conftest.py`` puts the embedded shared root on sys.path first.
_rs = importlib.import_module("run_status")

COMPONENT_DATA_LOADER = _rs.COMPONENT_DATA_LOADER
COMPONENT_LEADERBOARD = _rs.COMPONENT_LEADERBOARD
COMPONENT_MODELS_TRAINING = _rs.COMPONENT_MODELS_TRAINING
DOCUMENT_PIPELINE_ID_FIELD = _rs.DOCUMENT_PIPELINE_ID_FIELD
PIPELINE_TABULAR_TRAINING = _rs.PIPELINE_TABULAR_TRAINING
RUN_STATUS_ARTIFACT_FILENAME = _rs.RUN_STATUS_ARTIFACT_FILENAME
STATUS_COMPLETED = _rs.STATUS_COMPLETED
STATUS_PENDING = _rs.STATUS_PENDING
RunStatusRecorder = _rs.RunStatusRecorder
begin_component = _rs.begin_component
complete_component = _rs.complete_component
ensure_pipeline_plan = _rs.ensure_pipeline_plan
expected_stage_steps = _rs.expected_stage_steps
init_run_status = _rs.init_run_status
load_component_stage_catalog = _rs.load_component_stage_catalog
load_pipeline_run_status_manifest = _rs.load_pipeline_run_status_manifest
load_run_status = _rs.load_run_status
pipeline_component_ids = _rs.pipeline_component_ids
publish_run_status_artifact = _rs.publish_run_status_artifact
record_stage = _rs.record_stage
resolve_templates_dir = _rs.resolve_templates_dir
run_status_file_path = _rs.run_status_file_path
validate_component_stages = _rs.validate_component_stages

_SHARED_ROOT = str(Path(__file__).resolve().parents[1])


def test_pipeline_manifest_json_exists():
    """Tabular pipeline manifest JSON is present under the embedded templates tree."""
    manifest_path = resolve_templates_dir(_SHARED_ROOT) / "pipelines" / f"{PIPELINE_TABULAR_TRAINING}.json"
    assert manifest_path.is_file()


def test_tabular_pipeline_manifest_covers_all_components():
    """Manifest lists all three tabular pipeline components with at least one stage each."""
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
    """``init_run_status`` seeds every manifest component and stage as ``pending``."""
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
    """``ensure_pipeline_plan`` adds missing components without overwriting completed stages."""
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
    """``record_stage`` upserts a stage and stores optional ``steps`` on completion."""
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
    """``model_selection`` exposes optional steps; stages without steps return ``None``."""
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
    """Validation warns when a completed stage omits manifest ``steps``."""
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
    """Init, stage records, and complete update the workspace JSON as expected."""
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
    """``RunStatusRecorder`` wraps init, begin, record, complete, and publish."""
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
    """Missing run status file yields an empty document."""
    assert load_run_status(str(tmp_path)) == {}


def test_publish_run_status_artifact(tmp_path):
    """Publish copies workspace JSON into the artifact output path."""
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
    """Validation warns when manifest stage slots are missing from the document."""
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
    """Validation warns when recorded stage ids are not in the manifest."""
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
