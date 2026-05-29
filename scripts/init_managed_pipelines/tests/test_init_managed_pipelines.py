"""Unit tests for the init-managed-pipelines container entrypoint."""

import json
from pathlib import Path

import pytest

from scripts.generate_managed_pipelines.generate_managed_pipelines import (
    OUTPUT_FILENAME,
    RELATED_IMAGE_ENV_PREFIX,
)

from .. import init_managed_pipelines as init_mod


def _clear_related_image_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all RELATED_IMAGE_* variables from the environment."""
    import os

    for key in list(os.environ):
        if key.startswith(RELATED_IMAGE_ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)


def _write_image_fixture(app_root: Path) -> None:
    """Simulate build-time artifacts under ``/app``."""
    pipe_dir = app_root / "pipelines" / "training" / "p"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "pipeline.py").write_text("from kfp import dsl\n\n@dsl.pipeline\ndef p():\n    pass\n")
    (pipe_dir / "pipeline.yaml").write_text(
        "deploymentSpec:\n  executors:\n    exec:\n      container:\n        image: quay.io/build-time:tag\n",
        encoding="utf-8",
    )
    manifest = [
        {
            "name": "my_pipeline",
            "description": "",
            "path": "pipelines/training/p/pipeline.py",
            "stability": "Development Preview",
        },
    ]
    (app_root / OUTPUT_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def staging_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point init module at temporary app root and staging volume."""
    app_root = tmp_path / "app"
    output_dir = tmp_path / "config" / "managed-pipelines"
    app_root.mkdir()
    monkeypatch.setattr(init_mod, "APP_ROOT", app_root)
    monkeypatch.setattr(init_mod, "OUTPUT_DIR", output_dir)
    return app_root, output_dir


def test_main_copy_path_without_related_image(
    staging_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without RELATED_IMAGE env, init copies pre-built YAML from the image."""
    app_root, output_dir = staging_paths
    _write_image_fixture(app_root)
    _clear_related_image_env(monkeypatch)

    assert init_mod.main() == 0

    staged = output_dir / "my_pipeline.yaml"
    assert staged.is_file()
    assert "quay.io/build-time:tag" in staged.read_text()
    assert (output_dir / OUTPUT_FILENAME).is_file()
    assert json.loads((output_dir / OUTPUT_FILENAME).read_text())[0]["name"] == "my_pipeline"


def test_main_recompile_path_calls_stage_managed_pipelines(
    staging_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With RELATED_IMAGE env, init stages via stage_managed_pipelines (not image copy)."""
    app_root, output_dir = staging_paths
    _write_image_fixture(app_root)
    monkeypatch.setenv("RELATED_IMAGE_ODH_AUTOML_IMAGE", "quay.io/example/automl-runtime@sha256:aaa")

    calls: list[tuple[Path, Path]] = []

    def fake_stage(repo_root: Path, out_dir: Path) -> int:
        calls.append((repo_root, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = [{"name": "my_pipeline", "description": "", "path": "p", "stability": "Development Preview"}]
        (out_dir / OUTPUT_FILENAME).write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        (out_dir / "my_pipeline.yaml").write_text(
            "deploymentSpec:\n  executors:\n    exec:\n      container:\n"
            "        image: quay.io/example/automl-runtime@sha256:aaa\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(init_mod, "stage_managed_pipelines", fake_stage)

    assert init_mod.main() == 0
    assert calls == [(app_root, output_dir)]
    assert "quay.io/example/automl-runtime@sha256:aaa" in (output_dir / "my_pipeline.yaml").read_text()
    assert "quay.io/build-time:tag" not in (output_dir / "my_pipeline.yaml").read_text()


def test_main_returns_nonzero_when_stage_fails(
    staging_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Init propagates failure from stage_managed_pipelines."""
    app_root, output_dir = staging_paths
    _write_image_fixture(app_root)
    monkeypatch.setenv("RELATED_IMAGE_ODH_AUTOML_IMAGE", "quay.io/example/automl-runtime@sha256:aaa")
    monkeypatch.setattr(init_mod, "stage_managed_pipelines", lambda _repo, _out: 1)

    assert init_mod.main() == 1
    assert not (output_dir / "my_pipeline.yaml").exists()
