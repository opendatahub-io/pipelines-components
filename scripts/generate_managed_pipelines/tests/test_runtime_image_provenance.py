"""Verify image overrides reach MLflow through every compiled training task."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("pipeline_module", "pipeline_name", "executor_prefix", "image_env", "count"),
    [
        (
            "automl.autogluon_tabular_training_pipeline",
            "autogluon_tabular_training_pipeline",
            "exec-autogluon-models-training",
            "RELATED_IMAGE_ODH_AUTOML_IMAGE",
            3,
        ),
        (
            "automl.autogluon_timeseries_training_pipeline",
            "autogluon_timeseries_training_pipeline",
            "exec-autogluon-timeseries-models-training",
            "RELATED_IMAGE_ODH_AUTOML_IMAGE",
            3,
        ),
        (
            "autorag.documents_rag_optimization_pipeline",
            "documents_rag_optimization_pipeline",
            "exec-rag-templates-optimization",
            "RELATED_IMAGE_ODH_AUTORAG_IMAGE",
            1,
        ),
    ],
)
@pytest.mark.parametrize("image", ["quay.io/example/runtime:custom", "quay.io/example/runtime@sha256:" + "c" * 64])
def test_compiled_training_image_matches_provenance(
    tmp_path, pipeline_module, pipeline_name, executor_prefix, image_env, count, image
):
    """Capture the same image for execution and logging, across all presets."""
    output = tmp_path / "pipeline.yaml"
    env = {**os.environ, image_env: image}
    # Constants and component decorators resolve image overrides at import time.
    # A fresh interpreter reproduces managed-pipeline compilation without cached imports.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; from kfp import compiler; "
            "pipeline = getattr(importlib.import_module(sys.argv[1]), sys.argv[2]); "
            "compiler.Compiler().compile(pipeline, sys.argv[3])",
            f"kfp_components.pipelines.training.{pipeline_module}.pipeline",
            pipeline_name,
            str(output),
        ],
        env=env,
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    documents = list(yaml.safe_load_all(output.read_text()))
    spec = next(doc for doc in documents if "deploymentSpec" in doc)
    containers = [
        executor["container"]
        for name, executor in spec["deploymentSpec"]["executors"].items()
        if name.startswith(executor_prefix)
    ]
    assert len(containers) == count
    for container in containers:
        assert container["image"] == image
        assert {item["name"]: item["value"] for item in container.get("env", [])}[image_env] == image
