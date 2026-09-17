"""Helpers for generating per-run AutoML experiment launcher notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kfp_components.components.training.automl.shared.run_status import shared_automl_dir

EXPERIMENT_NOTEBOOK_FILENAME = "automl_experiment_notebook.ipynb"
EXPERIMENT_NOTEBOOK_RELATIVE_PATH = EXPERIMENT_NOTEBOOK_FILENAME


def _py_str(value: str) -> str:
    """Return a safe Python string literal for notebook code cells."""
    return json.dumps(value)


def _py_list(values: list[str] | None) -> str:
    """Return a safe Python list literal for notebook code cells."""
    return json.dumps(values or [])


def include_user_test_data_in_notebook(test_data_bucket_name: str, test_data_file_key: str) -> bool:
    """Return True when the run used a user-provided external test dataset."""
    bucket = (test_data_bucket_name or "").strip()
    key = (test_data_file_key or "").strip()
    return bool(bucket and key)


def _strip_user_test_data_from_source(source: list[str]) -> list[str]:
    """Remove optional test-data config and pipeline-argument lines."""
    result: list[str] = []
    i = 0
    while i < len(source):
        line = source[i]
        if line.lstrip().startswith("# Optional user-provided test dataset"):
            if result and not result[-1].strip():
                result.pop()
            i += 1
            while i < len(source) and ("test_data_bucket_name" in source[i] or "test_data_file_key" in source[i]):
                i += 1
            if i < len(source) and not source[i].strip():
                i += 1
            continue
        if '"test_data_bucket_name"' in line and "test_data_bucket_name" in line:
            i += 1
            continue
        if '"test_data_file_key"' in line and "test_data_file_key" in line:
            i += 1
            continue
        result.append(line)
        i += 1
    return result


def _strip_user_test_data_from_notebook(notebook: dict) -> dict:
    """Remove user test-data sections from generated experiment notebooks."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["source"] = _strip_user_test_data_from_source(cell.get("source", []))
    return notebook


def _strip_empty_positive_class_from_notebook(notebook: dict) -> dict:
    """Remove the optional tabular positive-class input from a generated notebook."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["source"] = [
            line
            for line in cell.get("source", [])
            if not line.lstrip().startswith("positive_class =") and '"positive_class": positive_class' not in line
        ]
    return notebook


def replace_placeholder_in_notebook(notebook: dict, replacements: dict[str, str]) -> dict:
    """Replace placeholder tokens in code cell sources."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        new_source = []
        for line in cell.get("source", []):
            for placeholder, value in replacements.items():
                line = line.replace(placeholder, value)
            new_source.append(line)
        cell["source"] = new_source
    return notebook


def _template_name(kind: Literal["tabular", "timeseries"]) -> str:
    return f"{kind}_experiment_notebook.ipynb"


@dataclass(frozen=True)
class TabularExperimentNotebookConfig:  # pylint: disable=too-many-instance-attributes
    """Pipeline parameters to pre-fill the tabular experiment notebook template."""

    train_data_secret_name: str
    train_data_bucket_name: str
    train_data_file_key: str
    test_data_bucket_name: str
    test_data_file_key: str
    label_column: str
    task_type: str
    top_n: int
    positive_class: str
    eval_metric: str
    preset: str

    @property
    def include_user_test_data(self) -> bool:
        """Return True when user-provided external test data was configured."""
        return include_user_test_data_in_notebook(
            self.test_data_bucket_name,
            self.test_data_file_key,
        )


@dataclass(frozen=True)
class TimeseriesExperimentNotebookConfig:  # pylint: disable=too-many-instance-attributes
    """Pipeline parameters to pre-fill the time series experiment notebook template."""

    train_data_secret_name: str
    train_data_bucket_name: str
    train_data_file_key: str
    test_data_bucket_name: str
    test_data_file_key: str
    target: str
    id_column: str
    timestamp_column: str
    known_covariates_names: list[str] | None
    prediction_length: int
    top_n: int
    eval_metric: str
    preset: str

    @property
    def include_user_test_data(self) -> bool:
        """Return True when user-provided external test data was configured."""
        return include_user_test_data_in_notebook(
            self.test_data_bucket_name,
            self.test_data_file_key,
        )


def tabular_experiment_notebook_replacements(
    config: TabularExperimentNotebookConfig,
) -> dict[str, str]:
    """Build placeholder replacements for the tabular experiment notebook template."""
    return {
        "<REPLACE_S3_SECRET>": _py_str(config.train_data_secret_name),
        "<REPLACE_DATA_BUCKET>": _py_str(config.train_data_bucket_name),
        "<REPLACE_DATA_FILE_KEY>": _py_str(config.train_data_file_key),
        "<REPLACE_TEST_DATA_BUCKET>": _py_str(config.test_data_bucket_name),
        "<REPLACE_TEST_DATA_FILE_KEY>": _py_str(config.test_data_file_key),
        "<REPLACE_LABEL_COLUMN>": _py_str(config.label_column),
        "<REPLACE_TASK_TYPE>": _py_str(config.task_type),
        "<REPLACE_TOP_N>": str(config.top_n),
        "<REPLACE_POSITIVE_CLASS>": _py_str(config.positive_class),
        "<REPLACE_EVAL_METRIC>": _py_str(config.eval_metric),
        "<REPLACE_PRESET>": _py_str(config.preset),
    }


def timeseries_experiment_notebook_replacements(
    config: TimeseriesExperimentNotebookConfig,
) -> dict[str, str]:
    """Build placeholder replacements for the timeseries experiment notebook template."""
    return {
        "<REPLACE_S3_SECRET>": _py_str(config.train_data_secret_name),
        "<REPLACE_DATA_BUCKET>": _py_str(config.train_data_bucket_name),
        "<REPLACE_DATA_FILE_KEY>": _py_str(config.train_data_file_key),
        "<REPLACE_TEST_DATA_BUCKET>": _py_str(config.test_data_bucket_name),
        "<REPLACE_TEST_DATA_FILE_KEY>": _py_str(config.test_data_file_key),
        "<REPLACE_TARGET>": _py_str(config.target),
        "<REPLACE_ID_COLUMN>": _py_str(config.id_column),
        "<REPLACE_TIMESTAMP_COLUMN>": _py_str(config.timestamp_column),
        "<REPLACE_KNOWN_COVARIATES_NAMES>": _py_list(config.known_covariates_names),
        "<REPLACE_PREDICTION_LENGTH>": str(config.prediction_length),
        "<REPLACE_TOP_N>": str(config.top_n),
        "<REPLACE_EVAL_METRIC>": _py_str(config.eval_metric),
        "<REPLACE_PRESET>": _py_str(config.preset),
    }


def write_experiment_notebook(
    *,
    output_dir: Path,
    kind: Literal["tabular", "timeseries"],
    replacements: dict[str, str],
    include_user_test_data: bool = False,
) -> Path:
    """Write a run-level experiment launcher notebook under ``output_dir``."""
    template_path = shared_automl_dir() / "notebook_templates" / _template_name(kind)
    with template_path.open(encoding="utf-8") as f:
        notebook = json.load(f)

    notebook = replace_placeholder_in_notebook(notebook, replacements)
    if not include_user_test_data:
        notebook = _strip_user_test_data_from_notebook(notebook)
    if kind == "tabular" and replacements.get("<REPLACE_POSITIVE_CLASS>") == _py_str(""):
        notebook = _strip_empty_positive_class_from_notebook(notebook)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path / EXPERIMENT_NOTEBOOK_FILENAME
    with destination.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        f.write("\n")
    return destination
