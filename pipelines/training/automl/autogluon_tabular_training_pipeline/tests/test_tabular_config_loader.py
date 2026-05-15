"""Tests for tabular integration test config JSON loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import test_configs


def _minimal_tabular_entry(**overrides) -> dict:
    base = {
        "id": "cfg-1",
        "dataset_path": "data/regression.csv",
        "label_column": "target",
        "problem_type": "regression",
        "task_type": "regression",
        "automl_settings": {},
        "tags": [],
    }
    base.update(overrides)
    return base


def test_load_configs_rejects_null_label_column(tmp_path: Path) -> None:
    """JSON null label_column must not become pipeline arguments."""
    bad = tmp_path / "configs.json"
    bad.write_text(
        json.dumps([_minimal_tabular_entry(label_column=None)]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"test_configs\.json\[0\] 'label_column'"):
        test_configs._load_configs(bad)


def test_load_configs_accepts_valid_file(tmp_path: Path) -> None:
    """Valid tabular config JSON loads into a single entry with expected fields."""
    path = tmp_path / "configs.json"
    path.write_text(json.dumps([_minimal_tabular_entry()]), encoding="utf-8")
    loaded = test_configs._load_configs(path)
    assert len(loaded) == 1
    assert loaded[0].label_column == "target"
