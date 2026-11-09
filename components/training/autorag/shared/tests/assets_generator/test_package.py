"""Installed-resource and dependency boundaries for the relocated package."""

import ast
import importlib.resources
import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

PACKAGE = "kfp_components.components.training.autorag.shared.assets_generator"


def test_all_resources_are_installed():
    """Every original notebook and nested starter-kit source ships in the wheel."""
    resources = importlib.resources.files(PACKAGE)
    expected = json.loads((Path(__file__).parent / "test_data/resource-manifest.json").read_text())
    for relative in expected:
        assert resources.joinpath(relative).is_file(), relative
    files = distribution("kfp-components").files
    assert not any(str(path).startswith("agentic_rag/") for path in files)
    assert not any(
        "__pycache__" in str(path) or "/.env" in str(path) for path in files if "assets_generator" in str(path)
    )


def test_import_requires_only_standard_library(tmp_path):
    """A fresh interpreter can import the installed package without ai4rag or KFP."""
    import kfp_components

    package_parent = str(Path(kfp_components.__file__).parent.parent)
    subprocess.run(
        [sys.executable, "-S", "-c", f"import sys; sys.path.insert(0, {package_parent!r}); import {PACKAGE}"],
        cwd=tmp_path,
        check=True,
    )


def test_generator_keeps_python_311_syntax():
    """Compilation and package imports remain available in normal component CI."""
    resources = importlib.resources.files(PACKAGE)
    for name in ("__init__.py", "notebook.py", "templates.py", "leaderboard.py", "starter_kit.py"):
        ast.parse(resources.joinpath(name).read_text(), feature_version=(3, 11))
