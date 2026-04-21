# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Consider also @AGENTS.md as a reference about the project.

## Project Overview

Kubeflow Pipelines Components Repository — a centralized hub for reusable KFP (Kubeflow Pipelines) components and pipelines for ML workflows. Package name: `kfp-components`. Python >=3.11.

Components are `@dsl.component`-decorated functions under `components/<category>/[<subcategory>/]<name>/`.
Pipelines are `@dsl.pipeline`-decorated functions under `pipelines/<category>/[<subcategory>/]<name>/`.
Categories: `data_processing`, `training`, `evaluation`, `deployment`.

## Common Commands

Uses `uv` as the package manager. All Python commands run through `uv run`.

```bash
# Formatting and linting
make format                    # Auto-fix Python formatting (ruff format + ruff check --fix)
make lint                      # Run all linters (format, python, markdown, yaml, imports)
make lint-python               # Ruff check only
make lint-imports              # Import guard check (components/pipelines only)

# Tests
make test                      # Run script tests: cd .github/scripts && uv run pytest */tests/ -v
make test ARGS="-k test_name"  # Run a single test

# Component/pipeline unit tests (run directly, not via make test)
uv run pytest components/<category>/<name>/tests/test_component_unit.py -v
uv run pytest pipelines/<category>/<name>/tests/test_pipeline.py -v

# Scaffolding
make component CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>] [NO_TESTS=true] [CREATE_SHARED=true]
make pipeline CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>] [NO_TESTS=true] [CREATE_SHARED=true]
make tests TYPE=component|pipeline CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>]

# README generation (READMEs are auto-generated, not hand-written)
make readme TYPE=component CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>]

# Package sync (after adding/removing components or pipelines)
make sync-packages
```

## Architecture

### Layout

- `components/` — Reusable ML components (each has `component.py`, `metadata.yaml`, `OWNERS`, `README.md`)
- `pipelines/` — Multi-step workflows composing components (each has `pipeline.py`, `metadata.yaml`, `OWNERS`, `README.md`)
- `scripts/` — Repository automation (skeleton generator, README generator, validators, package sync)
- `.github/scripts/` — CI helper scripts (import checker, README validator, base image checker)
- `conftest.py` — Global pytest config: adds project root to `sys.path`, provides `setup_and_teardown_subprocess_runner` fixture for LocalRunner tests

### Package structure

The repo is packaged as `kfp_components` with `setuptools.package-dir` mapping `.` -> `kfp_components`. The packages list in `pyproject.toml` is auto-synced via `make sync-packages` / `scripts/sync_packages/sync_packages.py`.

### Import guard

Components and pipelines have restricted imports enforced by CI and pre-commit. Allowed imports are defined in `.github/scripts/check_imports/import_exceptions.yaml`. Components must be self-contained — they cannot import from other components or from pipelines.

### READMEs are auto-generated

Component/pipeline READMEs are generated from docstrings and `metadata.yaml` by `scripts/generate_readme/`. Never edit them by hand — regenerate with `make readme`. CI will fail if READMEs are out of sync.

## Code Style

- Ruff enforces formatting and linting (config in `pyproject.toml [tool.ruff]`)
- Line length: 120
- Double quotes, Google-style docstrings
- Rules: E, W, F, I (isort), D (pydocstyle)
- `snake_case` for directory names
- Base images must be from the allowlist in `scripts/validate_base_images/base_image_allowlist.yaml` (no `:latest` tags)

## Key Patterns

### Component definition

```python
@dsl.component(
    base_image="quay.io/opendatahub/...:tag",
    packages_to_install=["package>=1.0"],
)
def my_component(
    output_artifact: dsl.Output[dsl.Dataset],
    param: str,
    optional_param: float = 0.9,
):
    """Docstring (Google style)."""
    ...
```

### Component unit test

Tests use `.python_func` attribute to test the underlying function and mock external dependencies. LocalRunner tests use the `setup_and_teardown_subprocess_runner` fixture from `conftest.py`.

### Required files per component/pipeline

Each asset directory must contain: the main `.py` file (`component.py` or `pipeline.py`), `metadata.yaml` (with required field order per CONTRIBUTING.md schema), `OWNERS`, and `README.md` (auto-generated).

## Important References

- `AGENTS.md` — Comprehensive guide for AI agents (reuse-first principles, mode-based guidance)
- `docs/CONTRIBUTING.md` — Required files, metadata schema, testing guide, workflow
- `docs/GOVERNANCE.md` — Ownership model, approval process, lifecycle
