# Contributing to Kubeflow Pipelines Components

Welcome! This guide covers everything you need to know to contribute components and pipelines to this repository.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Setup](#quick-setup)
- [What We Accept](#what-we-accept)
- [Component Structure](#component-structure)
- [Naming Conventions](#naming-conventions)
- [Development Workflow](#development-workflow)
- [Testing and Quality](#testing-and-quality)
  - [Component Testing Guide](#component-testing-guide)
- [Adding a Custom Base Image](#adding-a-custom-base-image)
- [Submitting Your Contribution](#submitting-your-contribution)
- [Getting Help](#getting-help)

## Prerequisites

Before contributing, ensure you have the following tools installed:

- **Python 3.11+** for component development
- **uv** ([installation guide](https://docs.astral.sh/uv/getting-started/installation)) to manage
  Python dependencies including `kfp` and `kfp-kubernetes` packages
- **pre-commit** ([installation guide](https://pre-commit.com/#installation)) for automated code
  quality checks
- **Docker or Podman** to build container images for custom components
- **kubectl** ([installation guide](https://kubernetes.io/docs/tasks/tools/)) for Kubernetes
  operations

All contributors must follow the
[Kubeflow Community Code of Conduct](https://github.com/kubeflow/community/blob/master/CODE_OF_CONDUCT.md).

## Quick Setup

### Installing uv

This project uses `uv` for fast Python package management.

Follow the installation instructions at: <https://docs.astral.sh/uv/getting-started/installation/>

Verify installation:

```bash
uv --version
```

### Setting Up Your Environment

Get your development environment ready with these commands:

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/pipelines-components.git
cd pipelines-components
git remote add upstream https://github.com/kubeflow/pipelines-components.git

# Set up Python environment
uv venv
source .venv/bin/activate
uv sync          # Installs package in editable mode
uv sync --extra dev    # Include dev dependencies if defined
pre-commit install

# Verify your setup works
pytest
```

### Building Packages

```bash
uv build
```

### Installing and Testing the Built Package

After building, you can install and test the wheel locally:

```bash
# Install the built wheel
uv pip install dist/kfp_components-*.whl

# Test imports work correctly
python -c "from kfp_components import components, pipelines; print('Core package imports OK')"
```

### Dependency management (`uv.lock`)

This repository uses `uv` with a committed lockfile:

- Dependency definitions live in `pyproject.toml`
- Resolved dependency graph lives in `uv.lock`

Prefer leaving dependency versions unpinned/unrestricted in `pyproject.toml` unless you have a concrete reason
(e.g., known incompatibility, security issue, or a required feature/behavior). If you restrict a dependency, add a
short comment explaining why (and link an issue if applicable). Use `uv.lock` to lock the resolved versions for
reproducible local development and CI.

If you change dependencies (e.g., edit `pyproject.toml`), update the lockfile and ensure it is in sync:

```bash
uv lock
uv lock --check
```

CI also verifies that `uv.lock` is in sync (see `.github/workflows/python-lint.yml`).

### Pre-commit validation

Before opening a PR, run pre-commit locally so you catch formatting/lint/validation issues early:

```bash
pre-commit run
```

## What We Accept

We welcome contributions of production-ready ML components and re-usable pipelines:

- **Components** are individual ML tasks (data processing, training, evaluation, deployment)
- **Pipelines** are complete multi-step workflows that can be nested within other pipelines
- **Bug fixes** improve existing components or fix documentation issues

## Component Structure

Components must be organized by category under `components/<category>/`.

Pipelines must be organized by category under `pipelines/<category>/`.

### Subcategories

For better organization of related components or pipelines, you can create **subcategories** within a category.
Subcategories provide:

- **Logical grouping** of related assets (e.g., all sklearn-based trainers, related ML workflows)
- **Dedicated ownership** via subcategory-level OWNERS file
- **Shared utilities** via an optional `shared/` package

**Component subcategory structure:**

```text
components/<category>/<subcategory>/
├── __init__.py            # Subcategory package
├── OWNERS                 # Subcategory maintainers
├── README.md              # Auto-generated subcategory index (lists components)
├── shared/                # Optional shared utilities package
│   ├── __init__.py
│   └── training_utils.py  # Common code for components in this subcategory
└── <component_name>/      # Individual component
    ├── __init__.py
    ├── component.py
    ├── metadata.yaml
    ├── OWNERS
    ├── README.md           # Auto-generated from metadata.yaml
    └── tests/
```

**Pipeline subcategory structure:**

```text
pipelines/<category>/<subcategory>/
├── __init__.py            # Subcategory package
├── OWNERS                 # Subcategory maintainers
├── README.md              # Auto-generated subcategory index (lists pipelines)
├── shared/                # Optional shared utilities package
│   ├── __init__.py
│   └── workflow_utils.py  # Common code for pipelines in this subcategory
└── <pipeline_name>/       # Individual pipeline
    ├── __init__.py
    ├── pipeline.py
    ├── metadata.yaml
    ├── OWNERS
    ├── README.md           # Auto-generated from metadata.yaml
    └── tests/
```

## Naming Conventions

- **Components and pipelines** use `snake_case` (e.g., `data_preprocessing`, `model_trainer`)
- **Commit messages** follow [Conventional Commits](https://conventionalcommits.org/) format with
  type prefix (feat, fix, docs, etc.)

### Required Files

Every component must include these files in its directory:

```text
components/<category>/<component_name>/
├── __init__.py            # Exposes the component function for imports
├── component.py           # Main implementation
├── metadata.yaml          # Complete specification (see schema below)
├── README.md              # Overview, inputs/outputs, usage examples, development instructions
├── OWNERS                 # Maintainers (approvers must be Kubeflow community members)
├── Containerfile          # Container definition (optional; required only when using a custom image)
├── example_pipelines.py   # Working usage examples (optional)
└── tests/
│   └── test_component.py  # Unit tests (optional)
└── <supporting_files>
```

Similarly, every pipeline must include these files:

```text
pipelines/<category>/<pipeline_name>/
├── __init__.py            # Exposes the pipeline function for imports
├── pipeline.py            # Main implementation
├── metadata.yaml          # Complete specification (see schema below)
├── README.md              # Overview, inputs/outputs, usage examples, development instructions
├── OWNERS                 # Maintainers (approvers must be Kubeflow community members)
├── example_pipelines.py   # Working usage examples (optional)
└── tests/
│   └── test_pipeline.py  # Unit tests (optional)
└── <supporting_files>
```

> **Note:** When using subcategories, the same required files apply at
> `components/<category>/<subcategory>/<component_name>/` or
> `pipelines/<category>/<subcategory>/<pipeline_name>/`.

### metadata.yaml Schema

Your `metadata.yaml` must include these fields:

```yaml
name: my_component
stability: stable  # 'experimental', 'alpha', 'beta', or 'stable'
dependencies:
  kubeflow:
    - name: Pipelines
      version: '>=2.5'
  external_services:  # Optional list of external dependencies
    - name: Argo Workflows
      version: "3.6"
tags:  # Optional keywords for discoverability
  - training
  - evaluation
lastVerified: 2025-11-18T00:00:00Z  # Updated annually; components are removed after 12 months without update
ci:
  skip_dependency_probe: false   # Optional. Set true only with justification
links:  # Optional, can use custom key-value (not limited to documentation, issue_tracker)
  documentation: https://kubeflow.org/components/my_component
  issue_tracker: https://github.com/kubeflow/pipelines-components/issues
```

### OWNERS File

The OWNERS file enables component owners to self-service maintenance tasks including approvals,
metadata updates, and lifecycle management:

```yaml
approvers:
  - maintainer1  # Approvers must be Kubeflow community members
  - maintainer2
reviewers:
  - reviewer1
```

The `OWNERS` file enables code review automation by leveraging PROW commands:

- **Reviewers** (as well as **Approvers**), upon reviewing a PR and finding it good to merge, can
  comment `/lgtm`, which applies the `lgtm` label to the PR
- **Approvers** (but not **Reviewers**) can comment `/approve`, which signifies the PR is approved
  for automation to merge into the repo.
- If a PR has been labeled with both `lgtm` and `approve`, and all required CI checks are passing,
  PROW will merge the PR into the destination branch.

See [full Prow documentation](https://docs.prow.k8s.io/docs/components/plugins/approve/approvers/#lgtm-label)
for usage details.

## Branching Strategy

This repository follows a branch naming convention aligned with Kubeflow Pipelines:

| Branch                    | Purpose                                    | Base Image Tag                 |
|---------------------------|--------------------------------------------|--------------------------------|
| `main`                    | Active development                         | `:main`                        |
| `release-<major>.<minor>` | Release maintenance (e.g., `release-1.11`) | `:v<major>.<minor>.<z-stream>` |

### Release Branches

Release branches are created for each minor version release:

- **Naming:** `release-<major>.<minor>` (e.g., `release-1.11`, `release-2.0`)
- **Purpose:** Maintain stable releases and backport critical fixes
- **Base images:** Components on release branches should reference the appropriate release tag (e.g., `:v1.11.0`, `:v1.11.1`, ...)

When working on a release branch:

```python
# For release-1.11, components should use the appropriate patch tag:
@dsl.component(base_image="ghcr.io/kubeflow/pipelines-components-example:v1.11.0")
```

### Z-stream (patch) releases

In addition to the initial `x.y.0` release for a given `release-x.y` branch, we may cut one or more patch (z-stream) releases (`x.y.1`, `x.y.2`, ...).

Typical characteristics:

- **Contents**: Backported bug fixes, security fixes, dependency/base-image updates, and other low-risk changes needed to keep the release usable.
- **Triggers**: Critical regressions, CVEs, or other issues that require updates on a maintained `release-x.y` branch.

## Development Workflow

### 1. Create Your Feature Branch

Start by syncing with upstream and creating a feature branch:

```bash
git remote add upstream https://github.com/kubeflow/pipelines-components.git  # if not already set
git fetch upstream
git checkout -b component/my-component upstream/main
```

### 2. Create Your Component or Pipeline

You can create components and pipelines using either the automated approach (recommended) or manually. Both approaches are detailed below.

#### Approach A: Automated Skeleton Generation (Recommended)

For rapid development, this repository provides convenient make commands that automate the entire development process.

<details>
<summary><strong>📋 Overview of Make Commands</strong></summary>

The following make targets simplify the development workflow:

| Command                                              | Description                                                       |
|------------------------------------------------------|-------------------------------------------------------------------|
| `make component CATEGORY=<cat> NAME=<name>`          | Create a new component skeleton                                   |
| `make pipeline CATEGORY=<cat> NAME=<name>`           | Create a new pipeline skeleton                                    |
| `make tests TYPE=<type> CATEGORY=<cat> NAME=<name>`  | Add tests to existing component/pipeline                          |
| `make readme TYPE=<type> CATEGORY=<cat> NAME=<name>` | Generate/update README from code                                  |
| `make format`                                        | Auto-fix code formatting and linting issues                       |
| `make lint`                                          | Check code quality (formatting, linting, imports)                 |
| `make sync-packages`                                 | Sync package entries in `pyproject.toml` with discovered packages |

> **Note:** `make component` and `make pipeline` automatically run `make sync-packages`,
> so `pyproject.toml` may be updated after generating a new skeleton.

**Optional flags** (append to component/pipeline commands):

- `SUBCATEGORY=<sub>` - Create asset in a subcategory
- `NO_TESTS=true` - Skip test file generation
- `CREATE_SHARED=true` - Create shared utilities package (requires SUBCATEGORY)

</details>

**Create a component with tests (recommended):**

```bash
make component CATEGORY=data_processing NAME=my_data_processor
```

**Create a pipeline with tests:**

```bash
make pipeline CATEGORY=training NAME=my_training_pipeline
```

**Create without tests (for rapid prototyping):**

```bash
make component CATEGORY=data_processing NAME=my_prototype NO_TESTS=true
make pipeline CATEGORY=training NAME=my_prototype NO_TESTS=true
```

This generates the complete directory structure:

```text
components/data_processing/my_data_processor/
├── __init__.py            # Import configuration
├── component.py           # Implementation template with TODOs
├── metadata.yaml          # Pre-configured metadata
├── README.md              # Documentation template
├── OWNERS                 # Maintainer template
└── tests/                 # Test directory (if not using NO_TESTS)
    ├── __init__.py
    ├── test_component_unit.py     # Unit test template
    └── test_component_local.py    # Integration test template
```

**Create a component within a subcategory:**

```bash
# Create component in a subcategory (subcategory files created automatically)
make component CATEGORY=training SUBCATEGORY=sklearn_trainer NAME=logistic_regression

# Create component in subcategory with shared utilities package
make component CATEGORY=training SUBCATEGORY=sklearn_trainer NAME=random_forest CREATE_SHARED=true
```

This generates a nested structure:

```text
components/training/sklearn_trainer/
├── __init__.py                    # Subcategory package
├── OWNERS                         # Subcategory maintainers
├── README.md                      # Subcategory documentation
├── shared/                        # (if CREATE_SHARED=true) Shared utilities
│   ├── __init__.py
│   └── sklearn_trainer_utils.py   # Placeholder utility file
└── logistic_regression/           # Your component
    ├── __init__.py
    ├── component.py
    ├── metadata.yaml
    ├── OWNERS
    ├── README.md
    └── tests/
        ├── __init__.py
        ├── test_component_local.py
        └── test_component_unit.py
```

**Create a pipeline within a subcategory:**

```bash
# Create pipeline in a subcategory (subcategory files created automatically)
make pipeline CATEGORY=training SUBCATEGORY=ml_workflows NAME=batch_training

# Create pipeline in subcategory with shared utilities package
make pipeline CATEGORY=training SUBCATEGORY=ml_workflows NAME=batch_training CREATE_SHARED=true
```

This generates a nested structure:

```text
pipelines/training/ml_workflows/
├── __init__.py                  # Subcategory package
├── OWNERS                       # Subcategory maintainers
├── README.md                    # Subcategory documentation
├── shared/                      # (if CREATE_SHARED=true) Shared utilities
│   ├── __init__.py
│   └── ml_workflows_utils.py    # Placeholder utility file
└── batch_training/              # Your pipeline
    ├── __init__.py
    ├── pipeline.py
    ├── metadata.yaml
    ├── OWNERS
    ├── README.md
    └── tests/
        ├── __init__.py
        ├── test_pipeline_local.py
        └── test_pipeline_unit.py
```

<details>
<summary><strong>🔧 Alternative: Manual Creation</strong></summary>

If you prefer to create components manually or need more control over the structure, you can create your component following the directory structure above. Here's a basic template:

```python
# component.py
from kfp import dsl

@dsl.component(base_image="python:3.11")
def hello_world(name: str = "World") -> str:
    """A simple hello world component.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting message.
    """
    message = f"Hello, {name}!"
    print(message)
    return message
```

Write comprehensive tests for your component:

```python
# tests/test_component.py
from ..component import hello_world

def test_hello_world_default():
    """Test hello_world with default parameter."""
    # Access the underlying Python function from the component
    result = hello_world.python_func()
    assert result == "Hello, World!"


def test_hello_world_custom_name():
    """Test hello_world with custom name."""
    result = hello_world.python_func(name="Kubeflow")
    assert result == "Hello, Kubeflow!"
```

</details>

### 3. Implement Your Logic

Edit the generated `component.py` or `pipeline.py` file to replace TODO placeholders with your actual implementation. The skeleton includes:

- Proper imports and decorators
- Parameter and return type hints
- Docstring templates
- Compilation logic

### 4. Add Tests (if created without tests initially)

```bash
make tests TYPE=component CATEGORY=data_processing NAME=my_data_processor
make tests TYPE=pipeline CATEGORY=training NAME=my_training_pipeline
```

### 5. Document Your Component

#### Automated README Generation (Recommended)

After implementing your logic, generate comprehensive README documentation using the existing [README generation utility](../scripts/generate_readme/README.md):

```bash
make readme TYPE=component CATEGORY=data_processing NAME=my_data_processor
make readme TYPE=pipeline CATEGORY=training NAME=my_training_pipeline
```

This automatically:

- Extracts parameters and return types from your code
- Parses docstrings for descriptions
- Generates usage examples
- Creates standardized documentation sections

#### Manual Documentation

Alternatively, you can create documentation manually following the standardized README.md format required by this repository. See the [README Generator Script Documentation](../scripts/generate_readme/README.md) for details on the expected structure.

### 6. Format and Validate Code

**Auto-fix formatting and linting issues:**

```bash
make format
```

**Check code quality before committing:**

```bash
make lint
```

**Run tests:**

```bash
# Run your component/pipeline tests
pytest components/data_processing/my_data_processor/tests/ -v
pytest pipelines/training/my_training_pipeline/tests/ -v

# Or run all tests
pytest
```

### 7. Pre-commit Validation

Run the complete pre-commit validation:

```bash
pre-commit run
```

This ensures your contribution meets all quality standards before submission.

### 8. Commit and Submit

Follow the [Submitting Your Contribution](#submitting-your-contribution) section below to commit your changes and create a pull request.

#### Complete Example Workflow

Here's a complete example creating a data processing component:

```bash
# 1. Create feature branch
git checkout -b component/csv-cleaner upstream/main

# 2. Create component skeleton
make component CATEGORY=data_processing NAME=csv_cleaner

# 3. Edit components/data_processing/csv_cleaner/component.py
# (Implement your logic, replace TODOs)

# 4. Generate documentation
make readme TYPE=component CATEGORY=data_processing NAME=csv_cleaner

# 5. Format and validate
make format
make lint

# 6. Run tests
pytest components/data_processing/csv_cleaner/tests/ -v

# 7. Final validation
pre-commit run

# 8. Commit and submit PR
git add .
git commit -m "feat(data_processing): add csv_cleaner component"
git push origin component/csv-cleaner
```

This workflow typically takes just a few minutes to set up the complete component structure with documentation and tests.

#### Example Workflow with Subcategory

When creating related components that share ownership or utilities:

```bash
# 1. Create component in subcategory
make component CATEGORY=training SUBCATEGORY=sklearn_trainer NAME=logistic_regression

# 2. Edit the component and subcategory files:
#    - components/training/sklearn_trainer/logistic_regression/component.py (your logic)
#    - components/training/sklearn_trainer/OWNERS (subcategory maintainers)

# 3. Generate documentation
make readme TYPE=component CATEGORY=training SUBCATEGORY=sklearn_trainer NAME=logistic_regression

# 4. Run tests
pytest components/training/sklearn_trainer/logistic_regression/tests/ -v

# 5. Format, lint, and submit
make format
make lint
pre-commit run
git add .
git commit -m "feat(training): add logistic_regression component in sklearn_trainer subcategory"
```

## Testing and Quality

### Running Tests Locally

Before running tests, install the project's test dependencies and any
component-specific packages your component requires at runtime:

```bash
# Install project-wide test dependencies (from the repo root)
uv sync --extra test
```

> **Note:** Some components require additional runtime packages for local testing
> (e.g., SDK libraries listed in `packages_to_install`). See each component's
> README for specific installation instructions.

Run these commands from your component/pipeline directory before submitting your contribution:

```bash
# Run all unit tests with coverage reporting
pytest --cov=src --cov-report=html

# Run specific test files when debugging
pytest tests/test_my_component.py -v
```

### Code Quality Checks

Ensure your code meets quality standards:

```bash
# Format and lint with ruff
uv run ruff format --check .      # Check formatting (120 char line length)
uv run ruff check .                # Check linting, docstrings, and import order

# Or use make commands for convenience
make lint                          # Run all linting checks
make format                        # Auto-format and auto-fix issues

# Validate import guard (enforces stdlib-only top-level imports)
uv run .github/scripts/check_imports/check_imports.py \
  --config .github/scripts/check_imports/import_exceptions.yaml \
  components pipelines

# Validate YAML files
uv run yamllint -c .yamllint.yml .

# Validate Markdown files
markdownlint -c .markdownlint.json **/*.md

# Validate metadata schema
python scripts/validate_metadata.py

# Run all pre-commit hooks
pre-commit run
```

### Base Image Validation

All components and pipelines must use approved base images. The validation script compiles components
using `kfp.compiler` to extract the actual runtime images, which correctly handles:

- Variable references (`base_image=MY_IMAGE`)
- `functools.partial` wrappers
- Default image resolution

**Valid base images:**

- Images starting with `ghcr.io/kubeflow/` (Kubeflow official registry)
- Standard Python images (`python:<version>`, e.g., `python:3.11`, `python:3.11-slim`)

Run the validation locally:

```bash
# Run with default settings
uv run scripts/validate_base_images/validate_base_images.py
```

The script allows any standard Python image matching `python:<version>` (e.g., `python:3.11`,
`python:3.10-slim`) in addition to Kubeflow registry images.

### Compile & Dependency Validation

Every component and pipeline that sets `ci.compile_check: true` in its `metadata.yaml` must compile
successfully and declare well-formed dependency metadata. The compile-check CLI discovers
metadata-backed assets, validates their `dependencies` block, and compiles the exposed
`@dsl.component`/`@dsl.pipeline` functions.

Run it locally with:

```bash
# Run against all metadata-backed targets
uv run python -m scripts.compile_check.compile_check

# Limit to one directory (can be repeated)
uv run python -m scripts.compile_check.compile_check \
  --path components/training/my_component
```

The script exits non-zero if any dependency metadata is malformed or if compilation fails, matching
the behaviour enforced by CI (`.github/workflows/compile-and-deps.yml`).

**Import Guard**: This repository enforces that top-level imports must be limited to Python's
standard library. Heavy dependencies (like `kfp`, `pandas`, etc.) should be imported within
function/pipeline bodies. Exceptions can be added to
`.github/scripts/check_imports/import_exceptions.yaml` when justified (e.g., for test files
importing `pytest`).
Note: `kfp` is allowlisted at module scope; `kfp_components` is allowlisted at module scope for `pipelines/**`.

**Common error**: `imports non-stdlib module '<module>' at top level`

This often happens in modules under `components/` or `pipelines/`.
Keep top-level imports to a bare minimum for compilation, and place imports needed at runtime inside pipeline/component bodies.

**Scripts tests (relative imports)**: For tests under `scripts/**/tests/` and `.github/scripts/**/tests/`, use relative
imports from the parent module so imports work consistently in both IDEs and pytest. Canonical guidance:
[`scripts/README.md` (Import Conventions)](../scripts/README.md#import-conventions).

### Component Testing Guide

This section explains how to write comprehensive tests for your components, using the `yoda_data_processor` component as a reference example.

#### Types of Tests

**Unit Tests** test your component's core logic in isolation:

- Use mocking to avoid external dependencies
- Test the component's Python function directly via `.python_func()`
- Fast execution, no external resources required
- Located in `tests/test_component_unit.py`

**Local Runner Tests** test your component in a real execution environment:

- Execute the component using KFP's [LocalRunner](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/)
- Test actual component behavior end-to-end
- Located in `tests/test_component_local.py`

#### Setting Up Component Tests

Create a `tests/` directory in your component folder with the following structure:

```text
components/<category>/<component_name>/tests/
├── __init__.py
├── test_component_unit.py      # Unit tests with mocking
└── test_component_local.py     # LocalRunner integration tests
```

#### Writing Unit Tests

Unit tests should verify your component's logic without external dependencies. Here's the pattern used in `yoda_data_processor`:

```python
# tests/test_component_unit.py
from unittest import mock
from ..component import your_component_function

class TestYourComponentUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(your_component_function)
        assert hasattr(your_component_function, "python_func")

    @mock.patch("external_library.some_function")
    def test_component_with_mocked_dependencies(self, mock_function):
        """Test component behavior with mocked external calls."""
        # Setup mocks
        mock_function.return_value = "expected_result"

        # Create mock output objects
        mock_output = mock.MagicMock()
        mock_output.path = "/tmp/test_output"

        # Call the component's Python function directly
        your_component_function.python_func(
            input_param="test_value",
            output_artifact=mock_output
        )

        # Verify expected interactions
        mock_function.assert_called_once_with("test_value")
```

Key patterns for unit tests:

- Use `@mock.patch` to mock external dependencies
- Call `your_component.python_func()` to test the underlying Python function
- Mock output artifacts with `.path` attributes pointing to test paths
- Verify function calls and parameter passing

#### Writing Local Runner Tests

Local Runner tests execute your component in a real KFP environment. Use the provided fixtures:

```python
# tests/test_component_local.py
from ..component import your_component_function
from tests.utils.fixtures import setup_and_teardown_subprocess_runner

class TestYourComponentLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    def test_local_execution(self, setup_and_teardown_subprocess_runner):
        """Test component execution with LocalRunner."""
        # Execute the component with real parameters
        your_component_function(
            input_param="real_value",
            # Output artifacts are handled automatically by LocalRunner
        )

        # Add assertions about expected outputs if needed
        # (files created, logs generated, etc.)
```

**Important notes for Local Runner tests:**

- The `setup_and_teardown_subprocess_runner` fixture is automatically available (no import required)
- Use the fixture as a test method parameter: `def test_local_execution(self, setup_and_teardown_subprocess_runner)`
- The fixture handles LocalRunner setup, workspace creation, and cleanup
- **Resource Requirements**: Ensure your test environment has sufficient CPU, memory, and disk space to execute the component's actual workload
- Tests may download data, install packages, or perform computationally intensive operations

#### Test Infrastructure and Configuration

The repository provides test infrastructure through a global `conftest.py` file at the project root:

**Global Test Configuration** (`conftest.py`):

- **Session Setup Hook**: Uses `pytest_sessionstart` to configure the test environment before any tests run
- **Path Management**: Automatically adds the project root to `sys.path` for clean imports during testing
- **LocalRunner Fixture**: `setup_and_teardown_subprocess_runner` (module-scoped)
  - Creates isolated workspace and output directories (`./test_workspace_subprocess`, `./test_pipeline_outputs_subprocess`)
  - Configures KFP LocalRunner with subprocess execution (no virtual environment)
  - Enables `raise_on_error=True` for immediate test failure on component errors
  - Automatically cleans up test artifacts after each test module completes

**Pytest Configuration** (`pyproject.toml`):

- **Test Discovery**: Configured to find tests in `components/*/tests` and `pipelines/*/tests` directories
- **Import Mode**: Uses `--import-mode=importlib` for better import handling
- **Automatic Detection**: Automatically discovers component and pipeline tests without manual configuration

#### Handling Import Issues with Ruff

If Ruff complains about pytest fixture imports, you may encounter two types of errors:

**F401 (unused import)** - If Ruff removes imports that are only used as pytest fixture parameters:

```python
from tests.utils.fixtures import setup_and_teardown_subprocess_runner  # noqa: F401
```

**F811 (redefinition)** - If Ruff thinks the fixture parameter redefines the imported name:

```python
def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
```

These comments tell Ruff that the import and parameter usage are intentional pytest fixture patterns.

#### Running Component Tests

From your component directory, run:

```bash
# Run all tests
pytest tests/

# Run only unit tests (fast)
pytest tests/test_component_unit.py -v

# Run only local runner tests (slower, requires resources)
pytest tests/test_component_local.py -v

# Run with coverage reporting
pytest tests/ --cov=. --cov-report=html
```

#### Test Requirements

- **Unit tests**: Should have high coverage of your component's logic
- **Local runner tests**: Should verify end-to-end component execution
- **Resource considerations**: Local runner tests require adequate system resources for your component's workload
- **Dependencies**: Mock external services in unit tests; use real dependencies in local runner tests.
  See each component's README for required runtime packages to install before running tests
- **Cleanup**: Use provided fixtures to ensure proper test environment cleanup

### Building Custom Container Images

If your component uses a custom image, test the container build:

```bash
# Build your component image
docker build -t my-component:test components/<category>/my-component/

# Test the container runs correctly
docker run --rm my-component:test echo "Hello, world!"
```

### CI Pipeline

GitHub Actions automatically runs these checks on every pull request:

- **Python linting**: Code formatting, style checks, docstring validation, and import sorting
- **Import guard**: Validates that top-level imports are limited to Python's standard library
- **YAML linting**: Validates YAML file syntax and style (yamllint)
- **Markdown linting**: Validates Markdown formatting and style (markdownlint)
- Unit and integration tests with coverage reporting
- Container image builds for components with Containerfiles
- Security vulnerability scans
- Metadata schema validation
- Standardized README content and formatting conformance

### Dependency updates (Dependabot)

This repository uses Dependabot to keep:

- Python dependencies (including pinned direct dependencies in `pyproject.toml`) and `uv.lock` up to date
- GitHub Actions versions in workflow files up to date

Configuration lives in `.github/dependabot.yml`.

## Adding a Custom Base Image

Components that require specific dependencies beyond what's available in standard KFP images can use
custom base images. This section explains how to add and maintain custom base images for your
components.

### Overview

Custom base images are:

- Built automatically by CI on every push to `main` and on tags
- Published to `ghcr.io/kubeflow/pipelines-components-<name>`
- Tagged with `:main` for the latest main branch build, or `:v<version>` for releases

### Step 1: Create the Containerfile

Create a `Containerfile` in your component's directory:

```text
components/
└── training/
    └── my_component/
        ├── Containerfile      # Your custom base image
        ├── component.py
        ├── metadata.yaml
        └── README.md
```

See [`examples/Containerfile`](examples/Containerfile) for a complete example with recommended patterns
(labels, environment settings, non-root user, etc.).

**Guidelines:**

- Keep images minimal - only include dependencies your component needs
- Pin dependency versions for reproducibility
- Use official base images when possible
- Avoid including secrets or credentials

### Step 2: Add Entry to the Workflow Matrix

Edit `.github/workflows/container-build.yml` and add your image to the `strategy.matrix.include`
array in the `build` job:

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - name: example
        context: docs/examples
      # Add your new image:
      - name: my-training-image
        context: components/training/my_component
```

**Matrix fields:**

- `name`: Unique identifier for your image. The final image will be
  `ghcr.io/kubeflow/pipelines-components-<name>`.
- `context`: Build context directory containing your `Containerfile`.

**Naming convention:**

- Use lowercase with hyphens: `my-training-component`
- Be descriptive: `sklearn-preprocessing`, `pytorch-training`
- The full image path will be: `ghcr.io/kubeflow/pipelines-components-my-training-component`

### Step 3: Reference the Image in Your Component

In your `component.py`, use the `base_image` parameter with the `:main` tag:

```python
from kfp import dsl

@dsl.component(
    base_image="ghcr.io/kubeflow/pipelines-components-my-training-image:main"
)
def my_component(input_path: str) -> str:
    import pandas as pd
    from sklearn import preprocessing
    
    # Your component logic here
    ...
```

**Important:** Always use the `:main` tag during development. This ensures:

- Your component uses the latest image from the main branch
- PR validation can override the tag to test against PR-built images

### How CI Handles Base Images

| Event | Behavior |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Pull Request | Images are built but **not pushed**. Validation uses locally-loaded `:<sha>` tags (full 40-character commit SHA). |
| Push to `main` | Images are built and pushed with tag: `:main` |
| Push to tag (e.g., `v1.0.0`) | Images are built and pushed with tag: `:<tag>` |

### Image Tags

Your image will be available with these tags:

| Tag | Description | Example |
| ---------- | ----------------------- | --------------- |
| `:main` | Latest build from main branch | `...-my-component:main` |
| `:<tag>` | Git tag | `...-my-component:v1.0.0` |
| `:<sha>` | PR validation tag (local only; full 40-character commit SHA; not pushed to the registry) | `...-my-component:3f5c8e2a9d4b7c1e0f6a3b9d8c2e4f1a6b7c3d9` |

### Testing Your Image Locally

Before submitting a PR, test your image locally:

<details>
<summary>Docker</summary>

```bash
# Build the image
docker build -t my-component:test -f components/training/my_component/Containerfile components/training/my_component

# Test it
docker run --rm my-component:test python -c "import pandas; print(pandas.__version__)"
```

</details>

<details>
<summary>Podman</summary>

```bash
# Build the image
podman build -t my-component:test -f components/training/my_component/Containerfile components/training/my_component

# Test it
podman run --rm my-component:test python -c "import pandas; print(pandas.__version__)"
```

</details>

## Submitting Your Contribution

### Commit Your Changes

Use descriptive commit messages following the [Conventional Commits](https://conventionalcommits.org/) format:

```bash
git add .
git status  # Review what you're committing
git diff --cached  # Check the actual changes

git commit -m "feat(training): add <my_component> training component

- Implements <my_component> component
- Includes comprehensive unit tests with 95% coverage
- Provides working pipeline examples
- Resolves #123"
```

### Push and Create Pull Request

Push your changes and create a pull request on GitHub:

```bash
git push origin component/my-component
```

On GitHub, click "Compare & pull request" and fill out the PR template provided with appropriate details

All PRs must pass:

- Automated checks (linting, tests, builds)
- Code review by maintainers and community members
- Documentation review

### Review Process

All pull requests must complete the following:

- All Automated CI checks successfully passing
- Code Review - reviewers will verify the following:
  - Component works as described
  - Code is clean and well-documented
  - Included tests provide good coverage.
- Receive approval from component OWNERS (for updates to existing components) or repository
  maintainers (for new components)

## Getting Help

- **Governance questions**: See [GOVERNANCE.md](GOVERNANCE.md) for ownership, verification, and process details
- **Community discussion**: Join `#kubeflow-pipelines` channel on the
  [CNCF Slack](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)
- **Bug reports and feature requests**: Open an issue at
  [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)

---

This repository was established through
[KEP-913: Components Repository](https://github.com/kubeflow/community/tree/master/proposals/913-components-repo).

Thanks for contributing to Kubeflow! 🚀
