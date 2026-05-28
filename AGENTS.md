# AI Agent Context Guide

*This guide provides context and information for AI agents working with the Kubeflow Pipelines Components Repository.*

See also:

- [Contributing Guide](docs/CONTRIBUTING.md)
- [Governance Guide](docs/GOVERNANCE.md)

## Sources of truth (keep this doc aligned)

If this guide conflicts with repository enforcement or process docs, treat these as sources of truth.
When repository enforcement, CI, or contribution process changes, update `AGENTS.md` alongside the change.

- [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) (required files, workflow, required metadata fields)
- [`GOVERNANCE.md`](docs/GOVERNANCE.md) (roles, ownership, lifecycle)
- [`CONTRIBUTING.md` (metadata.yaml schema)](docs/CONTRIBUTING.md#metadatayaml-schema)
- [`scripts/validate_base_images/README.md`](scripts/validate_base_images/README.md) (base image policy)
- [`CONTRIBUTING.md` (Testing and Quality)](docs/CONTRIBUTING.md#testing-and-quality)
- CI workflows live under `.github/workflows/` (example: [`.github/workflows/python-lint.yml`](.github/workflows/python-lint.yml))

## Agent modes

Agents typically interact with this repository in three modes. Use the mode to decide what you should optimize for.

1. **Contributing a component or pipeline** (authoring new assets or changing existing ones)
2. **End user building pipelines** from published components (consumption only; no repo changes)
3. **Maintaining/contributing to the repository** (scripts, tests, CI, automation)

## Quickstart (all agents)

- **Reuse-first**: search `components/<category>/` and `pipelines/<category>/` for similar functionality; prefer
  extending/composing instead of duplicating.
- **Create scaffolding**: use the Make targets in `Makefile`:
  - `make component CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>] [NO_TESTS=true] [CREATE_SHARED=true]`
  - `make pipeline CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>] [NO_TESTS=true] [CREATE_SHARED=true]`
  - `make tests TYPE=component|pipeline CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>]`
  - `make readme TYPE=component|pipeline CATEGORY=<cat> NAME=<name> [SUBCATEGORY=<sub>]`
- **Validate like CI**: follow [`CONTRIBUTING.md` (Testing and Quality)](docs/CONTRIBUTING.md#testing-and-quality) and
  reference the workflows under `.github/workflows/` (example: [`.github/workflows/python-lint.yml`](.github/workflows/python-lint.yml)).
- **New assets require approval**: for initial contributions (introducing a new component/pipeline to the catalog),
  follow the approval process in [`GOVERNANCE.md`](docs/GOVERNANCE.md).

## Mode 1: Contributing a component or pipeline

Goal: add or update an asset under `components/` or `pipelines/` that is reusable and passes repo validations.

### Required files

When the agent changes or adds a component/pipeline directory, follow
[the required files list](docs/CONTRIBUTING.md#required-files).

### Initial contributions: Pipelines Working Group approval

For initial contributions (e.g., a new component/pipeline being introduced to the catalog), the repo requires
Pipelines Working Group approval.

For context on repository roles, decision-making, and approvals, see [`GOVERNANCE.md`](docs/GOVERNANCE.md).

Process (expected for agents):

- Open a submission issue using `.github/ISSUE_TEMPLATE/component_submission.md`.
- Get Pipelines Working Group approval in that issue (link it from the PR).
- Open a PR with the implementation.
- Follow the repo's OWNERS-based review flow described in `CONTRIBUTING.md` (`/lgtm` + `/approve`).

### Common tasks

| Task | Command | Reference pattern |
|---|---|---|
| New component | `make component CATEGORY=<cat> NAME=<name>` | `components/data_processing/yoda_data_processor/` |
| New pipeline | `make pipeline CATEGORY=<cat> NAME=<name>` | `pipelines/training/sft/` |
| Subcategory asset | Add `SUBCATEGORY=<sub>`; add `CREATE_SHARED=true` for shared utils | Subcategory OWNERS/README auto-created |
| Generate tests | `make tests TYPE=component\|pipeline CATEGORY=<cat> NAME=<name>` | Unit tests + LocalRunner tests |
| Generate README | `make readme TYPE=component\|pipeline CATEGORY=<cat> NAME=<name>` | Keeps README in sync |
| Update existing | Minimal change, regenerate README if interface changed, keep `lastVerified` fresh | Same component dir |

## Mode 2: End user building pipelines from these components

Goal: compose pipelines using components/pipelines from this repository without changing repository content.

Recommended references:

- [`README.md`](README.md) (repository overview / usage entry point)
- Component and pipeline READMEs under `components/<category>/` and `pipelines/<category>/`
- Kubeflow Pipelines docs (usage and authoring concepts): `https://www.kubeflow.org/docs/components/pipelines/`

## Mode 3: Maintaining/contributing to the repository (scripts, tests, CI)

Goal: improve repository automation and tooling under `scripts/`, `.github/scripts/`, and `.github/workflows/`.

Canonical references:

- [`scripts/README.md`](scripts/README.md)
- [`.github/scripts/README.md`](.github/scripts/README.md)
- [`.github/actions/detect-changed-assets/README.md`](.github/actions/detect-changed-assets/README.md) (run work only for changed assets in CI)

Use the same validations section below; it applies to repository maintenance changes as well.

## Repository validations an agent must satisfy

### Dependencies and pre-commit

Follow [`CONTRIBUTING.md`](docs/CONTRIBUTING.md#dependency-management-uvlock) for dependency and lockfile management, and
[`CONTRIBUTING.md`](docs/CONTRIBUTING.md#pre-commit-validation) for pre-commit guidance.

### Python lint and formatting

Enforced by CI ([`.github/workflows/python-lint.yml`](.github/workflows/python-lint.yml)) using Ruff (see `pyproject.toml`).

Single-file commands:

- `uv run ruff check <file>` — lint one file
- `uv run ruff check --fix <file>` — lint + auto-fix one file
- `uv run ruff format <file>` — format one file

Whole-repo: `make lint` (check) or `make format` (fix).

### Other validations

| Validation | Config | CI workflow |
|---|---|---|
| Markdown lint | [`.markdownlint.json`](.markdownlint.json) | [`markdown-lint.yml`](.github/workflows/markdown-lint.yml) |
| YAML lint | [`.yamllint.yml`](.yamllint.yml) | [`yaml-lint.yml`](.github/workflows/yaml-lint.yml) |
| Import guard | [`import_exceptions.yaml`](.github/scripts/check_imports/import_exceptions.yaml); see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md#testing-and-quality) | [`ci-checks.yml`](.github/workflows/ci-checks.yml) |
| Metadata schema | [`CONTRIBUTING.md` (schema)](docs/CONTRIBUTING.md#metadatayaml-schema); keep required field order and fresh `lastVerified` | [`validate-metadata-schema.yml`](.github/workflows/validate-metadata-schema.yml) |
| Base images | [`validate_base_images/README.md`](scripts/validate_base_images/README.md) | [`base-image-check.yml`](.github/workflows/base-image-check.yml) |
| README sync | [`generate_readme/README.md`](scripts/generate_readme/README.md) | [`readme-check.yml`](.github/workflows/readme-check.yml) |

### Tests

Follow the canonical testing guidance:

- Component/pipeline tests: [`CONTRIBUTING.md` (Component Testing Guide)](docs/CONTRIBUTING.md#component-testing-guide)
- Scripts tests: [`scripts/README.md`](scripts/README.md) and [`.github/scripts/README.md`](.github/scripts/README.md)

Workflow references:

- Component/pipeline tests: [`.github/workflows/component-pipeline-tests.yml`](.github/workflows/component-pipeline-tests.yml)
- Scripts tests: [`.github/workflows/scripts-tests.yml`](.github/workflows/scripts-tests.yml)
