# CI Scripts

This directory contains scripts used in CI workflows.

This directory follows the same structure and conventions as `scripts/`. For directory structure, conventions, and how
to add new scripts, see [scripts/README.md](../../scripts/README.md).

## Running Unit Tests Locally

```bash
cd .github/scripts
uv run pytest */tests/ -v --tb=short
```

## Import guard

`check_imports/import_exceptions.yaml` lists allowed module-level imports and
repository-relative `exclude` globs. Exclusions apply before parsing, so bundled
application templates can use their own Python version and dependencies without
weakening checks on component and pipeline modules.
