# Refresh pipeline requirements

Refresh Hermeto-compatible `requirements.txt` lockfiles for RHOAI pipelines.

The RHOAI PyPI index does not publish macOS-compatible wheels, so this script
runs `pip-compile` inside a Linux container (`registry.access.redhat.com/ubi9/python-312:9.8`).
Use Podman or Docker; the runtime is auto-detected (podman first, then docker).

## Usage

Refresh the default AutoML and AutoRAG pipelines:

```bash
make pipeline-requirements
```

Refresh a specific pipeline:

```bash
make pipeline-requirements PIPELINE=pipelines/training/autorag/documents_rag_optimization_pipeline
```

Keep existing pins when possible:

```bash
make pipeline-requirements NO_UPGRADE=true
```

Dry run:

```bash
make pipeline-requirements DRY_RUN=true
```

Suppress live progress output:

```bash
make pipeline-requirements QUIET=true
```

Use a specific container runtime:

```bash
make pipeline-requirements RUNTIME=docker
CONTAINER_RUNTIME=docker make pipeline-requirements
```

Alternatively, invoke the script directly:

```bash
uv run python -m scripts.refresh_pipeline_requirements.refresh_pipeline_requirements
```

### Make variables

| Variable | Values | Description |
|----------|--------|-------------|
| `PIPELINE` | path | Single pipeline directory (default: both AutoML and AutoRAG) |
| `RUNTIME` | `podman`, `docker` | Container runtime (default: auto-detect) |
| `IMAGE` | image ref | Container image override |
| `NO_UPGRADE` | `true` | Keep existing pins from `requirements.txt` |
| `DRY_RUN` | `true` | Show changes without writing `requirements.txt` |
| `QUIET` | `true` | Suppress live pip-compile progress output |
| `PYTHON_PLATFORM` | uv platform | Target wheel platform (default: `x86_64-manylinux_2_34`) |

## Defaults

- **Pipelines**
  - `pipelines/training/automl/autogluon_tabular_training_pipeline`
  - `pipelines/training/autorag/documents_rag_optimization_pipeline`
- **Container image**: `registry.access.redhat.com/ubi9/python-312:9.8`
- **Container runtime**: auto-detect `podman`, then `docker` (override with `RUNTIME` or `CONTAINER_RUNTIME`)
- **Python platform**: `x86_64-manylinux_2_34` (RHEL UBI9 / glibc 2.34; matches root `make requirements`)

Each pipeline directory must contain a `requirements.in` with a `--index-url` line.
The generated `requirements.txt` keeps that index URL and includes package hashes
for Hermeto offline builds.
