# Refresh pipeline requirements

Refresh Hermeto-compatible `requirements.txt` lockfiles for RHOAI pipelines.

The RHOAI PyPI index does not publish macOS-compatible wheels, so this script
runs `uv pip compile` inside a Linux container (`registry.access.redhat.com/ubi9/python-312:9.8`).
`uv` is used instead of `pip-compile` for a much faster resolve while producing an
equivalent hashed lockfile. Use Podman or Docker; the runtime is auto-detected
(podman first, then docker).

## Lockfile verification

After compiling, the lockfile is verified against the index it was resolved
from. The RHAI index publishes the wheels of a version in a staggered fashion
(for example the `ppc64le` wheel of `pillow==13.0.0` appeared a day after the
other architectures), so a lockfile compiled against an incomplete index
silently misses `--hash` entries and Hermeto offline builds then fail on the
missing architectures. The verification detects this by comparing the lockfile
with the current index state:

- **Arch-less packages** (`py3-none-any` wheels, sdists) carry no architecture
  expectation, but every published file (including every build tag) must be
  hashed in the lockfile.
- **Architecture-specific packages** must have hashes for the wheels of every
  architecture the index publishes, and the index itself must publish wheels
  for all supported Linux architectures (`x86_64`, `aarch64`, `ppc64le`,
  `s390x`). Index-side gaps are errors by default because they usually mean
  the index is still syncing; downgrade them with `--arch-check warn`
  (`ARCH_CHECK=warn`) when a package legitimately supports fewer
  architectures.
- **Environment markers** narrow the expectation: a package pinned with
  `platform_machine` markers (in `requirements.in`, copied onto the lockfile)
  only needs the architectures the marker allows. For example
  `triton==3.6.0; platform_machine != "s390x"` skips `s390x`, and
  `ray>=2.43; platform_machine == "x86_64" or platform_machine == "aarch64"`
  only requires `x86_64` and `aarch64`.
- **Stale hashes** (lockfile hashes matching no file published on the index)
  are reported as errors as well.

Verification failures do not revert the lockfile; re-run the refresh once the
index is complete or adjust the markers.

## Usage

Refresh the default AutoML and AutoRAG pipelines (verification runs after
compiling):

```bash
make pipeline-requirements
```

Refresh a specific pipeline (the AutoRAG inference lockfile works the same
way):

```bash
make pipeline-requirements PIPELINE=pipelines/training/autorag/documents_rag_optimization_pipeline
make pipeline-requirements PIPELINE=pipelines/training/autorag/documents_rag_optimization_pipeline/inference
```

Verify existing lockfiles only, without recompiling (no Podman or Docker
needed):

```bash
make pipeline-requirements VERIFY_ONLY=true
```

Keep existing pins when possible:

```bash
make pipeline-requirements NO_UPGRADE=true
```

Dry run (no lockfile is written, so no verification runs):

```bash
make pipeline-requirements DRY_RUN=true
```

Suppress live progress output:

```bash
make pipeline-requirements QUIET=true
```

Skip verification (for example when offline):

```bash
make pipeline-requirements NO_VERIFY=true
```

Downgrade index-side architecture gaps to warnings:

```bash
make pipeline-requirements ARCH_CHECK=warn
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
| `QUIET` | `true` | Suppress live uv pip compile progress output |
| `VERIFY_ONLY` | `true` | Verify existing lockfiles against the index without recompiling |
| `NO_VERIFY` | `true` | Skip lockfile verification after compiling |
| `ARCH_CHECK` | `error`, `warn` | Severity of index-side architecture gaps (default: `error`) |

## Defaults

- **Pipelines**
  - `pipelines/training/automl/autogluon_tabular_training_pipeline`
  - `pipelines/training/autorag/documents_rag_optimization_pipeline`
- **Container image**: `registry.access.redhat.com/ubi9/python-312:9.8`
- **Container runtime**: auto-detect `podman`, then `docker` (override with `RUNTIME` or `CONTAINER_RUNTIME`)
- **Architectures**: `x86_64`, `aarch64`, `ppc64le`, `s390x`
- **Arch check mode**: `error`

Each pipeline directory must contain a `requirements.in` with a `--index-url` line.
The generated `requirements.txt` keeps that index URL and includes package hashes
for Hermeto offline builds.
