# AutoRAG assets generator

The generator is installed by the `kfp-components` wheel. Its public API is:

```python
from kfp_components.components.training.autorag.shared.assets_generator import (
    Notebook,
    build_leaderboard_html,
    generate_notebook_from_template,
    generate_starter_kit,
)
```

Importing this package does not require ai4rag. Generation uses ai4rag's template
registry, vector-store helpers, graph settings, and installed version at call time.
Generated notebooks and applications continue importing and depending on ai4rag.

See [notebooks](docs/notebooks.md), [leaderboards](docs/leaderboard.md), and
[starter kits and gateway setup](docs/starter-kit.md) for operational instructions.
The administrator gateway helper lives at
[`scripts/setup-openshell-gateway.sh`](../../../../../../scripts/setup-openshell-gateway.sh),
outside generated ZIPs. From the parent of a pipelines-components checkout, run:

```bash
bash pipelines-components/scripts/setup-openshell-gateway.sh --help
```

## Coordinated release and rollback

This is a breaking move from `ai4rag.assets_generator`; there is no compatibility
shim. Release the matching ai4rag removal and pipelines-components changes together.
The AutoRAG training image builds and installs the local wheel with `--no-deps`;
its hashed requirements provide KFP and kfp-kubernetes. The inference image does
not need this wheel. Keep ai4rag pinned to 0.25.4 while validating the relocation;
validate the removal candidate separately before release. Broader dependency
upgrades are a separate change.

Publish and select the matching training runtime image when recompiling managed
pipelines. Previously compiled pipelines contain the old import path: retain their
compatible image or recompile them. Record the previous image digest and compiled
pipeline definition before switching. Roll back the image and pipeline definition
together.

## Starter-kit lockfile

The bundled `uv.lock` is preserved from ai4rag and references an older ai4rag
release. After extraction, refresh it against the generated `pyproject.toml` before
building the application:

```bash
uv lock --upgrade-package ai4rag
uv lock --check
uv export --locked --no-dev --no-emit-project --no-hashes > requirements.txt
```

The generated ai4rag requirement uses the installed generator dependency's
version as its minimum. `uv lock --upgrade-package ai4rag` refreshes resolution within that
requirement. The Containerfile uses `uv export --locked`, so a stale lock must be
refreshed first. Modernizing the bundled lock is separate from relocation.

## Verification

The regular Python 3.11 component suites mock ai4rag. Run the optional generator
tests locally with Python 3.12+ and real ai4rag:

```bash
uv venv --python 3.12 /tmp/autorag-generator-venv
uv pip install --python /tmp/autorag-generator-venv/bin/python \
  -r components/training/autorag/shared/tests/assets_generator/requirements.txt
uv build
uv pip install --python /tmp/autorag-generator-venv/bin/python --no-deps dist/*.whl
cp -R components/training/autorag/shared/tests/assets_generator /tmp/autorag-generator-tests
cd /tmp
/tmp/autorag-generator-venv/bin/python -m pytest /tmp/autorag-generator-tests
```

The tests compare output with the original ai4rag 0.25.4 generator. ZIP timestamps,
gateway documentation edits, and the installed ai4rag version are normalized;
the baseline records the approved removal of template file headers.

Before accepting a coordinated release, also build the wheel and sdist, test the
installed wheel outside the checkout with read-only resources, and build the training
image and run `pip check`. Cluster gateway provisioning is not a migration test;
use `bash -n` and `--help` only.
