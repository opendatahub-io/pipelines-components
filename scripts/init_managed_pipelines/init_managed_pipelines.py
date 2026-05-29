"""Copy managed pipeline YAMLs to the shared volume for the KFP API server."""

import json
import shutil
import sys
from pathlib import Path

from scripts.generate_managed_pipelines.generate_managed_pipelines import (
    OUTPUT_FILENAME,
    should_recompile_managed_pipelines,
    stage_managed_pipelines,
    staged_pipeline_yaml_path,
)

APP_ROOT = Path("/app")
OUTPUT_DIR = Path("/config/managed-pipelines")
MANIFEST = OUTPUT_FILENAME


def _copy_prebuilt_from_image() -> None:
    """Copy build-time ``managed-pipelines.json`` and YAML from read-only ``/app``."""
    entries = json.loads((APP_ROOT / MANIFEST).read_text())

    for entry in entries:
        src = (APP_ROOT / entry["path"].replace("pipeline.py", "pipeline.yaml")).resolve()
        if not src.is_relative_to(APP_ROOT.resolve()):
            raise ValueError(f"Manifest path escapes app root: {entry['path']}")
        if not src.is_file():
            raise FileNotFoundError(f"Pipeline YAML missing for {entry['name']!r}: {src}")
        shutil.copy2(src, staged_pipeline_yaml_path(OUTPUT_DIR, entry["name"]))

    shutil.copy2(APP_ROOT / MANIFEST, OUTPUT_DIR / MANIFEST)
    print(f"Staged {len(entries)} pipeline(s) in {OUTPUT_DIR}")


def main() -> int:
    """Stage managed pipelines for the KFP API server init container."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if should_recompile_managed_pipelines():
        rc = stage_managed_pipelines(APP_ROOT, OUTPUT_DIR)
        if rc != 0:
            return rc
        entries = json.loads((OUTPUT_DIR / MANIFEST).read_text())
        print(f"Staged {len(entries)} pipeline(s) in {OUTPUT_DIR} (recompiled with RELATED_IMAGE)")
        return 0

    _copy_prebuilt_from_image()
    return 0


if __name__ == "__main__":
    sys.exit(main())
