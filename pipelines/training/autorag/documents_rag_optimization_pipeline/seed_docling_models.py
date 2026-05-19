#!/usr/bin/env python3
"""Populate Docling artifact dirs for offline use (matches docling 2.73.x layout).

Modes:
  --oci          Pull Red Hat OCI model artifacts from registry.stage.redhat.io (local builds).
  --hermeto-dir  Copy from Hermeto generic output (deps/generic/...); paths must match
                 artifacts.lock.yaml ``filename`` entries (networkless Konflux builds).

See https://docling-project.github.io/docling/usage/advanced_options/ and
https://github.com/hermetoproject/hermeto/blob/main/docs/generic.md
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LAYOUT_OCI_REF = "registry.stage.redhat.io/rhai/docling-project-docling-layout-heron:3.0"
MODELS_OCI_REF = "registry.stage.redhat.io/rhai/docling-project-docling-models:3.0"

LAYOUT_DIR = "docling-project--docling-layout-heron"
MODELS_DIR = "docling-project--docling-models"

_LAYER_TITLE_KEY = "org.opencontainers.image.title"


def _digest_filename(digest: str) -> str:
    return digest.split(":", 1)[-1] if ":" in digest else digest


def _skopeo_authfile() -> str | None:
    raw = os.environ.get("OCI_PULL_SECRET_MODEL_DOWNLOAD", "").strip()
    if raw:
        path = Path("/tmp/skopeo-auth.json")
        path.write_text(raw, encoding="utf-8")
        return str(path)
    for candidate in (
        os.environ.get("REGISTRY_AUTH_FILE", ""),
        str(Path.home() / ".docker" / "config.json"),
        "/run/containers/0/auth.json",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        auth = Path(xdg_runtime) / "containers" / "auth.json"
        if auth.is_file():
            return str(auth)
    return None


def _skopeo_copy(ref: str, dest: Path) -> None:
    if not shutil.which("skopeo"):
        print("error: skopeo not found in PATH (expected from Containerfile skopeo stage)", file=sys.stderr)
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    target_arch = os.environ.get("TARGETARCH", "amd64")
    cmd = [
        "skopeo",
        "copy",
        "--override-os",
        "linux",
        "--override-arch",
        target_arch,
    ]
    authfile = _skopeo_authfile()
    if authfile:
        cmd.extend(["--authfile", authfile])
    cmd.extend([f"docker://{ref}", f"dir:{dest}"])
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(f"error: skopeo copy failed for {ref}:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def _extract_oci_artifact_dir(artifact_dir: Path, dest_subdir: Path) -> int:
    """Extract files from a skopeo dir using manifest layer title annotations."""
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"error: missing manifest.json under {artifact_dir}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied = 0
    for layer in manifest.get("layers") or []:
        title = (layer.get("annotations") or {}).get(_LAYER_TITLE_KEY)
        digest = layer.get("digest")
        if not title or not digest:
            continue
        blob = artifact_dir / _digest_filename(digest)
        if not blob.is_file():
            print(f"error: missing blob {blob} for {title}", file=sys.stderr)
            sys.exit(1)
        target = dest_subdir / title
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(blob, target)
        copied += 1
    return copied


def _from_oci(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    work = dest.parent / ".oci-fetch"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    refs = (
        (LAYOUT_OCI_REF, LAYOUT_DIR),
        (MODELS_OCI_REF, MODELS_DIR),
    )
    total = 0
    for ref, dirname in refs:
        artifact_dir = work / dirname
        _skopeo_copy(ref, artifact_dir)
        copied = _extract_oci_artifact_dir(artifact_dir, dest / dirname)
        if copied == 0:
            print(f"error: no layers extracted from {ref}", file=sys.stderr)
            sys.exit(1)
        total += copied
    shutil.rmtree(work, ignore_errors=True)
    print(f"Seeded {total} files from OCI artifacts into {dest}")


def _rel_from_docling_component(rel: Path) -> Path | None:
    """Return path relative to first ``docling-project--*`` directory component."""
    for idx, part in enumerate(rel.parts):
        if part.startswith("docling-project--"):
            return Path(*rel.parts[idx:])
    return None


def hermeto_has_docling_models(source: Path) -> bool:
    """Return True if Hermeto output contains docling layout model config."""
    if not source.is_dir():
        return False
    return any(source.rglob(f"{LAYOUT_DIR}/config.json"))


def _from_hermeto(source: Path, dest: Path) -> None:
    """Copy docling files from Hermeto generic output (may be nested under package path)."""
    if not source.is_dir():
        print(f"error: Hermeto directory not found: {source}", file=sys.stderr)
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        docling_rel = _rel_from_docling_component(rel)
        if docling_rel is None:
            continue
        target = dest / docling_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    if copied == 0:
        print(
            f"error: no docling files under {source}. "
            "Ensure Hermeto generic prefetch ran and registry.stage.redhat.io is reachable.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Parse CLI arguments and populate the Docling artifact tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Directory that will contain docling-project--* model folders (same as DOCLING_ARTIFACTS_PATH).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--oci",
        action="store_true",
        help=f"Pull OCI model artifacts ({LAYOUT_OCI_REF}, {MODELS_OCI_REF}).",
    )
    group.add_argument(
        "--hermeto-dir",
        type=Path,
        metavar="DIR",
        help="Hermeto generic deps directory (e.g. .../deps/generic).",
    )
    args = parser.parse_args()
    if args.oci:
        _from_oci(args.dest)
    else:
        _from_hermeto(args.hermeto_dir, args.dest)


if __name__ == "__main__":
    main()
