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
import base64
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LAYOUT_OCI_REF = "registry.stage.redhat.io/rhai/docling-project-docling-layout-heron:3.0"
MODELS_OCI_REF = "registry.stage.redhat.io/rhai/docling-project-docling-models:3.0"

LAYOUT_DIR = "docling-project--docling-layout-heron"
MODELS_DIR = "docling-project--docling-models"

_LAYER_TITLE_KEY = "org.opencontainers.image.title"
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def _parse_ref(ref: str) -> tuple[str, str, str]:
    host, path = ref.split("/", 1)
    repo, tag = path.rsplit(":", 1)
    return host, repo, tag


def _read_docker_config_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_docker_config() -> dict:
    """Resolve registry credentials from env or standard container-auth locations."""
    raw = os.environ.get("OCI_PULL_SECRET_MODEL_DOWNLOAD", "").strip()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "OCI_PULL_SECRET_MODEL_DOWNLOAD is set but is not valid JSON (expected Docker config.json)."
            ) from exc
        if not isinstance(data, dict):
            raise ValueError("OCI_PULL_SECRET_MODEL_DOWNLOAD must be a JSON object.")
        return data

    candidates = [
        Path(os.environ.get("REGISTRY_AUTH_FILE", "")),
        Path.home() / ".docker" / "config.json",
        Path("/run/containers/0/auth.json"),
    ]
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        candidates.append(Path(xdg_runtime) / "containers" / "auth.json")

    for path in candidates:
        if path and path.is_file():
            config = _read_docker_config_file(path)
            if config.get("auths"):
                return config
    return {}


def _basic_auth_for_host(host: str, docker_config: dict) -> str | None:
    auths = docker_config.get("auths") or {}
    entry = auths.get(host) or auths.get(f"https://{host}")
    if not entry:
        return None
    if isinstance(entry, dict) and entry.get("auth"):
        return f"Basic {entry['auth']}"
    if isinstance(entry, dict) and entry.get("username") and entry.get("password"):
        token = base64.b64encode(f"{entry['username']}:{entry['password']}".encode()).decode()
        return f"Basic {token}"
    return None


def _parse_www_authenticate(header: str) -> dict[str, str]:
    if not header.lower().startswith("bearer "):
        return {}
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)="([^"]*)"', header[7:]):
        params[match.group(1)] = match.group(2)
    return params


def _http_get(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print(
                "error: registry authentication failed (HTTP 401). "
                "Log in to registry.stage.redhat.io (podman login / Konflux registry secret) "
                "or set OCI_PULL_SECRET_MODEL_DOWNLOAD to Docker config.json content.",
                file=sys.stderr,
            )
        raise


def _get_bearer_token(host: str, repository: str, basic_auth: str | None) -> str | None:
    probe_url = f"https://{host}/v2/"
    probe_headers: dict[str, str] = {}
    if basic_auth:
        probe_headers["Authorization"] = basic_auth
    try:
        _http_get(probe_url, probe_headers)
        return None
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        www_auth = exc.headers.get("WWW-Authenticate", "")
    params = _parse_www_authenticate(www_auth)
    realm = params.get("realm")
    if not realm:
        return None
    service = params.get("service", "docker-registry")
    scope = params.get("scope", f"repository:{repository}:pull")
    token_url = f"{realm}?{urllib.parse.urlencode({'service': service, 'scope': scope})}"
    token_headers = dict(probe_headers)
    token_body = _http_get(token_url, token_headers)
    token_data = json.loads(token_body.decode("utf-8"))
    token = token_data.get("token") or token_data.get("access_token")
    return f"Bearer {token}" if token else None


def _registry_headers(host: str, repository: str, docker_config: dict, extra: dict[str, str]) -> dict[str, str]:
    headers = dict(extra)
    basic_auth = _basic_auth_for_host(host, docker_config)
    bearer = _get_bearer_token(host, repository, basic_auth)
    if bearer:
        headers["Authorization"] = bearer
    elif basic_auth:
        headers["Authorization"] = basic_auth
    return headers


def _fetch_manifest(ref: str, docker_config: dict) -> dict:
    host, repository, tag = _parse_ref(ref)
    url = f"https://{host}/v2/{repository}/manifests/{tag}"
    headers = _registry_headers(
        host,
        repository,
        docker_config,
        {"Accept": _MANIFEST_ACCEPT},
    )
    body = _http_get(url, headers)
    return json.loads(body.decode("utf-8"))


def _fetch_blob(ref: str, digest: str, docker_config: dict) -> bytes:
    host, repository, _tag = _parse_ref(ref)
    algo, digest_hex = digest.split(":", 1)
    url = f"https://{host}/v2/{repository}/blobs/{algo}:{digest_hex}"
    headers = _registry_headers(
        host,
        repository,
        docker_config,
        {"Accept": "application/octet-stream, */*"},
    )
    return _http_get(url, headers)


def _pull_oci_artifact(ref: str, dest_subdir: Path, docker_config: dict) -> int:
    manifest = _fetch_manifest(ref, docker_config)
    dest_subdir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for layer in manifest.get("layers") or []:
        title = (layer.get("annotations") or {}).get(_LAYER_TITLE_KEY)
        digest = layer.get("digest")
        if not title or not digest:
            continue
        target = dest_subdir / title
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_fetch_blob(ref, digest, docker_config))
        copied += 1
    return copied


def _from_oci(dest: Path) -> None:
    docker_config = _load_docker_config()
    dest.mkdir(parents=True, exist_ok=True)
    refs = (
        (LAYOUT_OCI_REF, LAYOUT_DIR),
        (MODELS_OCI_REF, MODELS_DIR),
    )
    total = 0
    for ref, dirname in refs:
        copied = _pull_oci_artifact(ref, dest / dirname, docker_config)
        if copied == 0:
            print(f"error: no layers extracted from {ref}", file=sys.stderr)
            sys.exit(1)
        total += copied
    print(f"Seeded {total} files from OCI artifacts into {dest}")


def _from_hermeto(source: Path, dest: Path) -> None:
    """Hermeto stores files under deps/generic/ using lockfile ``filename`` (may include subdirs).

    Only paths whose first component starts with ``docling-project--`` are copied so other generic
    artifacts (e.g. SQLite source tarballs) can share the same Hermeto lockfile without landing
    under ``DOCLING_ARTIFACTS_PATH``.
    """
    if not source.is_dir():
        print(f"error: Hermeto directory not found: {source}", file=sys.stderr)
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if rel.parts and not rel.parts[0].startswith("docling-project--"):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    if copied == 0:
        print(f"error: no docling files under {source}", file=sys.stderr)
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
