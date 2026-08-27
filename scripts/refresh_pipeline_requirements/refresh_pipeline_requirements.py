#!/usr/bin/env python3
"""Refresh Hermeto-compatible requirements.txt lockfiles for RHOAI pipelines.

Compiles ``requirements.in`` into ``requirements.txt`` using ``uv pip compile``
with ``--generate-hashes`` inside a container (Podman or Docker). The RHOAI PyPI
index does not publish macOS-compatible wheels, so compilation must run on Linux
(UBI9 Python 3.12). ``uv`` is used instead of ``pip-compile`` because its resolver
is dramatically faster while producing an equivalent hashed lockfile.

Environment markers from ``requirements.in`` are copied onto the compiled
lockfile afterwards. ``uv pip compile`` evaluates markers for the container host
and would otherwise drop ones that are true there (for example
``platform_machine != 's390x'`` on x86_64).

See: https://hermetoproject.github.io/hermeto/pip/#requirementstxt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ..lib.discovery import get_repo_root

DEFAULT_CONTAINER_IMAGE = "registry.access.redhat.com/ubi9/python-312:9.8"
DEFAULT_PIPELINES: tuple[str, ...] = (
    "pipelines/training/automl/autogluon_tabular_training_pipeline",
    "pipelines/training/autorag/documents_rag_optimization_pipeline",
)
SUPPORTED_RUNTIMES: tuple[str, ...] = ("podman", "docker")
_CONTAINER_RUNTIME_ENV = "CONTAINER_RUNTIME"

_INDEX_URL_RE = re.compile(r"^--index-url\s+(\S+)", re.MULTILINE)
_PKG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_REQUIREMENTS_IN = "requirements.in"
_REQUIREMENTS_TXT = "requirements.txt"


class RefreshRequirementsError(Exception):
    """Raised when requirements refresh cannot proceed."""


def read_index_url(requirements_in: Path) -> str | None:
    """Return the ``--index-url`` value from a requirements input file."""
    content = requirements_in.read_text(encoding="utf-8")
    match = _INDEX_URL_RE.search(content)
    return match.group(1) if match else None


def _strip_inline_comment(line: str) -> str:
    """Remove a ``#`` comment that is not inside quotes."""
    in_quote: str | None = None
    for index, char in enumerate(line):
        if in_quote:
            if char == in_quote:
                in_quote = None
        elif char in {"'", '"'}:
            in_quote = char
        elif char == "#":
            return line[:index].rstrip()
    return line


def _canonicalize_name(name: str) -> str:
    """Normalize a package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_environment_marker(requirement_line: str) -> tuple[str, str | None]:
    """Split a PEP 508 requirement into the spec and optional environment marker."""
    in_quote: str | None = None
    for index, char in enumerate(requirement_line):
        if in_quote:
            if char == in_quote:
                in_quote = None
        elif char in {"'", '"'}:
            in_quote = char
        elif char == ";":
            spec = requirement_line[:index].rstrip()
            marker = requirement_line[index + 1 :].strip()
            return spec, marker or None
    return requirement_line.rstrip(), None


def read_requirement_markers(requirements_in: Path) -> dict[str, str]:
    """Return canonical package name to environment marker from ``requirements.in``."""
    markers: dict[str, str] = {}
    for raw_line in requirements_in.read_text(encoding="utf-8").splitlines():
        line = _strip_inline_comment(raw_line).strip()
        if not line or line.startswith("-"):
            continue
        name_match = _PKG_NAME_RE.match(line)
        if name_match is None:
            continue
        _, marker = _split_environment_marker(line)
        if marker:
            markers[_canonicalize_name(name_match.group(0))] = marker
    return markers


def _inject_marker(line: str, marker: str) -> str:
    """Attach ``marker`` to a compiled requirement pin line."""
    newline = "\n" if line.endswith("\n") else ""
    body = line[: -len(newline)] if newline else line
    continuation = ""
    if body.endswith("\\"):
        continuation = " \\"
        body = body[:-1].rstrip()
    spec, _existing_marker = _split_environment_marker(body)
    return f"{spec} ; {marker}{continuation}{newline}"


def apply_environment_markers(requirements_in: Path, requirements_txt: Path) -> None:
    """Copy PEP 508 environment markers from ``requirements.in`` onto ``requirements.txt``.

    ``uv pip compile`` evaluates markers against the compile-time environment (the
    Linux container host) and drops those that are true there, such as
    ``platform_machine != 's390x'`` on x86_64. Restore the input markers so the
    lockfile still skips those packages on other architectures.
    """
    markers = read_requirement_markers(requirements_in)
    if not markers:
        return
    if not requirements_txt.is_file():
        raise RefreshRequirementsError(f"Missing {_REQUIREMENTS_TXT}: {requirements_txt}")

    updated_lines: list[str] = []
    for line in requirements_txt.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.strip() or line[:1].isspace() or line.lstrip().startswith(("#", "-")):
            updated_lines.append(line)
            continue
        name_match = _PKG_NAME_RE.match(line)
        if name_match is None:
            updated_lines.append(line)
            continue
        marker = markers.get(_canonicalize_name(name_match.group(0)))
        updated_lines.append(_inject_marker(line, marker) if marker else line)

    requirements_txt.write_text("".join(updated_lines), encoding="utf-8")


def sanitize_index_url_for_log(url: str) -> str:
    """Return a log-safe index URL with credentials, path, and query removed."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, "", "", parts.fragment))


def resolve_pipeline_dir(repo_root: Path, pipeline: str | Path) -> Path:
    """Resolve a pipeline directory and validate required input files exist."""
    pipeline_dir = Path(pipeline)
    if not pipeline_dir.is_absolute():
        pipeline_dir = repo_root / pipeline_dir

    requirements_in = pipeline_dir / _REQUIREMENTS_IN
    if not requirements_in.is_file():
        raise RefreshRequirementsError(f"Missing {_REQUIREMENTS_IN}: {requirements_in}")

    return pipeline_dir.resolve()


def _require_runtime(runtime: str) -> str:
    """Validate and return a supported container runtime available on PATH."""
    if runtime not in SUPPORTED_RUNTIMES:
        supported = ", ".join(SUPPORTED_RUNTIMES)
        raise RefreshRequirementsError(f"Unsupported container runtime: {runtime} (supported: {supported})")
    if shutil.which(runtime) is None:
        raise RefreshRequirementsError(f"Container runtime not found on PATH: {runtime}")
    return runtime


def resolve_container_runtime(runtime: str | None = None) -> str:
    """Resolve the container runtime from CLI, environment, or auto-detection.

    Priority:
    1. Explicit ``runtime`` argument (for example ``--runtime docker``)
    2. ``CONTAINER_RUNTIME`` environment variable
    3. Auto-detect: podman, then docker
    """
    if runtime is not None:
        return _require_runtime(runtime)

    env_runtime = os.environ.get(_CONTAINER_RUNTIME_ENV)
    if env_runtime is not None:
        return _require_runtime(env_runtime)

    for candidate in SUPPORTED_RUNTIMES:
        if shutil.which(candidate) is not None:
            return candidate

    supported = ", ".join(SUPPORTED_RUNTIMES)
    raise RefreshRequirementsError(
        f"No supported container runtime found on PATH (tried: {supported}). Install Podman or Docker."
    )


def build_volume_mount(pipeline_dir: Path) -> str:
    """Build a container volume mount string for the pipeline directory."""
    return f"{pipeline_dir}:{pipeline_dir}"


def build_container_command(
    *,
    runtime: str,
    container_image: str,
    pipeline_dir: Path,
    upgrade: bool,
    dry_run: bool,
    verbose: bool,
) -> list[str]:
    """Build the container command that runs uv pip compile."""
    python_bin = "python3 -u"
    compile_flags = [
        f"{python_bin} -m uv pip compile",
        _REQUIREMENTS_IN,
        "--generate-hashes",
        "--emit-index-url",
        "--no-header",
    ]
    # uv pip compile has no --dry-run; emit to stdout instead of writing the lockfile.
    if not dry_run:
        compile_flags.append(f"--output-file {_REQUIREMENTS_TXT}")
    if upgrade:
        compile_flags.append("--upgrade")
    if not verbose:
        compile_flags.append("--quiet")

    pip_install = f"{python_bin} -m pip install uv" if verbose else f"{python_bin} -m pip install --quiet uv"
    compile_command = " ".join([pip_install, "&&", " ".join(compile_flags)])

    command = [
        runtime,
        "run",
        "--rm",
        "-v",
        build_volume_mount(pipeline_dir),
        "-w",
        str(pipeline_dir),
    ]
    command.extend([container_image, "/bin/bash", "-lc", compile_command])
    return command


def compile_pipeline_requirements(
    pipeline_dir: Path,
    *,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_runtime: str | None = None,
    upgrade: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Compile ``requirements.in`` to ``requirements.txt`` for one pipeline."""
    requirements_in = pipeline_dir / _REQUIREMENTS_IN
    index_url = read_index_url(requirements_in)
    if index_url is None:
        raise RefreshRequirementsError(f"{requirements_in} must declare --index-url for RHOAI package resolution")

    runtime = resolve_container_runtime(container_runtime)
    command = build_container_command(
        runtime=runtime,
        container_image=container_image,
        pipeline_dir=pipeline_dir,
        upgrade=upgrade,
        dry_run=dry_run,
        verbose=verbose,
    )

    print(f"Refreshing {pipeline_dir / _REQUIREMENTS_TXT}")
    print(f"  index-url: {sanitize_index_url_for_log(index_url)}")
    print(f"  runtime: {runtime}")
    print(f"  image: {container_image}")
    run_kwargs: dict[str, object] = {"check": True, "timeout": 3600}
    if not verbose:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    subprocess.run(command, **run_kwargs)

    if not dry_run:
        apply_environment_markers(requirements_in, pipeline_dir / _REQUIREMENTS_TXT)


def refresh_pipeline_requirements(
    pipelines: Sequence[str | Path],
    *,
    repo_root: Path | None = None,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_runtime: str | None = None,
    upgrade: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Refresh requirements.txt for all given pipeline directories."""
    if repo_root is None:
        repo_root = get_repo_root()

    for pipeline in pipelines:
        pipeline_dir = resolve_pipeline_dir(repo_root, pipeline)
        compile_pipeline_requirements(
            pipeline_dir,
            container_image=container_image,
            container_runtime=container_runtime,
            upgrade=upgrade,
            dry_run=dry_run,
            verbose=verbose,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Hermeto-compatible requirements.txt lockfiles for RHOAI pipelines "
            "using uv pip compile in Podman or Docker"
        ),
    )
    parser.add_argument(
        "pipelines",
        nargs="*",
        help=(f"Pipeline directories containing requirements.in (default: {', '.join(DEFAULT_PIPELINES)})"),
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_CONTAINER_IMAGE,
        help=f"Container image to run uv pip compile in (default: {DEFAULT_CONTAINER_IMAGE})",
    )
    parser.add_argument(
        "--runtime",
        choices=SUPPORTED_RUNTIMES,
        help=("Container runtime to use (default: $CONTAINER_RUNTIME, otherwise auto-detect podman then docker)"),
    )
    parser.add_argument(
        "--no-upgrade",
        action="store_true",
        help="Keep existing pins from requirements.txt when possible instead of upgrading all dependencies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resolved requirements on stdout without writing requirements.txt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live uv pip compile progress output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    pipelines = args.pipelines or list(DEFAULT_PIPELINES)

    try:
        refresh_pipeline_requirements(
            pipelines,
            container_image=args.image,
            container_runtime=args.runtime,
            upgrade=not args.no_upgrade,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except (RefreshRequirementsError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
