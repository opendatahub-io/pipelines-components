#!/usr/bin/env python3
"""Refresh Hermeto-compatible requirements.txt lockfiles for RHOAI pipelines.

Compiles ``requirements.in`` into ``requirements.txt`` using ``pip-compile``
with ``--generate-hashes`` inside a container (Podman or Docker). The RHOAI PyPI
index does not publish macOS-compatible wheels, so compilation must run on Linux
(UBI9 Python 3.12).

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
# RHEL UBI9 (glibc 2.34); matches Makefile PYTHON_PLATFORM
DEFAULT_PYTHON_PLATFORM = "x86_64-manylinux_2_34"
DEFAULT_PYTHON_VERSION = "3.12"
# Install pip-tools from PyPI; AIPCC/RHOAI base images often pin pip to an index that
# does not publish pip-tools. pip-compile still uses --index-url from requirements.in.
PIP_TOOLS_INDEX_URL = "https://pypi.org/simple"
DEFAULT_PIPELINES: tuple[str, ...] = (
    "pipelines/training/automl/autogluon_tabular_training_pipeline",
    "pipelines/training/autorag/documents_rag_optimization_pipeline",
)
SUPPORTED_RUNTIMES: tuple[str, ...] = ("podman", "docker")
_CONTAINER_RUNTIME_ENV = "CONTAINER_RUNTIME"

_INDEX_URL_RE = re.compile(r"^--index-url\s+(\S+)", re.MULTILINE)
_REQUIREMENTS_IN = "requirements.in"
_REQUIREMENTS_TXT = "requirements.txt"


class RefreshRequirementsError(Exception):
    """Raised when requirements refresh cannot proceed."""


def read_index_url(requirements_in: Path) -> str | None:
    """Return the ``--index-url`` value from a requirements input file."""
    content = requirements_in.read_text(encoding="utf-8")
    match = _INDEX_URL_RE.search(content)
    return match.group(1) if match else None


def sanitize_index_url_for_log(url: str) -> str:
    """Return a log-safe index URL with credentials, path, and query removed."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, "", "", parts.fragment))


def build_pip_platform_args(python_platform: str, python_version: str) -> str:
    """Build pip resolver args from a uv-style platform string.

    Converts ``x86_64-manylinux_2_34`` to pip's ``--platform manylinux_2_34_x86_64``
    format, matching ``uv pip compile --python-platform``.
    """
    arch, separator, platform_tag = python_platform.partition("-")
    if not separator or not arch or not platform_tag:
        raise RefreshRequirementsError(
            f"Invalid python platform {python_platform!r}; expected format like x86_64-manylinux_2_34"
        )

    major, _, minor = python_version.partition(".")
    if not major.isdigit() or not minor.isdigit():
        raise RefreshRequirementsError(f"Invalid python version {python_version!r}; expected format like 3.12")

    pip_platform = f"{platform_tag}_{arch}"
    abi = f"cp{major}{minor}"
    return f"--platform {pip_platform} --python-version {python_version} --implementation cp --abi {abi}"


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
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
    python_version: str = DEFAULT_PYTHON_VERSION,
) -> list[str]:
    """Build the container command that runs pip-compile."""
    python_bin = "python3 -u"
    pip_platform_args = build_pip_platform_args(python_platform, python_version)
    compile_flags = [
        f"{python_bin} -m piptools compile",
        _REQUIREMENTS_IN,
        "--generate-hashes",
        "--emit-index-url",
        "--allow-unsafe",
        "--no-header",
        f"--pip-args '{pip_platform_args}'",
        f"--output-file {_REQUIREMENTS_TXT}",
    ]
    if upgrade:
        compile_flags.append("--upgrade")
    if dry_run:
        compile_flags.append("--dry-run")
    if verbose:
        compile_flags.append("-v")

    pip_install_flags = f"--index-url {PIP_TOOLS_INDEX_URL}"
    if not verbose:
        pip_install_flags = f"--quiet {pip_install_flags}"
    pip_install = f"{python_bin} -m pip install {pip_install_flags} pip-tools"
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
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
    python_version: str = DEFAULT_PYTHON_VERSION,
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
        python_platform=python_platform,
        python_version=python_version,
    )

    print(f"Refreshing {pipeline_dir / _REQUIREMENTS_TXT}")
    print(f"  index-url: {sanitize_index_url_for_log(index_url)}")
    print(f"  runtime: {runtime}")
    print(f"  image: {container_image}")
    print(f"  python-platform: {python_platform}")
    run_kwargs: dict[str, object] = {"check": True, "timeout": 3600}
    if not verbose:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    subprocess.run(command, **run_kwargs)


def refresh_pipeline_requirements(
    pipelines: Sequence[str | Path],
    *,
    repo_root: Path | None = None,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_runtime: str | None = None,
    upgrade: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
    python_version: str = DEFAULT_PYTHON_VERSION,
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
            python_platform=python_platform,
            python_version=python_version,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Hermeto-compatible requirements.txt lockfiles for RHOAI pipelines "
            "using pip-compile in Podman or Docker"
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
        help=f"Container image to run pip-compile in (default: {DEFAULT_CONTAINER_IMAGE})",
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
        help="Show what pip-compile would change without writing requirements.txt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live pip-compile progress output",
    )
    parser.add_argument(
        "--python-platform",
        default=DEFAULT_PYTHON_PLATFORM,
        help=(f"Target platform for wheel resolution in uv format (default: {DEFAULT_PYTHON_PLATFORM})"),
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_VERSION,
        help=f"Target Python version for wheel resolution (default: {DEFAULT_PYTHON_VERSION})",
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
            python_platform=args.python_platform,
            python_version=args.python_version,
        )
    except (RefreshRequirementsError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
