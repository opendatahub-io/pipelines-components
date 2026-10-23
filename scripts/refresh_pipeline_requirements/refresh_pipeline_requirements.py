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

After compiling, the lockfile is verified against the index (see
``verify_lockfile``): every wheel the index publishes for a pinned version must
carry a ``--hash`` entry, and architecture-specific packages must cover all
supported Linux architectures unless a ``platform_machine`` marker excludes
some. This detects lockfiles compiled while the RHAI index was incomplete.

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
from .verify_lockfile import SimpleIndexClient, verify_lockfile_coverage

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


def verify_pipeline_lockfile(pipeline_dir: Path, *, arch_check: str = "error") -> None:
    """Verify a pipeline's requirements.txt against the index it was resolved from.

    Raises ``RefreshRequirementsError`` when any error-severity finding is reported
    (missing hashes, stale hashes, or incomplete index architecture coverage).
    """
    requirements_in = pipeline_dir / _REQUIREMENTS_IN
    requirements_txt = pipeline_dir / _REQUIREMENTS_TXT
    index_url = read_index_url(requirements_in)
    if index_url is None:
        raise RefreshRequirementsError(f"{requirements_in} must declare --index-url for lockfile verification")
    if not requirements_txt.is_file():
        raise RefreshRequirementsError(f"Missing {_REQUIREMENTS_TXT}: {requirements_txt}")

    print(f"Verifying {requirements_txt}")
    print(f"  index-url: {sanitize_index_url_for_log(index_url)}")
    findings = verify_lockfile_coverage(
        requirements_txt,
        client=SimpleIndexClient(index_url),
        input_markers=read_requirement_markers(requirements_in),
        arch_check=arch_check,
    )
    errors = 0
    for finding in findings:
        print(f"  {finding.severity}: {finding.package}: {finding.message}", flush=True)
        if finding.severity == "error":
            errors += 1
    if errors:
        raise RefreshRequirementsError(
            f"lockfile verification failed with {errors} error(s); the lockfile is out of sync with the index "
            "(it was likely compiled while the index was incomplete) - re-run the refresh once the index is complete"
        )
    if findings:
        print(f"  {len(findings)} warning(s); lockfile otherwise in sync with the index")


def verify_pipeline_requirements(
    pipelines: Sequence[str | Path],
    *,
    repo_root: Path | None = None,
    arch_check: str = "error",
) -> None:
    """Verify existing requirements.txt lockfiles without recompiling them.

    Every pipeline is verified even when earlier ones fail; the error is raised
    only after all findings have been reported.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    failures: list[str] = []
    for pipeline in pipelines:
        try:
            pipeline_dir = resolve_pipeline_dir(repo_root, pipeline)
            verify_pipeline_lockfile(pipeline_dir, arch_check=arch_check)
        except RefreshRequirementsError as exc:
            print(f"error: {pipeline}: {exc}", file=sys.stderr)
            failures.append(str(pipeline))
    if failures:
        raise RefreshRequirementsError(
            f"lockfile verification failed for {len(failures)} of {len(pipelines)} pipeline(s); see findings above"
        )


def strip_lockfile_hashes(requirements_txt: Path) -> None:
    """Remove ``--hash`` entries from a lockfile in place.

    ``uv pip compile`` without ``--upgrade`` keeps the hash entries of unchanged
    pins from the existing output file, so a no-upgrade refresh cannot repair a
    hash set that was compiled against an incomplete index. Stripping the hashes
    first forces uv to regenerate them from the current index while keeping the
    pinned versions.
    """
    lines = requirements_txt.read_text(encoding="utf-8").splitlines()
    # Drop hash lines, then remove the now dangling "\" continuations from the
    # pin lines that carried them (only hash lines followed a continuation).
    kept = [line for line in lines if not line.lstrip().startswith("--hash=")]
    kept = [line[:-2].rstrip() if line.endswith(" \\") else line for line in kept]
    requirements_txt.write_text("\n".join(kept) + "\n", encoding="utf-8")


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
    verify: bool = True,
    arch_check: str = "error",
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

    # Strip hashes only once the runtime resolved; keep the original content so
    # a failed compile restores the lockfile instead of leaving it hash-less.
    requirements_txt = pipeline_dir / _REQUIREMENTS_TXT
    original_lockfile: str | None = None
    if not dry_run and not upgrade and requirements_txt.is_file():
        original_lockfile = requirements_txt.read_text(encoding="utf-8")
        strip_lockfile_hashes(requirements_txt)

    print(f"Refreshing {pipeline_dir / _REQUIREMENTS_TXT}")
    print(f"  index-url: {sanitize_index_url_for_log(index_url)}")
    print(f"  runtime: {runtime}")
    print(f"  image: {container_image}")
    run_kwargs: dict[str, object] = {"check": True, "timeout": 3600}
    if not verbose:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    try:
        subprocess.run(command, **run_kwargs)
    except BaseException:
        if original_lockfile is not None:
            requirements_txt.write_text(original_lockfile, encoding="utf-8")
        raise

    if not dry_run:
        apply_environment_markers(requirements_in, pipeline_dir / _REQUIREMENTS_TXT)
        if verify:
            verify_pipeline_lockfile(pipeline_dir, arch_check=arch_check)


def refresh_pipeline_requirements(
    pipelines: Sequence[str | Path],
    *,
    repo_root: Path | None = None,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_runtime: str | None = None,
    upgrade: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    verify: bool = True,
    arch_check: str = "error",
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
            verify=verify,
            arch_check=arch_check,
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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing requirements.txt files against the index without recompiling (no container needed)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip lockfile verification after compiling",
    )
    parser.add_argument(
        "--arch-check",
        choices=("error", "warn"),
        default="error",
        help=("Severity when the index lacks wheels for some architectures of a pinned package (default: error)"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    pipelines = args.pipelines or list(DEFAULT_PIPELINES)

    try:
        if args.verify_only:
            verify_pipeline_requirements(pipelines, arch_check=args.arch_check)
        else:
            refresh_pipeline_requirements(
                pipelines,
                container_image=args.image,
                container_runtime=args.runtime,
                upgrade=not args.no_upgrade,
                dry_run=args.dry_run,
                verbose=not args.quiet,
                verify=not args.no_verify,
                arch_check=args.arch_check,
            )
    except (RefreshRequirementsError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
