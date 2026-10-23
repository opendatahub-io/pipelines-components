#!/usr/bin/env python3
"""Verify Hermeto lockfiles cover every architecture wheel published on the RHAI index.

The RHAI PyPI index publishes wheels for one version in a staggered fashion (for
example the ``ppc64le`` wheel of ``pillow==13.0.0`` appeared a day after the other
architectures). When ``uv pip compile --generate-hashes`` runs while the index is
incomplete, the lockfile silently misses the ``--hash`` entries for the wheels that
were not published yet, and Hermeto offline builds then fail on those architectures.

This module detects such situations by comparing a compiled ``requirements.txt``
against the current index state:

1. Arch-less packages (``py3-none-any`` wheels, sdists) carry no architecture
   expectation; every published file must still be hashed in the lockfile.
2. Packages with architecture-specific wheels must have hashes for the wheels of
   every architecture the index publishes, and the index itself must publish
   wheels for all supported Linux architectures (``x86_64``, ``aarch64``,
   ``ppc64le``, ``s390x``).
3. Packages carrying a ``platform_machine`` environment marker (in
   ``requirements.in`` or on the lockfile pin) only need the architectures the
   marker allows; marker-excluded architectures are skipped.

See: https://peps.python.org/pep-0691/ (simple index JSON API).
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from packaging.markers import InvalidMarker, Marker, default_environment
from packaging.version import InvalidVersion, Version

#: Linux architectures RHOAI pipelines must support with hashed wheels.
LINUX_ARCHES: tuple[str, ...] = ("x86_64", "aarch64", "ppc64le", "s390x")

#: Fetch attempts per package before an index lookup is reported as a failure.
_FETCH_ATTEMPTS = 3

#: Classification for files that are not architecture-specific.
ARCH_LESS = "any"  # wheels with an ``any`` platform tag (for example py3-none-any)
SOURCE = "source"  # sdists
OTHER_PLATFORM = "other"  # wheels for non-Linux platforms or unparsable names

#: Python version of the UBI9 compile container; markers are evaluated against it.
_COMPILE_PYTHON_VERSION = "3.12"

_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?P<extras>\[[^\]]*\])?"
    r"==(?P<version>[^\s;\\]+)"
    r"(?:\s*;\s*(?P<marker>.+?))?"
    r"\s*\\?$"
)
_HASH_RE = re.compile(r"^\s*--hash=(?P<algorithm>[a-z0-9]+):(?P<digest>[0-9a-fA-F]+)\s*\\?$")
# Wheel names are escaped per PEP 427 (runs of non-alphanumeric characters become
# underscores), so the name group must not accept dashes; versions never contain
# dashes either. RHAI additionally inserts a numeric build tag after the version.
_WHEEL_FILENAME_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._]*)-(?P<version>[^-]+)"
    r"(?:-(?P<build>[^-]+))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)
_SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar")

_ARCH_CHECK_MODES = ("error", "warn")


class IndexLookupError(Exception):
    """Raised when index metadata for a package cannot be fetched or parsed."""


@dataclass(frozen=True)
class IndexFile:
    """One file published on the simple index for some package version."""

    filename: str
    sha256: str | None
    arch: str


@dataclass
class PinnedRequirement:
    """One pinned package parsed from a compiled requirements.txt."""

    name: str  # PEP 503 canonical name
    display: str  # "name==version" as shown in findings
    version: str
    marker: str | None
    hashes: set[str] = field(default_factory=set)  # "algorithm:digest" strings


@dataclass(frozen=True)
class Finding:
    """One verification result for a pinned package."""

    severity: str  # "error" or "warning"
    package: str
    message: str


def _canonicalize_name(name: str) -> str:
    """Normalize a package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lockfile_pins(requirements_txt: Path) -> list[PinnedRequirement]:
    """Parse pinned packages, markers, and hashes from a compiled requirements.txt."""
    pins: list[PinnedRequirement] = []
    current: PinnedRequirement | None = None
    for raw_line in requirements_txt.read_text(encoding="utf-8").splitlines():
        pin_match = _PIN_RE.match(raw_line)
        if pin_match is not None:
            name = pin_match.group("name")
            version = pin_match.group("version")
            current = PinnedRequirement(
                name=_canonicalize_name(name),
                display=f"{name}{pin_match.group('extras') or ''}=={version}",
                version=version,
                marker=pin_match.group("marker"),
            )
            pins.append(current)
            continue
        hash_match = _HASH_RE.match(raw_line)
        if hash_match is not None and current is not None:
            current.hashes.add(f"{hash_match.group('algorithm')}:{hash_match.group('digest')}")
    return pins


def file_arch(filename: str) -> str:
    """Classify an index file as an architecture, arch-less wheel, sdist, or other.

    Returns one of ``LINUX_ARCHES``, ``ARCH_LESS``, ``SOURCE``, or
    ``OTHER_PLATFORM``.
    """
    wheel_match = _WHEEL_FILENAME_RE.match(filename)
    if wheel_match is not None:
        for tag in wheel_match.group("platform").split("."):
            if tag == "any":
                return ARCH_LESS
            for arch in LINUX_ARCHES:
                if tag.endswith(arch):
                    return arch
        return OTHER_PLATFORM
    if filename.endswith(_SDIST_SUFFIXES):
        return SOURCE
    return OTHER_PLATFORM


def _parse_filename(filename: str) -> tuple[str, str] | None:
    """Return (name, version) parsed from a wheel or sdist filename, else None."""
    wheel_match = _WHEEL_FILENAME_RE.match(filename)
    if wheel_match is not None:
        return wheel_match.group("name"), wheel_match.group("version")
    for suffix in _SDIST_SUFFIXES:
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
            name, _, version = stem.rpartition("-")
            if name and version:
                return name, version
    return None


def _versions_equal(left: str, right: str) -> bool:
    """Compare two version strings, tolerating PEP 440 normalization (2.17 vs 2.17.0)."""
    if left == right:
        return True
    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return False


def _files_for_version(files: Sequence[IndexFile], name: str, version: str) -> list[IndexFile]:
    """Filter index files down to those matching the pinned name and version."""
    canonical = _canonicalize_name(name)
    matched: list[IndexFile] = []
    for index_file in files:
        parsed = _parse_filename(index_file.filename)
        if parsed is None:
            continue
        file_name, file_version = parsed
        if _canonicalize_name(file_name) != canonical:
            continue
        if _versions_equal(file_version, version):
            matched.append(index_file)
    return matched


def _marker_environment(arch: str) -> dict[str, str]:
    """Build a PEP 508 marker environment for one machine architecture.

    The environment mirrors the UBI9 Python 3.12 compile container so markers
    evaluate exactly as they would during ``uv pip compile``.
    """
    environment = dict(default_environment())
    environment.update(
        {
            "implementation_name": "cpython",
            "implementation_version": f"{_COMPILE_PYTHON_VERSION}.0",
            "os_name": "posix",
            "platform_machine": arch,
            "platform_python_implementation": "CPython",
            "platform_release": "",
            "platform_system": "Linux",
            "platform_version": "",
            "python_full_version": f"{_COMPILE_PYTHON_VERSION}.0",
            "python_version": _COMPILE_PYTHON_VERSION,
            "sys_platform": "linux",
        }
    )
    return environment


def marker_allows_arch(marker: str, arch: str) -> bool:
    """Evaluate an environment marker for one machine architecture."""
    return Marker(marker).evaluate(_marker_environment(arch))


def allowed_arches(marker: str | None) -> set[str]:
    """Return the Linux architectures a marker permits (all when there is no marker)."""
    if not marker:
        return set(LINUX_ARCHES)
    return {arch for arch in LINUX_ARCHES if marker_allows_arch(marker, arch)}


class SimpleIndexClient:
    """Fetch package file listings from a PEP 691 simple index with caching."""

    def __init__(self, index_url: str, *, timeout: float = 30.0) -> None:
        """Store the index base URL and per-request timeout."""
        self.index_url = index_url.rstrip("/")
        self.timeout = timeout
        self._cache: dict[str, list[IndexFile]] = {}
        self._lock = threading.Lock()

    def files(self, package: str) -> list[IndexFile]:
        """Return every file the index publishes for a package (all versions)."""
        canonical = _canonicalize_name(package)
        with self._lock:
            cached = self._cache.get(canonical)
        if cached is not None:
            return cached

        url = f"{self.index_url}/{canonical}/"
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.pypi.simple.v1+json"})
        last_error: Exception | None = None
        for attempt in range(_FETCH_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # Transient network blips (DNS hiccups, resets) fail the whole
                # verification otherwise; retry before giving up.
                last_error = exc
                if attempt + 1 < _FETCH_ATTEMPTS:
                    time.sleep(1)
            except json.JSONDecodeError as exc:
                raise IndexLookupError(f"{url}: index did not return JSON: {exc}") from exc
        else:
            raise IndexLookupError(f"{url}: {last_error}") from last_error

        files = [
            IndexFile(
                filename=entry["filename"],
                sha256=entry.get("hashes", {}).get("sha256"),
                arch=file_arch(entry["filename"]),
            )
            for entry in payload.get("files", [])
        ]
        with self._lock:
            self._cache[canonical] = files
        return files


def _verify_pin(
    pin: PinnedRequirement,
    files: Sequence[IndexFile],
    *,
    input_markers: Mapping[str, str],
    arch_check: str,
) -> list[Finding]:
    """Verify one pinned package against its index file listing."""
    findings: list[Finding] = []
    version_files = _files_for_version(files, pin.name, pin.version)
    if not version_files:
        findings.append(Finding("error", pin.display, "the index publishes no files for this version"))
        return findings

    marker = pin.marker or input_markers.get(pin.name)
    try:
        expected = allowed_arches(marker)
    except InvalidMarker as exc:
        findings.append(Finding("error", pin.display, f"invalid environment marker {marker!r}: {exc}"))
        expected = set(LINUX_ARCHES)

    # Every published file for an expected architecture (plus arch-less wheels and
    # sdists) must carry a hash in the lockfile; a missing hash means the lockfile
    # was compiled while the index was incomplete.
    for index_file in version_files:
        if index_file.arch in LINUX_ARCHES and index_file.arch not in expected:
            continue
        if index_file.sha256 is None:
            findings.append(Finding("warning", pin.display, f"{index_file.filename} has no sha256 digest on the index"))
            continue
        if f"sha256:{index_file.sha256}" not in pin.hashes:
            findings.append(
                Finding("error", pin.display, f"missing --hash for {index_file.filename} published on the index")
            )

    # Hashes that match no published file mean the index changed since compile.
    index_hashes = {f"sha256:{index_file.sha256}" for index_file in version_files if index_file.sha256}
    for stale in sorted(pin.hashes - index_hashes):
        findings.append(Finding("error", pin.display, f"--hash={stale} matches no file published on the index"))

    # Architecture-specific packages need wheels for every expected architecture.
    published = {index_file.arch for index_file in version_files if index_file.arch in LINUX_ARCHES}
    missing = expected - published
    if published and missing:
        severity = "error" if arch_check == "error" else "warning"
        findings.append(
            Finding(
                severity,
                pin.display,
                f"index publishes architecture-specific wheels only for {', '.join(sorted(published))} "
                f"and none for {', '.join(sorted(missing))}",
            )
        )
    return findings


def verify_lockfile_coverage(
    requirements_txt: Path,
    *,
    client: SimpleIndexClient,
    input_markers: Mapping[str, str] | None = None,
    arch_check: str = "error",
    max_workers: int = 8,
) -> list[Finding]:
    """Verify a compiled lockfile against the index it was resolved from.

    Args:
        requirements_txt: Compiled lockfile with pins and ``--hash`` entries.
        client: Simple index client for the index the lockfile was resolved from.
        input_markers: Canonical package name to marker from ``requirements.in``;
            used as a fallback when the lockfile pin carries no marker.
        arch_check: Severity ("error" or "warn") for index-side architecture gaps.
        max_workers: Concurrent index lookups.

    Returns:
        Findings in lockfile order; empty means the lockfile fully covers the index.

    Raises:
        ValueError: If ``arch_check`` is not a supported mode.
    """
    if arch_check not in _ARCH_CHECK_MODES:
        raise ValueError(f"Unsupported arch check mode: {arch_check} (supported: {', '.join(_ARCH_CHECK_MODES)})")

    pins = parse_lockfile_pins(requirements_txt)
    input_markers = input_markers or {}

    def fetch(pin: PinnedRequirement) -> tuple[Sequence[IndexFile] | None, str | None]:
        try:
            return client.files(pin.name), None
        except IndexLookupError as exc:
            return None, str(exc)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fetched = list(executor.map(fetch, pins))

    findings: list[Finding] = []
    for pin, (files, error) in zip(pins, fetched, strict=True):
        if error is not None:
            findings.append(Finding("error", pin.display, f"could not fetch index metadata: {error}"))
            continue
        assert files is not None  # guaranteed by the fetch wrapper
        findings.extend(_verify_pin(pin, files, input_markers=input_markers, arch_check=arch_check))
    return findings
