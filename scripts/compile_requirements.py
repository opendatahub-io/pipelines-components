"""Generate requirements lockfiles with multi-arch hashes from the AIPCC index.

Resolves dependencies with uv pip compile, then fetches per-architecture
wheel hashes directly from the AIPCC simple index API. No Docker required.

Usage:
    python scripts/compile_requirements.py
"""

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

AIPCC_INDEX_URL = "https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple"

PYTHON_VERSION = "3.12"

ARCHES = ["x86_64", "aarch64", "ppc64le", "s390x"]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_build_requires() -> list[str]:
    """Extract [build-system] requires from pyproject.toml."""
    content = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r"\[build-system\].*?requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not match:
        raise SystemExit("Could not find [build-system] requires in pyproject.toml")
    return [dep.strip().strip("\"'") for dep in match.group(1).split(",") if dep.strip().strip("\"'")]


def log(msg: str) -> None:
    """Print a progress message to stderr."""
    print(msg, file=sys.stderr, flush=True)


def _canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve_packages(
    in_files: list[str] | None = None,
    packages: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Run uv pip compile to resolve package names and versions.

    Either in_files (paths relative to REPO_ROOT) or packages (list of
    package specifiers) must be provided.
    """
    extra_args = extra_args or []
    cmd = [
        "uv",
        "pip",
        "compile",
        "--no-header",
        "--no-annotate",
        "--python-version",
        PYTHON_VERSION,
        "--index-url",
        AIPCC_INDEX_URL,
        *extra_args,
    ]

    stdin_input = None
    if in_files:
        cmd.extend(str(REPO_ROOT / f) for f in in_files)
    elif packages:
        stdin_input = "\n".join(packages)
        cmd.append("-")

    log(f"  running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            input=stdin_input,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit("uv pip compile timed out after 120s. Check AIPCC index connectivity.")
    if result.returncode != 0:
        log(result.stderr)
        raise SystemExit(f"uv pip compile failed (exit {result.returncode}).")

    resolved = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        m = re.match(r"([a-zA-Z0-9_.-]+)==([^\s;]+)\s*(;.*)?", line)
        if m:
            resolved.append((m.group(1), m.group(2), (m.group(3) or "").strip()))

    return resolved


def fetch_hashes_from_index(name: str, version: str) -> list[str]:
    """Fetch wheel hashes for all target architectures from the AIPCC simple index."""
    canon = _canonicalize(name)
    url = f"{AIPCC_INDEX_URL}/{canon}/"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            html = resp.read().decode()
    except Exception as e:
        log(f"  WARNING: could not fetch index for {name}: {e}")
        return []

    hashes = []
    version_escaped = re.escape(version)
    name_pattern = re.sub(r"[-_.]", "[-_.]", canon)

    for match in re.finditer(
        rf'href="[^"]*({name_pattern}-{version_escaped}(?:-\d+)?-[^"]*\.whl)#sha256=([a-f0-9]+)"',
        html,
        re.IGNORECASE,
    ):
        wheel_name = match.group(1)
        sha = match.group(2)

        is_any = "none-any" in wheel_name
        is_target_arch = any(arch in wheel_name for arch in ARCHES)
        is_abi3 = "abi3" in wheel_name
        is_cp = f"cp{PYTHON_VERSION.replace('.', '')}" in wheel_name

        if is_any or (is_target_arch and (is_abi3 or is_cp)):
            hashes.append(f"sha256:{sha}")

    return sorted(set(hashes))


def write_lockfile(out_file: str, resolved: list[tuple[str, str, str]]) -> None:
    """Fetch multi-arch hashes and write the lockfile."""
    lines = [
        f"--index-url {AIPCC_INDEX_URL}",
        "",
    ]

    for name, version, marker in resolved:
        hashes = fetch_hashes_from_index(name, version)
        marker_part = f" {marker}" if marker else ""

        if not hashes:
            log(f"  WARNING: no matching wheels found for {name}=={version}")
            lines.append(f"{name}=={version}{marker_part}")
            continue

        lines.append(f"{name}=={version}{marker_part} \\")
        for i, h in enumerate(hashes):
            suffix = " \\" if i < len(hashes) - 1 else ""
            lines.append(f"    --hash={h}{suffix}")

    out_path = REPO_ROOT / out_file
    out_path.write_text("\n".join(lines) + "\n")
    log(f"  wrote {out_file} ({', '.join(ARCHES)})")


def main() -> None:
    """Generate requirements.txt and requirements-build.txt with multi-arch AIPCC hashes."""
    log("Compiling requirements ...")
    resolved = resolve_packages(
        in_files=["pyproject.toml"],
        extra_args=["--no-emit-package", "kfp-components"],
    )
    log(f"  resolved {len(resolved)} packages, fetching multi-arch hashes ...")
    write_lockfile("requirements.txt", resolved)

    log("Compiling requirements-build ...")
    build_deps = _parse_build_requires()
    log(f"  build-system requires: {build_deps}")
    resolved = resolve_packages(packages=build_deps)
    log(f"  resolved {len(resolved)} packages, fetching multi-arch hashes ...")
    write_lockfile("requirements-build.txt", resolved)

    log("Done.")


if __name__ == "__main__":
    main()
