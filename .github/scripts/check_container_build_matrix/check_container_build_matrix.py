"""Check that every Containerfile/Dockerfile has a matching entry in container-build.yml.

This ensures contributors do not add a Containerfile without registering it in
the container-build matrix, which would cause CI to silently skip building it.
"""

import argparse
import sys
from pathlib import Path

import yaml

SEARCH_ROOTS = ["components", "pipelines", "docs/examples"]
CONTAINER_FILENAMES = {"Containerfile", "Dockerfile"}
IGNORE_FILENAME = ".container-build-ignore"
WORKFLOW_PATH = ".github/workflows/container-build.yml"


def get_repo_root() -> Path:
    """Locate and return the repository root directory."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / WORKFLOW_PATH).is_file():
            return parent
    raise RuntimeError(f"Could not locate repo root containing {WORKFLOW_PATH}")


def discover_container_files(repo_root: Path, search_roots: list[str]) -> list[Path]:
    """Recursively find all Containerfile/Dockerfile paths under search_roots."""
    found = []
    for root in search_roots:
        base = repo_root / root
        if not base.exists():
            continue
        for name in CONTAINER_FILENAMES:
            found.extend(base.rglob(name))
    return sorted(found)


def load_ignore_list(repo_root: Path) -> set[str]:
    """Load directories to ignore from IGNORE_FILENAME at the repo root.

    Each non-empty, non-comment line is treated as a path relative to the
    repo root (e.g. components/foo/bar).
    """
    ignore_file = repo_root / IGNORE_FILENAME
    if not ignore_file.exists():
        return set()
    ignored = set()
    for line in ignore_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ignored.add(str(Path(line)))
    return ignored


def parse_matrix_contexts(workflow_path: Path) -> set[str]:
    """Extract context values from jobs.build.strategy.matrix.include."""
    try:
        with open(workflow_path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"Error: failed to parse workflow YAML at {workflow_path}: {exc}", file=sys.stderr)
        return set()
    if not isinstance(workflow, dict):
        return set()
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return set()
    build = jobs.get("build")
    if not isinstance(build, dict):
        return set()
    strategy = build.get("strategy")
    if not isinstance(strategy, dict):
        return set()
    matrix = strategy.get("matrix")
    if not isinstance(matrix, dict):
        return set()
    includes = matrix.get("include", [])
    if not isinstance(includes, list):
        return set()

    contexts = set()
    for entry in includes:
        if not isinstance(entry, dict):
            continue
        if "context" in entry:
            contexts.add(str(Path(entry["context"])))
    return contexts


def check(
    repo_root: Path,
    search_roots: list[str],
    workflow_path: Path,
) -> tuple[bool, list[dict]]:
    """Run the matrix check and return (all_matched, results).

    Each result dict has:
      - file: str path relative to repo root
      - status: "ok" | "unmatched" | "ignored"
      - suggestion: str (only present when status == "unmatched" and first file in dir)
    """
    container_files = discover_container_files(repo_root, search_roots)
    matrix_contexts = parse_matrix_contexts(workflow_path)
    ignore_list = load_ignore_list(repo_root)

    results = []
    all_matched = True
    seen_dirs: set[str] = set()

    for cf in container_files:
        rel_dir = str(cf.parent.relative_to(repo_root))

        if rel_dir in ignore_list:
            results.append({"file": str(cf.relative_to(repo_root)), "status": "ignored"})
            continue

        if rel_dir in matrix_contexts:
            results.append({"file": str(cf.relative_to(repo_root)), "status": "ok"})
        else:
            all_matched = False
            name = rel_dir.replace("/", "-").replace("\\", "-")
            suggestion = f"          - name: {name}\n            context: {rel_dir}"
            result: dict = {
                "file": str(cf.relative_to(repo_root)),
                "status": "unmatched",
            }
            if rel_dir not in seen_dirs:
                result["suggestion"] = suggestion
                seen_dirs.add(rel_dir)
            results.append(result)

    return all_matched, results


def _print_results(results: list[dict], all_matched: bool, workflow_path: str) -> None:
    """Print check results to stdout."""
    print(f"🔍 Checking container build matrix entries in {workflow_path}...")

    unmatched = [r for r in results if r["status"] == "unmatched"]
    ignored = [r for r in results if r["status"] == "ignored"]
    ok = [r for r in results if r["status"] == "ok"]

    if ignored:
        print()
        for r in ignored:
            print(f"  ⏭️ {r['file']} (ignored via {IGNORE_FILENAME})")

    if all_matched:
        print()
        print(f"✅ All Containerfiles/Dockerfiles have a matching matrix entry ({len(ok)} checked)")
        return

    print()
    for r in unmatched:
        print(f"  ❌ {r['file']} has no matching entry in the container-build matrix")

    print()
    print("❌ Some Containerfiles/Dockerfiles are not registered in the container-build matrix.")
    print(f"   Add the following entries to the strategy.matrix.include section of {workflow_path}:")
    print()
    for r in unmatched:
        if "suggestion" in r:
            print(r["suggestion"])
    print()
    print("   See docs/CONTRIBUTING.md#adding-a-custom-base-image for details.")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check that every Containerfile/Dockerfile has a matching container-build matrix entry"
    )
    parser.add_argument("--repo-root", default=None, help="Path to the repository root")
    parser.add_argument("--workflow", default=WORKFLOW_PATH, help="Path to container-build workflow")
    parser.add_argument("--search-roots", nargs="+", default=SEARCH_ROOTS, help="Directories to scan")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else get_repo_root()
    workflow_path = repo_root / args.workflow

    if not workflow_path.exists():
        print(f"Workflow file not found: {workflow_path}")
        return 2  # distinct exit code for missing workflow

    all_matched, results = check(repo_root, args.search_roots, workflow_path)
    _print_results(results, all_matched, args.workflow)
    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
