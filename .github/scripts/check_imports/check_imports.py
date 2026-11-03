#!/usr/bin/env python3
"""Validate module-level imports are limited to Python's standard library.

This script enforces the repository’s import guard strategy: third-party or
heavy dependencies must be imported within the function or pipeline body rather
than at module import time.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "import_exceptions.yaml"


class ImportGuardConfig:
    """Holds allowed imports and excluded paths for the import guard."""

    def __init__(
        self,
        module_allowlist: Optional[Iterable[str]] = None,
        path_scoped_allowlist: Optional[dict[str, Iterable[str]]] = None,
        excluded_paths: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize allowed imports and excluded repository-relative paths."""
        self.module_allowlist: set[str] = {canonicalize_module_name(item) for item in module_allowlist or []}
        self.excluded_paths = tuple(excluded_paths or [])
        # Store both exact path matches and pattern matches
        self.path_scoped_allowlist: dict[Path, set[str]] = {}
        self.pattern_scoped_allowlist: dict[str, set[str]] = {}

        for raw_path, modules in (path_scoped_allowlist or {}).items():
            canonical_modules = {canonicalize_module_name(mod) for mod in modules}

            # If path contains glob patterns (* or ?), store as pattern
            if "*" in raw_path or "?" in raw_path:
                self.pattern_scoped_allowlist[raw_path] = canonical_modules
            else:
                # Store exact path for backward compatibility
                normalized = Path(raw_path).resolve()
                self.path_scoped_allowlist[normalized] = canonical_modules

    @classmethod
    def from_path(cls, path: Path) -> "ImportGuardConfig":
        """Instantiate configuration from a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        modules = data.get("modules", [])
        path_scoped = data.get("files", {})
        return cls(modules, path_scoped, data.get("exclude", []))

    def is_excluded(self, file_path: Path) -> bool:
        """Return whether a repository-relative path matches an excluded glob."""
        try:
            relative_path = file_path.resolve().relative_to(Path.cwd()).as_posix()
        except ValueError:
            return False
        return any(fnmatch.fnmatch(relative_path, pattern) for pattern in self.excluded_paths)

    def is_allowed(self, module: str, file_path: Path) -> bool:
        """Return True when a module is allow-listed for the given file path."""
        canonical_module = canonicalize_module_name(module)
        if canonical_module in self.module_allowlist:
            return True

        resolved = file_path.resolve()

        # Check exact path matches (backward compatibility)
        modules = self.path_scoped_allowlist.get(resolved)
        if modules and canonical_module in modules:
            return True

        for parent in resolved.parents:
            modules = self.path_scoped_allowlist.get(parent)
            if modules and canonical_module in modules:
                return True

        # Check pattern matches
        # Convert resolved path to relative path from current working directory for pattern matching
        try:
            rel_path = str(resolved.relative_to(Path.cwd()))
        except ValueError:
            # If file is not relative to cwd, use full path
            rel_path = str(resolved)

        for pattern, modules in self.pattern_scoped_allowlist.items():
            if canonical_module in modules:
                # Check if the file path matches the pattern
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
                # Also check parent directories in the path
                path_parts = rel_path.split("/")
                for i in range(len(path_parts)):
                    partial_path = "/".join(path_parts[: i + 1])
                    if fnmatch.fnmatch(partial_path, pattern):
                        return True

        return False


def canonicalize_module_name(name: str) -> str:
    """Return the top-level portion of a dotted module path."""
    return name.split(".")[0]


def discover_python_files(paths: Sequence[str]) -> list[Path]:
    """Collect Python files from individual files or by walking directories."""
    python_files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)
        # Skip hidden files/directories in any part of the path
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.is_file() and path.suffix == ".py":
            python_files.append(path)
        elif path.is_dir():
            for candidate in path.rglob("*.py"):
                if any(part.startswith(".") for part in candidate.parts):
                    continue
                python_files.append(candidate)

    return python_files


@lru_cache(maxsize=1)
def build_stdlib_index() -> frozenset[str]:
    """Return a set containing names of standard-library modules."""
    candidates: set[str] = set(sys.builtin_module_names)
    candidates.update(canonicalize_module_name(name) for name in sys.stdlib_module_names)
    return frozenset(candidates)


class TopLevelImportVisitor(ast.NodeVisitor):
    """Collect absolute imports that appear at module scope."""

    def __init__(self) -> None:
        """Initialize storage for discovered imports."""
        self.imports: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Don't descend into functions."""
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Don't descend into async functions."""
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Don't descend into classes."""
        return

    def visit_Import(self, node: ast.Import) -> None:
        """Record absolute `import foo` statements."""
        for alias in node.names:
            if alias.name:
                self.imports.append((canonicalize_module_name(alias.name), node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record absolute `from foo import bar` statements."""
        if node.level == 0 and node.module:
            self.imports.append((canonicalize_module_name(node.module), node.lineno))

    def generic_visit(self, node: ast.AST) -> None:
        """Continue walking child nodes unless blocked by other handlers."""
        for child in ast.iter_child_nodes(node):
            self.visit(child)


def extract_top_level_imports(node: ast.AST) -> Iterable[tuple[str, int]]:
    """Yield (module, line) tuples for top-level import statements."""
    visitor = TopLevelImportVisitor()
    visitor.visit(node)
    return visitor.imports


def check_imports(files: Sequence[Path], config: ImportGuardConfig, *, quiet: bool = False) -> int:
    """Validate import style across a collection of Python files."""
    stdlib_modules = build_stdlib_index()
    violations: list[str] = []

    for file_path in files:
        resolved_path = file_path.resolve()
        if config.is_excluded(resolved_path):
            continue
        try:
            with resolved_path.open("r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=str(resolved_path))
        except SyntaxError as exc:
            violations.append(f"{resolved_path}: failed to parse ({exc})")
            continue

        for module_name, lineno in extract_top_level_imports(tree):
            if module_name in stdlib_modules:
                continue
            if config.is_allowed(module_name, resolved_path):
                continue
            violations.append(
                f"{resolved_path}:{lineno} imports non-stdlib module '{module_name}' at top level",
            )

    if violations:
        for entry in violations:
            print(entry, file=sys.stderr)
        print("You may add exceptions in the configuration if any of these modules are required")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ensure top-level Python imports are limited to the standard library.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file with allowed modules/files.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress warning output; only emit errors.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the import guard script."""
    args = parse_args()
    python_files = discover_python_files(args.paths)
    if not python_files:
        print("No Python files found to inspect.", file=sys.stderr)
        return 0
    config = ImportGuardConfig.from_path(Path(args.config))
    return check_imports(python_files, config, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
