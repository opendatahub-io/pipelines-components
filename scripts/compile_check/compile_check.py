#!/usr/bin/env python3
"""Compile and dependency validation tool for Kubeflow Pipelines components.

This script discovers component and pipeline modules based on the presence of
`metadata.yaml` files, validates declared dependencies, and ensures each target
compiles successfully with the Kubeflow Pipelines SDK.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kfp.dsl import graph_component

from ..lib.discovery import get_repo_root
from ..lib.kfp_compilation import (
    compile_and_get_yaml,
    find_decorated_functions_runtime,
    load_module_from_path,
)
from ..lib.metadata_utils import (
    MetadataTarget,
    create_metadata_targets,
    discover_metadata_files,
    validate_dependencies,
)

REPO_ROOT = get_repo_root()


@dataclass
class ValidationResult:
    """Stores the outcome of validating a single metadata target."""

    target: MetadataTarget
    success: bool
    compiled_objects: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Record a validation error and mark the result unsuccessful."""
        logging.error(message)
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """Record a non-fatal validation warning."""
        logging.warning(message)
        self.warnings.append(message)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the compile check tool."""
    parser = argparse.ArgumentParser(description="Compile Kubeflow components and pipelines.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Restrict validation to metadata paths under this directory. May be supplied multiple times.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first validation failure.",
    )
    parser.add_argument(
        "--include-flagless",
        action="store_true",
        help="Include targets that do not set ci.compile_check explicitly.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    """Configure logging verbosity for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _module_name_from_path(module_path: Path) -> str:
    """Generate a unique module name for dynamic imports."""
    relative = module_path.relative_to(REPO_ROOT).with_suffix("")
    sanitized_parts = [part.replace("-", "_").replace(".", "_") for part in relative.parts]
    return "compile_check_" + "_".join(sanitized_parts)


def validate_target(target: MetadataTarget) -> ValidationResult:
    """Validate a single metadata target by compiling exposed objects."""
    result = ValidationResult(target=target, success=True)
    dep_errors, dep_warnings = validate_dependencies(target.metadata)
    for warning in dep_warnings:
        result.add_warning(warning)
    for error in dep_errors:
        result.add_error(error)
    if dep_errors:
        return result

    try:
        module_name = _module_name_from_path(target.module_path)
        module = load_module_from_path(str(target.module_path), module_name)
    except Exception:
        result.add_error(f"Failed to load module defined in {target.module_path}.\n{traceback.format_exc()}")
        return result

    objects = find_decorated_functions_runtime(module, target.target_kind)
    if target.target_kind == "pipeline":
        objects = [(name, obj) for name, obj in objects if isinstance(obj, graph_component.GraphComponent)]
    else:
        objects = [(name, obj) for name, obj in objects if not isinstance(obj, graph_component.GraphComponent)]

    if not objects:
        result.add_error(f"No @dsl.{target.target_kind} decorated functions discovered in module {target.module_path}.")
        return result

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for attr_name, obj in objects:
            try:
                output_path = temp_path / f"{module_name}_{attr_name}.yaml"
                compile_and_get_yaml(obj, str(output_path))
                result.compiled_objects.append(f"{attr_name} -> {output_path.name}")
                logging.debug(
                    "Compiled %s from %s to %s",
                    attr_name,
                    target.module_path,
                    output_path,
                )
            except Exception:
                result.add_error(
                    f"Failed to compile {target.target_kind} `{attr_name}` from {target.module_path}.\n"
                    f"{traceback.format_exc()}"
                )
                if result.errors:
                    # stop compiling additional objects from this module to avoid noise
                    break

    return result


def run_validation(args: argparse.Namespace) -> int:
    """Validate all discovered metadata targets."""
    configure_logging(args.verbose)

    discovered = discover_metadata_files(repo_root=REPO_ROOT)
    targets = create_metadata_targets(
        discovered,
        args.include_flagless,
        args.path,
        repo_root=REPO_ROOT,
        logger=logging.getLogger(__name__),
    )

    if not targets:
        logging.info("No targets discovered for compile check.")
        return 0

    results: list[ValidationResult] = []
    for target in targets:
        display_name = target.metadata.get("name")
        if not display_name:
            try:
                display_name = str(target.metadata_path.parent.relative_to(REPO_ROOT))
            except ValueError:
                display_name = str(target.metadata_path.parent)
        logging.info(
            "Validating %s (%s) from %s",
            display_name,
            target.target_kind,
            target.metadata_path,
        )
        result = validate_target(target)
        results.append(result)

        if result.success:
            logging.info(
                "✓ %s compiled successfully (%s)",
                display_name,
                ", ".join(result.compiled_objects) if result.compiled_objects else "no output",
            )
        else:
            logging.error(
                "✗ %s failed validation (%d error(s))",
                display_name,
                len(result.errors),
            )
            if args.fail_fast:
                break

    failed = [res for res in results if not res.success]
    logging.info(
        "Validation complete: %d succeeded, %d failed.",
        len(results) - len(failed),
        len(failed),
    )

    if failed:
        logging.error("Compile check failed for the targets listed above.")
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint for running compile checks via the CLI."""
    args = parse_args(argv)
    return run_validation(args)


if __name__ == "__main__":
    sys.exit(main())
