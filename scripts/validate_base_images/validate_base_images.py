#!/usr/bin/env python3
"""Validate base images used in Kubeflow Pipelines components and pipelines.

This script discovers all components and pipelines in the components/ and pipelines/
directories, compiles them using kfp.compiler to generate IR YAML, and extracts
base_image values from the pipeline specifications.

Usage:
    uv run python -m scripts.validate_base_images.validate_base_images
"""

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..lib.base_image import (
    BaseImageAllowlist,
    get_base_images_from_compile_result,
    load_base_image_allowlist,
)
from ..lib.base_image import is_valid_base_image as _is_valid_base_image
from ..lib.base_image import validate_base_images as _validate_base_images
from ..lib.discovery import (
    build_component_asset,
    build_pipeline_asset,
    discover_assets,
    get_repo_root,
    resolve_component_path,
    resolve_pipeline_path,
)
from ..lib.kfp_compilation import compile_and_get_yaml, find_decorated_functions_runtime, load_module_from_path


@dataclass
class ValidationConfig:
    """Configuration for base image validation."""

    allowlist_path: Path = Path(__file__).parent / "base_image_allowlist.yaml"
    allowlist: BaseImageAllowlist | None = None


_config: ValidationConfig | None = None


def get_config() -> ValidationConfig:
    """Get the current validation configuration."""
    global _config
    if _config is None:
        config = ValidationConfig()
        config.allowlist = load_base_image_allowlist(config.allowlist_path)
        _config = config
    return _config


def set_config(config: ValidationConfig) -> None:
    """Set the validation configuration."""
    global _config
    _config = config


def is_valid_base_image(image: str, config: ValidationConfig | None = None) -> bool:
    """Check if a base image is valid according to configuration.

    Valid base images either:
    - Are empty/unset (represented as empty string or None)
    - Match the configured allowlist file

    Args:
        image: The base image string to validate
        config: Optional ValidationConfig; uses global config if not provided

    Returns:
        True if the image is valid, False otherwise
    """
    if config is None:
        config = get_config()

    if config.allowlist is None:
        config.allowlist = load_base_image_allowlist(config.allowlist_path)

    return _is_valid_base_image(image, config.allowlist)


def validate_base_images(images: set[str], config: ValidationConfig | None = None) -> set[str]:
    """Validate a set of base images and return invalid ones.

    Args:
        images: Set of base image strings to validate
        config: Optional ValidationConfig; uses global config if not provided

    Returns:
        Set of invalid base image strings
    """
    if config is None:
        config = get_config()

    if config.allowlist is None:
        config.allowlist = load_base_image_allowlist(config.allowlist_path)

    return _validate_base_images(images, config.allowlist)


def _create_result(asset: dict[str, Any], asset_type: str) -> dict[str, Any]:
    """Create an initial result dict for an asset."""
    return {
        "category": asset["category"],
        "name": asset["name"],
        "type": asset_type,
        "path": str(asset["path"]),
        "base_images": set(),
        "invalid_base_images": set(),
        "errors": [],
        "compiled": False,
    }


def process_asset(
    asset: dict[str, Any],
    asset_type: str,
    temp_dir: str,
    config: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Process a single component or pipeline asset.

    Returns a dict with asset info and extracted base images.
    """
    if config is None:
        config = get_config()

    result = _create_result(asset, asset_type)
    module_name = f"{asset['category']}_{asset['name']}_{asset_type}"

    try:
        module = load_module_from_path(asset["module_path"], module_name)
    except Exception as e:
        result["errors"].append(f"Failed to load module: {e}")
        return result

    functions = find_decorated_functions_runtime(module, asset_type)
    if not functions:
        result["errors"].append(f"No @dsl.{asset_type} decorated functions found")
        return result

    compiled_count = 0
    failed_count = 0
    for func_name, func in functions:
        output_path = os.path.join(temp_dir, f"{module_name}_{func_name}.yaml")
        try:
            ir_yaml = compile_and_get_yaml(func, output_path)
            result["compiled"] = True
            compiled_count += 1
            result["base_images"].update(get_base_images_from_compile_result(ir_yaml))
        except Exception as e:
            failed_count += 1
            print(f"  Warning: Failed to compile {func}: {e}")

    if not result["compiled"]:
        result["errors"].append(f"All {len(functions)} function(s) failed to compile")
    elif failed_count:
        result["errors"].append(
            f"{failed_count}/{len(functions)} function(s) failed to compile ({compiled_count} succeeded)"
        )

    result["invalid_base_images"] = validate_base_images(result["base_images"], config)
    result["base_images"] = sorted(result["base_images"])

    return result


def _print_result(result: dict[str, Any]) -> None:
    """Print the processing result for a single asset."""
    if result["errors"]:
        for error in result["errors"]:
            print(f"    Error: {error}")
    elif result["base_images"]:
        for image in result["base_images"]:
            is_invalid = image in result["invalid_base_images"]
            status = " [INVALID]" if is_invalid else ""
            print(f"    Base image: {image}{status}")
    elif result["compiled"]:
        print("    No custom base image (using default)")


def _process_assets(
    assets: list[dict[str, Any]],
    asset_type: str,
    label: str,
    temp_dir: str,
    config: ValidationConfig | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Process a batch of assets and return results and base images."""
    results: list[dict[str, Any]] = []
    base_images: set[str] = set()

    if not assets:
        return results, base_images

    print("-" * 70)
    print(f"Processing {label}")
    print("-" * 70)

    for asset in assets:
        print(f"  Processing: {asset['category']}/{asset['name']}")
        result = process_asset(asset, asset_type, temp_dir, config)
        results.append(result)
        base_images.update(result["base_images"])
        _print_result(result)

    print()
    return results, base_images


def _collect_violations(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect all base image violations from results."""
    violations = []
    for result in all_results:
        if result["invalid_base_images"]:
            for image in result["invalid_base_images"]:
                violations.append(
                    {
                        "path": result["path"],
                        "category": result["category"],
                        "name": result["name"],
                        "type": result["type"],
                        "image": image,
                    }
                )
    return violations


def _print_violations(violations: list[dict[str, Any]], config: ValidationConfig) -> None:
    """Print base image violations."""
    print("=" * 70)
    print("BASE IMAGE VIOLATIONS")
    print("=" * 70)
    print()
    print(f"Found {len(violations)} violation(s).")
    print()

    print(f"Invalid base images ({len(violations)}):")
    print("  Base images must be unset or match the allowlist.")
    print(f"  Allowlist: {config.allowlist_path}")
    print()
    print("  To fix this issue, either:")
    print("    1. Use an approved base image (e.g., 'ghcr.io/kubeflow/pipelines-components-<name>:<tag>')")
    print("    2. Leave base_image unset to use the KFP SDK default image")
    print(f"    3. Add an allowlist entry in {config.allowlist_path}")
    print()

    for violation in violations:
        print(f"  {violation['type'].title()}: {violation['category']}/{violation['name']}")
        print(f"    Path: {violation['path']}")
        print(f"    Invalid image: {violation['image']}")
        print()


def _compute_summary_counts(all_results: list[dict[str, Any]]) -> tuple[int, int, int, int, int]:
    total_assets = len(all_results)
    compiled_assets = sum(1 for r in all_results if r["compiled"])
    failed_assets = sum(1 for r in all_results if r["errors"])
    assets_with_images = sum(1 for r in all_results if r["base_images"])
    assets_with_invalid_images = sum(1 for r in all_results if r["invalid_base_images"])
    return (
        total_assets,
        compiled_assets,
        failed_assets,
        assets_with_images,
        assets_with_invalid_images,
    )


def _print_base_images_section(
    total_assets: int, failed_assets: int, all_base_images: set[str], violations: list[dict[str, Any]]
) -> None:
    if all_base_images:
        all_invalid = {v["image"] for v in violations}
        print("All unique base images found:")
        for image in sorted(all_base_images):
            status = " [INVALID]" if image in all_invalid else " [VALID]"
            print(f"  - {image}{status}")
        return

    if total_assets == 0:
        return

    if failed_assets > 0:
        print("No base images could be extracted (some assets failed to compile/load)")
        return

    print("No custom base images found (all using defaults)")


def _print_final_status(
    total_assets: int, failed_assets: int, violations: list[dict[str, Any]], config: ValidationConfig
) -> int:
    if total_assets == 0:
        print("No components or pipelines were discovered.")
        print("Components should be at: components/<category>/<name>/component.py")
        print("Pipelines should be at: pipelines/<category>/<name>/pipeline.py")
        return 0

    if violations:
        print(f"FAILED: {len(violations)} violation(s) found.")
        print(f"  - {len(violations)} invalid base image(s): must match the allowlist")
        print("    (e.g., 'ghcr.io/kubeflow/pipelines-components-<name>:<tag>'), leave unset, or match the allowlist.")
        print(f"    Allowlist: {config.allowlist_path}")
        return 1

    if failed_assets > 0:
        print(f"FAILED: {failed_assets} asset(s) could not be processed. See errors above.")
        return 1

    print("SUCCESS: All base images are valid.")
    return 0


def _print_summary(
    all_results: list[dict[str, Any]],
    all_base_images: set[str],
    config: ValidationConfig,
) -> int:
    """Print summary and return exit code."""
    violations = _collect_violations(all_results)

    if violations:
        _print_violations(violations, config)

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    (
        total_assets,
        compiled_assets,
        failed_assets,
        assets_with_images,
        assets_with_invalid_images,
    ) = _compute_summary_counts(all_results)

    print(f"Total assets discovered: {total_assets}")
    print(f"Successfully compiled: {compiled_assets}")
    print(f"Failed to process: {failed_assets}")
    print(f"Assets with custom base images: {assets_with_images}")
    print(f"Assets with invalid base images: {assets_with_invalid_images}")
    print()

    _print_base_images_section(total_assets, failed_assets, all_base_images, violations)

    print()
    return _print_final_status(total_assets, failed_assets, violations, config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate base images used by Kubeflow Pipelines components and pipelines.\n\n"
            "The validator compiles components/pipelines with the KFP compiler and extracts\n"
            "runtime images from the generated IR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Valid base images are:
  - images matching scripts/validate_base_images/base_image_allowlist.yaml

Examples:
  # Run with default settings
  %(prog)s

  # Validate specific assets only
  %(prog)s --component components/training/sample_model_trainer
  %(prog)s --pipeline pipelines/training/simple_training
  %(prog)s --component components/training/sample_model_trainer --pipeline pipelines/training/simple_training
        """,
    )

    parser.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Validate a specific component. Accepts either a directory like "
            "'components/<category>/<name>' or a direct '.../component.py' path. Repeatable."
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Validate a specific pipeline. Accepts either a directory like "
            "'pipelines/<category>/<name>' or a direct '.../pipeline.py' path. Repeatable."
        ),
    )
    parser.add_argument(
        "--allow-list",
        default=None,
        metavar="PATH",
        help=(
            "Path to a base-image allowlist YAML file. Defaults to "
            "'scripts/validate_base_images/base_image_allowlist.yaml'."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for base image validation.

    Args:
        argv: Command line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success, 1 for validation failures).
    """
    args = parse_args(argv)
    config = ValidationConfig()
    if args.allow_list:
        config.allowlist_path = Path(args.allow_list)
    config.allowlist = load_base_image_allowlist(config.allowlist_path)
    set_config(config)

    repo_root = get_repo_root()

    print("=" * 70)
    print("Kubeflow Pipelines Base Image Validator")
    print("=" * 70)
    print()
    print(f"Allowlist: {config.allowlist_path}")
    print()

    selected_components = bool(args.component)
    selected_pipelines = bool(args.pipeline)
    is_targeted = selected_components or selected_pipelines

    if is_targeted:
        components: list[dict[str, Any]] = []
        pipelines: list[dict[str, Any]] = []

        for raw in args.component:
            component_file = resolve_component_path(repo_root, raw)
            components.append(build_component_asset(repo_root, component_file))

        for raw in args.pipeline:
            pipeline_file = resolve_pipeline_path(repo_root, raw)
            pipelines.append(build_pipeline_asset(repo_root, pipeline_file))

        print(f"Selected {len(components)} component(s)")
        print(f"Selected {len(pipelines)} pipeline(s)")
    else:
        components = discover_assets(repo_root / "components", "component")
        print(f"Discovered {len(components)} component(s)")

        pipelines = discover_assets(repo_root / "pipelines", "pipeline")
        print(f"Discovered {len(pipelines)} pipeline(s)")
    print()

    all_results: list[dict[str, Any]] = []
    all_base_images: set[str] = set()

    with tempfile.TemporaryDirectory() as temp_dir:
        results, images = _process_assets(components, "component", "Components", temp_dir, config)
        all_results.extend(results)
        all_base_images.update(images)

        results, images = _process_assets(pipelines, "pipeline", "Pipelines", temp_dir, config)
        all_results.extend(results)
        all_base_images.update(images)

    return _print_summary(all_results, all_base_images, config)


if __name__ == "__main__":
    sys.exit(main())
