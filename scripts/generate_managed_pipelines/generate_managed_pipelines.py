"""Generate managed-pipelines.json and compile managed pipeline YAMLs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from kfp.dsl import graph_component

from ..lib.discovery import get_repo_root
from ..lib.kfp_compilation import (
    compile_and_get_yaml,
    find_decorated_functions_runtime,
    load_module_from_path,
)
from .pipeline_description import extract_pipeline_description_from_file

OUTPUT_FILENAME = "managed-pipelines.json"
METADATA_FILENAME = "metadata.yaml"
PIPELINE_PY = "pipeline.py"

# Must match values accepted in metadata.yaml (see scripts/validate_metadata).
METADATA_STABILITY_VALUES = frozenset({"experimental", "alpha", "beta", "stable"})

# Map metadata stability to consumer-facing labels in managed-pipelines.json.
STABILITY_TO_MANAGED_DISPLAY: dict[str, str] = {
    "alpha": "Development Preview",
    "beta": "Technology Preview",
    "stable": "General Availability",
}


class ManagedPipelineMetadataError(ValueError):
    """Invalid ``metadata.yaml`` for a pipeline marked ``managed: true``."""

    def __init__(self, message: str, *, pipeline_dir: Path | None = None) -> None:
        """Initialize the metadata validation error with optional pipeline location."""
        super().__init__(message)
        self.pipeline_dir = pipeline_dir


class ManagedPipelineCompilationError(RuntimeError):
    """Failure to compile a managed pipeline's ``pipeline.py``."""


def _pipeline_dir_label(dir_path: Path, repo_root: Path) -> str:
    """Relative path for error messages (POSIX)."""
    try:
        return dir_path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(dir_path)


@dataclass(frozen=True)
class ManagedPipelineEntry:
    """One record in managed-pipelines.json."""

    name: str
    description: str
    path: str
    stability: str


def _resolve_pipeline_description(metadata: dict, pipeline_py: Path) -> str:
    """Prefer non-empty ``description`` from metadata; else decorator/docstring from ``pipeline.py``."""
    yaml_description = metadata.get("description")
    if isinstance(yaml_description, str) and yaml_description.strip():
        return yaml_description.strip()
    from_decorator = extract_pipeline_description_from_file(
        pipeline_py,
        function_name=metadata.get("name") if isinstance(metadata.get("name"), str) else None,
    )
    return from_decorator or ""


def managed_pipeline_entry_from_dir(
    *,
    dir_path: Path,
    repo_root: Path,
    metadata: dict,
) -> ManagedPipelineEntry:
    """Build a :class:`ManagedPipelineEntry`; raises :class:`ManagedPipelineMetadataError` if metadata is invalid.

    ``description`` may be empty if neither ``metadata.yaml`` nor ``pipeline.py`` provides text.
    """
    label = _pipeline_dir_label(dir_path, repo_root)

    raw_name = metadata.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ManagedPipelineMetadataError(
            f"{label}: metadata 'name' must be a non-empty string",
            pipeline_dir=dir_path,
        )

    raw_stability = metadata.get("stability")
    if not isinstance(raw_stability, str) or not raw_stability.strip():
        raise ManagedPipelineMetadataError(
            f"{label}: metadata 'stability' must be a non-empty string",
            pipeline_dir=dir_path,
        )
    stability = raw_stability.strip()
    if stability not in METADATA_STABILITY_VALUES:
        raise ManagedPipelineMetadataError(
            f"{label}: metadata 'stability' must be one of {sorted(METADATA_STABILITY_VALUES)}, got {stability!r}",
            pipeline_dir=dir_path,
        )
    if stability == "experimental":
        raise ManagedPipelineMetadataError(
            f"{label}: managed pipelines cannot use 'experimental' stability; use alpha, beta, or stable",
            pipeline_dir=dir_path,
        )
    display_stability = STABILITY_TO_MANAGED_DISPLAY[stability]

    try:
        rel_path = dir_path.relative_to(repo_root)
    except ValueError as e:
        raise ManagedPipelineMetadataError(
            f"{label}: pipeline directory must be under repository root {repo_root}",
            pipeline_dir=dir_path,
        ) from e
    path_str = f"{rel_path.as_posix()}/{PIPELINE_PY}"

    pipeline_py = dir_path / PIPELINE_PY
    description = _resolve_pipeline_description(metadata, pipeline_py)

    return ManagedPipelineEntry(
        name=raw_name.strip(),
        description=description,
        path=path_str,
        stability=display_stability,
    )


def load_metadata(metadata_path: Path) -> dict | None:
    """Load and return metadata from a metadata.yaml file.

    Args:
        metadata_path: Path to metadata.yaml.

    Returns:
        Parsed metadata dict or None if file is missing or invalid.
    """
    if not metadata_path.is_file():
        return None
    with open(metadata_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_pipeline_dirs(pipelines_root: Path) -> list[Path]:
    """Discover all directories under pipelines/ that contain both metadata.yaml and pipeline.py.

    Args:
        pipelines_root: Path to the pipelines/ directory.

    Returns:
        List of paths to pipeline directories (each has metadata.yaml and pipeline.py).
    """
    result = []
    for meta_path in pipelines_root.rglob(METADATA_FILENAME):
        dir_path = meta_path.parent
        if (dir_path / PIPELINE_PY).is_file():
            result.append(dir_path)
    return sorted(result)


def collect_managed_pipelines(repo_root: Path) -> list[ManagedPipelineEntry]:
    """Collect all pipelines that have managed: true in their metadata.yaml.

    Args:
        repo_root: Repository root path.

    Returns:
        List of ``ManagedPipelineEntry`` records.

    Raises:
        FileNotFoundError: If ``<repo_root>/pipelines`` is missing or not a directory.
        ManagedPipelineMetadataError: If any ``managed: true`` pipeline has invalid metadata.
    """
    pipelines_root = repo_root / "pipelines"
    if not pipelines_root.is_dir():
        raise FileNotFoundError(
            f"pipelines directory not found or not a directory: {pipelines_root}",
        )

    result: list[ManagedPipelineEntry] = []
    for dir_path in discover_pipeline_dirs(pipelines_root):
        meta_path = dir_path / METADATA_FILENAME
        metadata = load_metadata(meta_path)
        if not metadata or not isinstance(metadata, Mapping):
            continue
        if metadata.get("managed") is not True:
            continue

        result.append(
            managed_pipeline_entry_from_dir(
                dir_path=dir_path,
                repo_root=repo_root,
                metadata=metadata,
            )
        )

    return result


def _module_name_for_compilation(pipeline_py: Path, repo_root: Path) -> str:
    """Derive a unique Python module name for dynamic import during compilation."""
    try:
        relative = pipeline_py.relative_to(repo_root).with_suffix("")
        parts = [p.replace("-", "_").replace(".", "_") for p in relative.parts]
        return "managed_compile_" + "_".join(parts)
    except ValueError:
        return "managed_compile_" + pipeline_py.stem


def compile_managed_pipeline(
    *,
    pipeline_py: Path,
    output_path: Path,
    repo_root: Path,
) -> None:
    """Compile a managed pipeline's ``pipeline.py`` to a KFP YAML spec.

    Args:
        pipeline_py: Path to the ``pipeline.py`` source file.
        output_path: Destination path for the compiled YAML.
        repo_root: Repository root (used for generating a unique module name).

    Raises:
        ManagedPipelineCompilationError: On import failure, missing decorator, or compilation error.
    """
    module_name = _module_name_for_compilation(pipeline_py, repo_root)

    try:
        module = load_module_from_path(str(pipeline_py), module_name)
    except Exception as e:
        raise ManagedPipelineCompilationError(f"Failed to load module {pipeline_py}: {e}") from e

    functions = find_decorated_functions_runtime(module, "pipeline")
    functions = [(name, obj) for name, obj in functions if isinstance(obj, graph_component.GraphComponent)]

    if not functions:
        raise ManagedPipelineCompilationError(f"No @dsl.pipeline decorated function found in {pipeline_py}")

    _, func = functions[0]
    try:
        compile_and_get_yaml(func, str(output_path))
    except Exception as e:
        raise ManagedPipelineCompilationError(f"Failed to compile pipeline from {pipeline_py}: {e}") from e


def main() -> int:
    """CLI entry point. Generate managed-pipelines.json and compile managed pipeline YAMLs."""
    parser = argparse.ArgumentParser(
        description="Generate managed-pipelines.json and compile managed pipeline YAMLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output file path (default: repo root / {OUTPUT_FILENAME})",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    output_path = args.output if args.output is not None else repo_root / OUTPUT_FILENAME

    pipelines_root = repo_root / "pipelines"
    if not pipelines_root.is_dir():
        print(
            f"Error: pipelines directory not found or not a directory: {pipelines_root}",
            file=sys.stderr,
        )
        return 1

    try:
        pipelines = collect_managed_pipelines(repo_root)
    except ManagedPipelineMetadataError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    payload = [asdict(entry) for entry in pipelines]
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(pipelines)} pipeline(s) to {output_path}", file=sys.stderr)

    for entry in pipelines:
        pipeline_py = repo_root / entry.path
        output_yaml = pipeline_py.parent / "pipeline.yaml"
        try:
            compile_managed_pipeline(
                pipeline_py=pipeline_py,
                output_path=output_yaml,
                repo_root=repo_root,
            )
        except ManagedPipelineCompilationError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Compiled {entry.name} -> {output_yaml}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
