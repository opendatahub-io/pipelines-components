"""Generate managed-pipelines.json from pipeline metadata with managed: true."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from scripts.lib.discovery import get_repo_root

from .pipeline_description import extract_pipeline_description_from_file

OUTPUT_FILENAME = "managed-pipelines.json"
METADATA_FILENAME = "metadata.yaml"
PIPELINE_PY = "pipeline.py"

# Must match values accepted in metadata.yaml (see scripts/validate_metadata).
STABILITY_VALUES = frozenset({"experimental", "alpha", "beta", "stable"})


class ManagedPipelineMetadataError(ValueError):
    """Invalid ``metadata.yaml`` for a pipeline marked ``managed: true``."""

    def __init__(self, message: str, *, pipeline_dir: Path | None = None) -> None:
        super().__init__(message)
        self.pipeline_dir = pipeline_dir


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

    @classmethod
    def from_managed_pipeline_dir(
        cls,
        *,
        dir_path: Path,
        repo_root: Path,
        metadata: dict,
    ) -> ManagedPipelineEntry:
        """Build an entry; raises :class:`ManagedPipelineMetadataError` if metadata is invalid.

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
        if stability not in STABILITY_VALUES:
            raise ManagedPipelineMetadataError(
                f"{label}: metadata 'stability' must be one of {sorted(STABILITY_VALUES)}, got {stability!r}",
                pipeline_dir=dir_path,
            )

        try:
            rel_path = dir_path.relative_to(repo_root)
        except ValueError as e:
            raise ManagedPipelineMetadataError(
                f"{label}: pipeline directory must be under repository root {repo_root}",
                pipeline_dir=dir_path,
            ) from e
        path_str = f"{rel_path.as_posix()}/{PIPELINE_PY}"
        if not path_str.strip():
            raise ManagedPipelineMetadataError(
                f"{label}: could not build pipeline path",
                pipeline_dir=dir_path,
            )

        pipeline_py = dir_path / PIPELINE_PY
        description = _resolve_pipeline_description(metadata, pipeline_py)

        return cls(
            name=raw_name.strip(),
            description=description,
            path=path_str,
            stability=stability,
        )


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


def load_metadata(metadata_path: Path) -> dict | None:
    """Load and return metadata from a metadata.yaml file.

    Args:
        metadata_path: Path to metadata.yaml.

    Returns:
        Parsed metadata dict or None if file is missing or invalid.
    """
    if not metadata_path.is_file():
        return None
    with open(metadata_path) as f:
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
        ManagedPipelineMetadataError: If any ``managed: true`` pipeline has invalid metadata.
    """
    pipelines_root = repo_root / "pipelines"
    if not pipelines_root.is_dir():
        return []

    result: list[ManagedPipelineEntry] = []
    for dir_path in discover_pipeline_dirs(pipelines_root):
        meta_path = dir_path / METADATA_FILENAME
        metadata = load_metadata(meta_path)
        if not metadata:
            continue
        if metadata.get("managed") is not True:
            continue

        result.append(
            ManagedPipelineEntry.from_managed_pipeline_dir(
                dir_path=dir_path,
                repo_root=repo_root,
                metadata=metadata,
            )
        )

    return result


def main() -> int:
    """CLI entry point. Generate managed-pipelines.json at repo root."""
    parser = argparse.ArgumentParser(
        description="Generate managed-pipelines.json from pipeline metadata (managed: true).",
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

    try:
        pipelines = collect_managed_pipelines(repo_root)
    except ManagedPipelineMetadataError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    payload = [asdict(entry) for entry in pipelines]
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(pipelines)} pipeline(s) to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
