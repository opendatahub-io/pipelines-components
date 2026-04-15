"""Compare checked-in compiled pipeline YAML with a fresh compile from source.

Compares ``components``, ``root``, ``pipelineInfo``, and ``deploymentSpec`` after
normalization:

* **Container images** in ``deploymentSpec`` are replaced with a placeholder so
  registry/tag differences (e.g. env overrides) do not fail the check.
* **KFP embedded notebook archives** (``__KFP_EMBEDDED_ARCHIVE_B64 = '...'``)
  are replaced with a placeholder; the base64 payload changes whenever templates
  are re-embedded at compile time.

Re-compile and commit ``pipeline.yaml`` when pipeline or component **source**
under those sections changes.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from kfp import compiler

# KFP inlines a gzip tarball of notebook assets as a long base64 assignment.
_RE_KFP_EMBEDDED_ARCHIVE_B64_SINGLE = re.compile(
    r"__KFP_EMBEDDED_ARCHIVE_B64\s*=\s*'[^']*'",
    re.DOTALL,
)
_RE_KFP_EMBEDDED_ARCHIVE_B64_DOUBLE = re.compile(
    r'__KFP_EMBEDDED_ARCHIVE_B64\s*=\s*"[^"]*"',
    re.DOTALL,
)

_COMPARISON_KEYS = ("components", "root", "pipelineInfo", "deploymentSpec")


def _is_platform_spec(doc: dict[str, Any]) -> bool:
    return isinstance(doc.get("platforms"), dict)


def _is_pipeline_spec(doc: dict[str, Any]) -> bool:
    return "deploymentSpec" in doc or "root" in doc


def load_pipeline_spec_document(path: str | Path) -> dict[str, Any]:
    """Load the pipeline spec dict from a compiled YAML file (single or multi-doc)."""
    import yaml

    path = Path(path)
    with path.open() as f:
        docs = [d for d in yaml.safe_load_all(f) if isinstance(d, dict)]
    if not docs:
        msg = f"Compiled YAML at {path} has no dict documents."
        raise ValueError(msg)
    if len(docs) == 1:
        return docs[0]
    pipeline_spec = next(
        (d for d in docs if _is_pipeline_spec(d) and not _is_platform_spec(d)),
        None,
    )
    if pipeline_spec is None:
        msg = f"Could not find pipeline spec document in {path}."
        raise ValueError(msg)
    return pipeline_spec


def _redact_kfp_embedded_archive_assignments(s: str) -> str:
    """Remove volatile KFP-inlined notebook/tar base64 blobs from launcher strings."""
    s = _RE_KFP_EMBEDDED_ARCHIVE_B64_SINGLE.sub(
        "__KFP_EMBEDDED_ARCHIVE_B64 = '__REDACTED__'",
        s,
    )
    return _RE_KFP_EMBEDDED_ARCHIVE_B64_DOUBLE.sub(
        '__KFP_EMBEDDED_ARCHIVE_B64 = "__REDACTED__"',
        s,
    )


def _normalize_command_source_lines(s: str) -> str:
    """Drop trailing whitespace per line to avoid KFP/YAML line-wrap differences."""
    return "\n".join(line.rstrip() for line in s.splitlines())


def sanitize_for_pipeline_comparison(obj: Any) -> Any:
    """Redact volatile fields and normalize strings for deterministic comparison."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "image" and isinstance(value, str):
                out[key] = "__REDACTED_IMAGE__"
            else:
                out[key] = sanitize_for_pipeline_comparison(value)
        return out
    if isinstance(obj, list):
        return [sanitize_for_pipeline_comparison(item) for item in obj]
    if isinstance(obj, str):
        text = _redact_kfp_embedded_archive_assignments(obj)
        if "def " in text and "import kfp" in text:
            text = _normalize_command_source_lines(text)
        return text
    return obj


def extract_pipeline_comparison_slice(pipeline_spec: dict[str, Any]) -> dict[str, Any]:
    """Subset of pipeline spec compared against a fresh compile."""
    return {k: pipeline_spec[k] for k in _COMPARISON_KEYS if k in pipeline_spec}


def normalize_sorted(obj: Any) -> Any:
    """Recursively sort dict keys for deterministic comparison."""
    if isinstance(obj, dict):
        return {k: normalize_sorted(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [normalize_sorted(x) for x in obj]
    return obj


def _describe_mismatch(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} vs {type(actual).__name__}"
    if isinstance(expected, dict):
        ek, ak = set(expected), set(actual)
        if ek != ak:
            parts: list[str] = []
            if ek - ak:
                parts.append(f"only in checked-in YAML: {sorted(ek - ak)[:30]}")
            if ak - ek:
                parts.append(f"only in fresh compile: {sorted(ak - ek)[:30]}")
            return f"{path} keys differ: " + "; ".join(parts)
        for key in sorted(ek):
            msg = _describe_mismatch(expected[key], actual[key], f"{path}.{key}")
            if msg is not None:
                return msg
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: list len {len(expected)} vs {len(actual)}"
        for i, (e_item, a_item) in enumerate(zip(expected, actual, strict=True)):
            msg = _describe_mismatch(e_item, a_item, f"{path}[{i}]")
            if msg is not None:
                return msg
        return None
    if expected != actual:
        ev, av = repr(expected), repr(actual)
        if len(ev) > 200:
            ev = ev[:200] + "..."
        if len(av) > 200:
            av = av[:200] + "..."
        return f"{path}: {ev} != {av}"
    return None


def assert_checked_in_pipeline_yaml_matches_compiled_ir(
    *,
    pipeline_func: Any,
    checked_in_yaml_path: str | Path,
) -> None:
    """Assert that ``pipeline.yaml`` matches a fresh compile after sanitization.

    Args:
        pipeline_func: The ``@dsl.pipeline`` function object.
        checked_in_yaml_path: Path to the committed ``pipeline.yaml``.

    Raises:
        AssertionError: If compared sections differ.
        FileNotFoundError: If ``checked_in_yaml_path`` is missing.
        ValueError: If the YAML cannot be parsed or has no pipeline spec.
    """
    checked_in_yaml_path = Path(checked_in_yaml_path)
    if not checked_in_yaml_path.is_file():
        msg = f"Checked-in pipeline YAML not found: {checked_in_yaml_path}"
        raise FileNotFoundError(msg)

    committed = normalize_sorted(
        sanitize_for_pipeline_comparison(
            extract_pipeline_comparison_slice(load_pipeline_spec_document(checked_in_yaml_path)),
        ),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=tmp_path)
        fresh = normalize_sorted(
            sanitize_for_pipeline_comparison(
                extract_pipeline_comparison_slice(load_pipeline_spec_document(tmp_path)),
            ),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if committed != fresh:
        detail = _describe_mismatch(committed, fresh) or "unknown mismatch"
        msg = (
            "Checked-in pipeline.yaml does not match a fresh compile from pipeline.py "
            "(components, root, pipelineInfo, deploymentSpec with embedded archive + "
            f"images redacted). First difference: {detail}. "
            "Re-compile and commit pipeline.yaml, or fix pipeline/component source."
        )
        raise AssertionError(msg)
