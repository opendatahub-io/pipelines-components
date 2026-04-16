"""Helpers to assert stable root-level DAG task IDs on compiled pipelines."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

from kfp import compiler


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


def root_dag_task_ids(pipeline_spec: dict[str, Any]) -> tuple[str, ...]:
    """Return sorted task IDs from ``root.dag.tasks`` in a pipeline spec dict."""
    root = pipeline_spec.get("root")
    if not isinstance(root, dict):
        msg = "Pipeline spec has no 'root' mapping."
        raise ValueError(msg)
    dag = root.get("dag")
    if not isinstance(dag, dict):
        msg = "Pipeline spec root has no 'dag' mapping."
        raise ValueError(msg)
    tasks = dag.get("tasks")
    if not isinstance(tasks, dict):
        msg = "Pipeline spec root.dag has no 'tasks' mapping."
        raise ValueError(msg)
    ids = [k for k in tasks if isinstance(k, str)]
    if len(ids) != len(tasks):
        msg = "Pipeline root.dag.tasks contains non-string keys."
        raise ValueError(msg)
    return tuple(sorted(ids))


def assert_compiled_pipeline_root_dag_task_ids(
    *,
    pipeline_func: Any,
    expected_task_ids: Sequence[str],
) -> None:
    """Fail if a fresh compile's root DAG task IDs differ from ``expected_task_ids``.

    Task IDs are the YAML keys under ``root.dag.tasks`` (KFP task names). Renaming a
    step, adding, or removing a top-level task fails this check. Update the
    expected sequence intentionally when the pipeline graph changes, then refresh
    ``pipeline.yaml`` as usual.

    Comparison uses sorted IDs so YAML key order does not matter.

    Args:
        pipeline_func: The ``@dsl.pipeline`` function object.
        expected_task_ids: Exact set of root task IDs (order ignored).

    Raises:
        AssertionError: If the compiled DAG task set does not match.
        ValueError: If the compiled YAML has no parseable root DAG tasks.
    """
    want = tuple(sorted(expected_task_ids))
    if len(want) != len(set(want)):
        msg = "expected_task_ids contains duplicate task IDs"
        raise ValueError(msg)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=tmp_path)
        spec = load_pipeline_spec_document(Path(tmp_path))
        got = root_dag_task_ids(spec)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if got != want:
        only_got = sorted(set(got) - set(want))
        only_want = sorted(set(want) - set(got))
        msg = (
            "Compiled pipeline root DAG task IDs do not match expected set.\n"
            f"  expected (sorted): {list(want)}\n"
            f"  actual (sorted):   {list(got)}\n"
        )
        if only_want:
            msg += f"  missing vs compile: {only_want}\n"
        if only_got:
            msg += f"  extra vs expected:    {only_got}\n"
        msg += "If this change is intentional, update expected_task_ids in the unit test and re-commit pipeline.yaml."
        raise AssertionError(msg)
