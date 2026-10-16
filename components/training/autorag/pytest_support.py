"""Test-only helpers for AutoRAG training component unit tests."""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def autorag_shared_dir() -> Path:
    """Path to ``components/training/autorag/shared``."""
    return Path(__file__).resolve().parent / "shared"


def autorag_runtime_embed_dir() -> Path:
    """Path to the KFP-embedable helper directory under ``shared/runtime_embed``."""
    return autorag_shared_dir() / "runtime_embed"


def wrap_component_python_func(
    component,
    monkeypatch,
    tmp_path: Path,
    *,
    embedded_path: str | None = None,
) -> None:
    """Inject embedded-artifact and component-status mocks omitted by unit tests."""
    original = component.python_func
    signature = inspect.signature(original)
    # Default to runtime_embed so file-based loads get the real modules, not the
    # package re-export stubs at shared/{component_status,mlflow_tracking}.py.
    embed_root = embedded_path or str(autorag_runtime_embed_dir())

    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        if "embedded_artifact" in signature.parameters and "embedded_artifact" not in bound.arguments:
            embedded = mock.MagicMock()
            embedded.path = embed_root
            kwargs["embedded_artifact"] = embedded
        if "component_status" in signature.parameters and "component_status" not in bound.arguments:
            status = mock.MagicMock()
            status.path = str(tmp_path / "component_status_out")
            status.metadata = {}
            kwargs["component_status"] = status
        if "leaderboard" in signature.parameters and "leaderboard" not in bound.arguments:
            html = mock.MagicMock()
            html.path = str(tmp_path / "leaderboard.html")
            kwargs["leaderboard"] = html
        if "starter_kit" in signature.parameters and "starter_kit" not in bound.arguments:
            starter_kit = mock.MagicMock()
            starter_kit.path = str(tmp_path / "starter_kit-output")
            starter_kit.uri = "gs://bucket/starter_kit"
            starter_kit.metadata = {}
            kwargs["starter_kit"] = starter_kit
        return original(*args, **kwargs)

    wrapper = functools.wraps(original)(wrapper)
    monkeypatch.setattr(component, "python_func", wrapper)


class FakeRunInfo:
    """Stand-in for ``mlflow.entities.RunInfo``."""

    def __init__(self, run_id: str, experiment_id: str) -> None:
        """Record the identifiers the tracking module reads off an MLflow run."""
        self.run_id = run_id
        self.experiment_id = experiment_id


class FakeRun:
    """Stand-in for ``mlflow.entities.Run``."""

    def __init__(self, run_id: str, experiment_id: str) -> None:
        """Wrap the run identifiers in the ``.info`` attribute MLflow exposes."""
        self.info = FakeRunInfo(run_id, experiment_id)


class FakeMlflow:
    """In-memory MLflow double recording everything written to the active run.

    Implements only the surface ``shared/mlflow_tracking.py`` touches. Runs are kept in
    ``runs`` keyed by run id, each holding the params, metrics, tags, and artifact names
    written while it was the innermost active run.
    """

    def __init__(self, *, experiment_id: str = "7", nested_supported: bool = True) -> None:
        """Create an empty tracking server; ``nested_supported`` toggles nested-run support."""
        self.experiment_id = experiment_id
        self.nested_supported = nested_supported
        self.tracking_uri = ""
        self.set_experiment_calls: list[str] = []
        self.started: list[dict] = []
        self.runs: dict[str, dict] = {}
        self.terminated: list[str] = []
        self._stack: list[str] = []
        self._counter = 0

    # -- API surface used by the tracking module -------------------------

    def set_tracking_uri(self, uri: str) -> None:
        """Record the tracking URI the client was pointed at."""
        self.tracking_uri = uri

    def set_experiment(self, experiment_name: str | None = None, experiment_id: str | None = None) -> None:
        """Record a get-or-create / activate-by-id experiment call."""
        self.set_experiment_calls.append(experiment_id if experiment_id is not None else experiment_name)

    @contextmanager
    def _run_scope(self, run_id: str):
        """Make ``run_id`` the active run for the duration of the block."""
        self._stack.append(run_id)
        try:
            yield FakeRun(run_id, self.runs[run_id]["experiment_id"])
        except Exception:
            self.runs[run_id]["status"] = "FAILED"
            raise
        else:
            self.runs[run_id]["status"] = "FINISHED"
        finally:
            self._stack.pop()

    def start_run(self, run_id: str = "", run_name: str = "", nested: bool = False, experiment_id: str = ""):
        """Open (or resume) a run and return it as a context manager."""
        if nested and not self.nested_supported:
            raise RuntimeError("nested runs unsupported")
        self.started.append({"run_id": run_id, "run_name": run_name, "nested": nested})
        if not run_id:
            self._counter += 1
            run_id = f"run-{self._counter}"
        self.runs.setdefault(
            run_id,
            {
                "experiment_id": experiment_id or self.experiment_id,
                "name": run_name,
                "params": {},
                "metrics": {},
                "tags": {},
                "artifacts": [],
            },
        )
        return self._run_scope(run_id)

    @property
    def _active(self) -> dict:
        """The innermost active run's record."""
        return self.runs[self._stack[-1]]

    def log_params(self, params: dict, run_id: str = "") -> None:
        """Write a batch of params to the active run or an explicitly selected run."""
        target = self.runs[run_id] if run_id else self._active
        target["params"].update(params)

    def log_param(self, key: str, value) -> None:
        """Write a single param to the active run."""
        self._active["params"][key] = value

    def log_metrics(self, metrics: dict) -> None:
        """Write a batch of metrics to the active run."""
        self._active["metrics"].update(metrics)

    def set_tags(self, tags: dict) -> None:
        """Write a batch of tags to the active run."""
        self._active["tags"].update(tags)

    def set_tag(self, key: str, value) -> None:
        """Write a single tag to the active run."""
        self._active["tags"][key] = value

    def log_artifact(self, local_path: str, artifact_path: str = "") -> None:
        """Record an artifact upload as ``(artifact_path, filename)``."""
        self._active["artifacts"].append((artifact_path, Path(local_path).name))

    def MlflowClient(self):  # noqa: N802 - mirrors the MLflow API name
        """Return self; the double implements the client API inline."""
        return self

    def get_run(self, run_id: str):
        """Return the stored run tags through MLflow's ``Run.data.tags`` shape."""
        return SimpleNamespace(data=SimpleNamespace(tags=self.runs[run_id]["tags"]))

    def create_run(self, experiment_id: str, run_name: str, tags: dict):
        """Create a run without activating it (the ``MlflowClient`` fallback path)."""
        self._counter += 1
        run_id = f"created-{self._counter}"
        self.runs[run_id] = {
            "experiment_id": experiment_id,
            "name": run_name,
            "params": {},
            "metrics": {},
            "tags": dict(tags),
            "artifacts": [],
        }
        return FakeRun(run_id, experiment_id)

    def set_terminated(self, run_id: str, status: str = "") -> None:
        """Record that a run was force-terminated."""
        self.terminated.append(run_id)

    # -- assertion helpers -----------------------------------------------

    def child_runs(self) -> list[dict]:
        """Return the per-pattern child runs, in creation order."""
        return [r for r in self.runs.values() if r["tags"].get("run_type") == "rag_pattern"]


def mlflow_pattern_payload(**overrides) -> dict:
    """Build an ai4rag ``PatternPayload``-shaped dict for MLflow mapping assertions."""
    payload = {
        "name": "rag_pattern_1",
        "iteration": 2,
        "max_combinations": 144,
        "duration_seconds": 12.5,
        "evaluation": {
            "metrics": [
                {
                    "name": "faithfulness",
                    "evaluator": "unitxt",
                    "description": "Faithfulness",
                    "scores": {"mean": 0.82, "ci_low": 0.75, "ci_high": 0.9},
                },
                {
                    "name": "overall_score",
                    "evaluator": "custom",
                    "description": "Aggregate",
                    "scores": {"mean": 0.77, "ci_low": None, "ci_high": None},
                    "optimization_metric": True,
                },
            ]
        },
        "settings": {
            "vector_store_binding": {"provider_type": "milvus", "collection_name": "c1"},
            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
            "embedding": {"model_id": "embed-a", "embedding_params": {"truncate_input_tokens": 512}},
            "retrieval": {"method": "window", "number_of_chunks": 5, "search_mode": "dense", "window_size": 2},
            "generation": {
                "model_id": "gen-a",
                "context_template_text": "ctx",
                "user_message_text": "user",
                "system_message_text": "system",
            },
        },
    }
    payload.update(overrides)
    return payload
