"""Component-local status tracking for AutoRAG pipelines.

Each component publishes its own stage progress to an artifact. No workspace required.
The dashboard aggregates component statuses to show overall pipeline progress.

Usage:
    from kfp_components.components.training.autorag.shared.component_status import (
        component_status_tracker,
    )

    status = component_status_tracker(component_status, "test_data_loader")
    with status:
        with status.stage("load_benchmark"):
            ...
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def utc_now_z() -> str:
    """Return current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ComponentStatusEncoder(json.JSONEncoder):
    """Custom JSON encoder for component status data.

    Handles types commonly found in component metadata that aren't JSON-serializable
    by default: datetime, Path, bytes, sets, and other objects.
    """

    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-compatible types."""
        if isinstance(obj, datetime):
            return obj.isoformat().replace("+00:00", "Z") if obj.tzinfo is not None else obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        if isinstance(obj, set):
            return sorted(obj, key=str)
        return super().default(obj)


COMPONENT_STATUS_FILENAME = "component_status.json"

STATUS_STARTED = "started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_VALID_STATES = frozenset({STATUS_STARTED, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED})


class NullComponentStatusTracker:
    """No-op status tracker for notebook or direct ``python_func`` invocations."""

    def set_metadata(self, **_metadata: Any) -> None:
        """Ignore metadata updates."""

    def record(self, *_args: Any, **_kwargs: Any) -> None:
        """Ignore status records."""

    def __enter__(self) -> NullComponentStatusTracker:
        """Enter the no-op context."""
        return self

    def __exit__(self, _exc_type: type[BaseException] | None, _exc: BaseException | None, _tb: Any) -> bool:
        """Exit the no-op context."""
        return False

    @contextmanager
    def stage(self, _stage_id: str, **_start_kwargs: Any) -> Iterator[None]:
        """Yield without recording stage transitions."""
        yield


def null_component_status_tracker() -> NullComponentStatusTracker:
    """Return a tracker that skips all status persistence."""
    return NullComponentStatusTracker()


class ComponentStatusTracker:
    """Track stage-level progress within a single AutoRAG component.

    Emits the canonical AutoX component_status.json schema: each stage carries a
    nested ``status`` object (state, optional step/message/running_at), an optional
    ``metrics`` bag for counters, and an optional ``error`` string.
    """

    def __init__(self, artifact_path: str, component_id: str) -> None:
        """Initialize the status tracker.

        Args:
            artifact_path: Path to the KFP artifact directory where component_status.json will be written.
            component_id: Unique component identifier (e.g., "test_data_loader").
        """
        self.artifact_path = Path(artifact_path)
        self.component_id = component_id
        self.stages: list[dict[str, Any]] = []
        self.started_at = utc_now_z()
        self.completed_at: str | None = None
        self.metadata: dict[str, Any] = {}

    def record(
        self,
        stage_id: str,
        state: str,
        *,
        step: str | None = None,
        message: dict[str, str] | None = None,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Record or update a stage's status.

        Args:
            stage_id: Stage identifier (e.g., "load_benchmark", "optimize_templates").
            state: Stage state ("started", "running", "completed", "failed").
            step: Current sub-step id. Only when the stage map lists steps[].
            message: Status message dict with "level" (info/warning/error) and "text".
            metrics: Counters and measurements for this stage (e.g., {"rows": 1000}).
            error: Error description. Only when state is "failed".
        """
        if state not in _VALID_STATES:
            raise ValueError(f"state must be one of {sorted(_VALID_STATES)}; got {state!r}")

        status_obj: dict[str, Any] = {"state": state}
        if step is not None:
            status_obj["step"] = step
        if message is not None:
            status_obj["message"] = message
        if state == STATUS_RUNNING:
            status_obj["running_at"] = utc_now_z()

        stage_data: dict[str, Any] = {"id": stage_id, "status": status_obj}
        if metrics:
            stage_data["metrics"] = metrics
        if error is not None:
            stage_data["error"] = error

        existing_idx = next((i for i, s in enumerate(self.stages) if s["id"] == stage_id), None)

        if existing_idx is not None:
            existing = self.stages[existing_idx]
            existing["status"] = stage_data["status"]
            if "metrics" in stage_data:
                existing.setdefault("metrics", {}).update(stage_data["metrics"])
            if state == STATUS_FAILED:
                if "error" in stage_data:
                    existing["error"] = stage_data["error"]
            else:
                existing.pop("error", None)
        else:
            self.stages.append(stage_data)

        logger.info(
            "COMPONENT_STATUS component=%s stage=%s state=%s%s",
            self.component_id,
            stage_id,
            state,
            f" metrics={metrics}" if metrics else "",
        )

    def set_metadata(self, **metadata: Any) -> None:
        """Set component-level metadata."""
        self.metadata.update(metadata)

    def _is_finished(self) -> bool:
        """Return True if all recorded stages are completed or any stage has failed."""
        if not self.stages:
            return False
        states = [s["status"]["state"] for s in self.stages]
        if STATUS_FAILED in states:
            return True
        return all(s == STATUS_COMPLETED for s in states)

    def save(self) -> None:
        """Write the final status to the artifact."""
        self.artifact_path.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "component_id": self.component_id,
            "started_at": self.started_at,
            "stages": self.stages,
            "metadata": self.metadata,
        }

        if self._is_finished():
            self.completed_at = self.completed_at or utc_now_z()
            data["completed_at"] = self.completed_at

        output_file = self.artifact_path / COMPONENT_STATUS_FILENAME
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=ComponentStatusEncoder)

        logger.info(
            "COMPONENT_STATUS component=%s saved status with %d stages to %s",
            self.component_id,
            len(self.stages),
            output_file,
        )

    def save_best_effort(self) -> None:
        """Write status to the artifact, logging instead of raising on I/O errors."""
        try:
            self.save()
        except Exception:
            logger.exception(
                "Failed to save component status for %s",
                self.component_id,
            )

    def mark_active_failed(self, error: str | BaseException) -> None:
        """Mark the in-progress stage as failed, or the last open stage if none is active."""
        if isinstance(error, BaseException):
            error_msg = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        else:
            error_msg = error

        active_statuses = (STATUS_STARTED, STATUS_RUNNING)
        for stage in reversed(self.stages):
            if stage["status"]["state"] in active_statuses:
                self.record(stage["id"], STATUS_FAILED, error=error_msg)
                return

        if self.stages and self.stages[-1]["status"]["state"] != STATUS_COMPLETED:
            self.record(self.stages[-1]["id"], STATUS_FAILED, error=error_msg)
            return

        self.set_metadata(status=STATUS_FAILED, error=error_msg)

    @contextmanager
    def stage(
        self,
        stage_id: str,
        *,
        step: str | None = None,
        message: dict[str, str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record stage started/completed, or failed when an exception escapes the block."""
        self.record(stage_id, STATUS_STARTED, step=step, message=message, metrics=metrics)
        try:
            yield
        except BaseException as exc:
            error_msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            self.record(stage_id, STATUS_FAILED, error=error_msg)
            raise
        else:
            latest = next((s for s in reversed(self.stages) if s["id"] == stage_id), None)
            if latest is None or latest["status"]["state"] not in (STATUS_COMPLETED, STATUS_FAILED):
                self.record(stage_id, STATUS_COMPLETED)

    def __enter__(self) -> ComponentStatusTracker:
        """Enter context: return this tracker."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        """On exit, mark active stage failed and save status best-effort."""
        if exc is not None:
            self.mark_active_failed(exc)
        self.save_best_effort()
        return False


def _resolve_embedded_module_path(embedded_artifact: Any) -> Path:
    """Return the path to embedded ``component_status.py`` after validating the artifact."""
    if embedded_artifact is None:
        raise ValueError(
            "embedded_artifact is required for component status tracking. "
            "Injected by KFP when using embedded_artifact_path."
        )
    if not getattr(embedded_artifact, "path", None):
        raise ValueError("embedded_artifact.path is missing or empty")

    embedded_path = Path(embedded_artifact.path)
    if embedded_path.is_file():
        module_path = embedded_path
    elif embedded_path.is_dir():
        module_path = embedded_path / "component_status.py"
    else:
        raise ValueError(f"Invalid embedded_artifact.path: {embedded_path}")
    if not module_path.is_file():
        raise ValueError(f"Embedded component_status module not found at {module_path}")
    return module_path


def resolve_embedded_import_root(embedded_artifact: Any) -> Path:
    """Return the directory containing embedded shared assets (e.g. script templates)."""
    embedded_path = Path(embedded_artifact.path)
    if embedded_path.is_file():
        return embedded_path.parent
    if embedded_path.is_dir():
        return embedded_path
    raise ValueError(f"Invalid embedded_artifact.path: {embedded_path}")


def load_embedded_component_status_module(embedded_artifact: Any) -> ModuleType:
    """Load ``component_status`` from a KFP embedded artifact path via importlib."""
    import importlib.util

    module_path = _resolve_embedded_module_path(embedded_artifact)
    spec = importlib.util.spec_from_file_location("_autorag_component_status", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load embedded module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracker_from_embedded(
    embedded_artifact: Any,
    component_status: Any,
    component_id: str,
) -> ComponentStatusTracker:
    """Validate embedded artifact path and return a configured tracker."""
    _resolve_embedded_module_path(embedded_artifact)
    return component_status_tracker(component_status, component_id)


def bootstrap_status_tracker(
    embedded_artifact: Any,
    component_status: Any,
    component_id: str,
) -> ComponentStatusTracker | NullComponentStatusTracker:
    """Load this module from ``embedded_artifact`` and return a configured tracker."""
    if component_status is None:
        return null_component_status_tracker()
    status_module = load_embedded_component_status_module(embedded_artifact)
    return status_module.tracker_from_embedded(embedded_artifact, component_status, component_id)


def component_status_tracker(
    component_status: Any, component_id: str
) -> ComponentStatusTracker | NullComponentStatusTracker:
    """Build a tracker from a KFP component_status output artifact.

    Args:
        component_status: KFP output artifact for component status tracking.
        component_id: Unique component identifier.

    Returns:
        Configured ComponentStatusTracker instance, or a no-op tracker when ``component_status`` is None.
    """
    if component_status is None:
        return null_component_status_tracker()
    return ComponentStatusTracker(component_status.path, component_id)


def load_component_status(artifact_path: str) -> dict[str, Any]:
    """Load component status from an artifact directory.

    Returns an empty dict if the file doesn't exist or is corrupted.
    """
    status_file = Path(artifact_path) / COMPONENT_STATUS_FILENAME
    if not status_file.exists():
        return {}

    try:
        with status_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load status from %s: %s", status_file, e)
        return {}
