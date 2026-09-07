"""Component-local status tracking for AutoML pipelines.

Each component publishes its own stage progress to an artifact. No workspace required.
The dashboard aggregates component statuses to show overall pipeline progress.

Usage:
    with ComponentStatusTracker(artifact.path, "autogluon_models_training") as status:
        status.record("load_data", "started")
        status.record("load_data", "completed", metrics={"rows": 1000})
        with status.stage("model_selection"):
            ...  # marks started/completed; marks failed on exception
    # context exit saves best-effort and marks active stage failed on error
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COMPONENT_STATUS_FILENAME = "component_status.json"

STATUS_STARTED = "started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_VALID_STATES = frozenset({STATUS_STARTED, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED})


class ComponentStatusEncoder(json.JSONEncoder):
    """JSON encoder for component status metadata (Path, bytes, set, datetime)."""

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


def utc_now_z() -> str:
    """Return current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ComponentStatusTracker:
    """Track stage-level progress within a single component.

    Publishes component-local status to an artifact without requiring workspace.
    Each component independently tracks its stages and metadata.

    Emits the canonical AutoX component_status.json schema: each stage carries a
    nested ``status`` object (state, optional step/message/running_at), an optional
    ``metrics`` bag for counters, and an optional ``error`` string.
    """

    def __init__(self, artifact_path: str, component_id: str) -> None:
        """Initialize the status tracker.

        Args:
            artifact_path: Path to the KFP artifact directory where component_status.json will be written.
            component_id: Unique component identifier (e.g., "autogluon_models_training").
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

        If the stage already exists, it will be updated. Otherwise, a new stage is appended.

        Args:
            stage_id: Stage identifier (e.g., "load_data", "model_selection").
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
        """Set component-level metadata.

        Args:
            **metadata: Key-value pairs to store at component level.
                        Must include display_name before save().
        """
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
        """Write the final status to the artifact.

        Creates the artifact directory if needed and writes component_status.json
        with all recorded stages and metadata.
        """
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


def load_component_status(artifact_path: str) -> dict[str, Any]:
    """Load component status from an artifact.

    Utility function for dashboards/monitoring to read component status.

    Args:
        artifact_path: Path to the artifact directory containing component_status.json.

    Returns:
        Dict containing component_id, started_at, completed_at, stages, and metadata.
        Returns empty dict if file doesn't exist or is unreadable.
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


_ALLOWED_STAGE_KEYS = frozenset({"id", "status", "metrics", "error"})


def validate_component_status_document(data: dict[str, Any]) -> None:
    """Raise ValueError when ``data`` does not match the canonical component_status schema.

    Stages use a nested ``status`` object and optional ``metrics`` bag; counters must not
    be emitted as top-level stage keys.
    """
    required = {"component_id", "started_at", "stages", "metadata"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"component_status missing required fields: {sorted(missing)}")

    if not isinstance(data["stages"], list):
        raise ValueError("component_status stages must be a list")

    for stage in data["stages"]:
        if not isinstance(stage, dict):
            raise ValueError("each component_status stage must be an object")
        extra = set(stage.keys()) - _ALLOWED_STAGE_KEYS
        if extra:
            raise ValueError(f"component_status stage has unexpected top-level keys: {sorted(extra)}")
        status_obj = stage.get("status")
        if not isinstance(status_obj, dict) or "state" not in status_obj:
            raise ValueError("each component_status stage must include status.state")
