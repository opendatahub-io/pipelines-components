"""Component-local status tracking for AutoRAG pipelines.

Each component publishes its own stage progress to an artifact. No workspace required.
The dashboard aggregates component statuses to show overall pipeline progress.

Usage:
    from kfp_components.components.training.autorag.shared.component_status import (
        component_status_tracker,
    )

    status = component_status_tracker(component_status, "test_data_loader")
    with status:
        with status.stage("download_and_sample"):
            ...
"""

from __future__ import annotations

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


class ComponentStatusTracker:
    """Track stage-level progress within a single AutoRAG component."""

    def __init__(self, artifact_path: str | None, component_id: str) -> None:
        """Initialize the status tracker.

        Args:
            artifact_path: Path to the KFP artifact directory where status.json will be written.
                When ``None``, tracking is disabled (e.g. direct unit-test invocations without a mock artifact).
            component_id: Unique component identifier (e.g., "test_data_loader").
        """
        self._enabled = artifact_path is not None
        self.artifact_path = Path(artifact_path) if self._enabled else Path(".")
        self.component_id = component_id
        self.stages: list[dict[str, Any]] = []
        self.started_at = self._utc_now_iso()
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _utc_now_iso() -> str:
        """Return current UTC timestamp in ISO format."""
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def record(self, stage_id: str, status: str, **metadata: Any) -> None:
        """Record or update a stage's status."""
        existing_idx = next((i for i, s in enumerate(self.stages) if s["id"] == stage_id), None)

        stage_data = {
            "id": stage_id,
            "status": status,
            "timestamp": self._utc_now_iso(),
            **metadata,
        }

        if existing_idx is not None:
            self.stages[existing_idx].update(stage_data)
        else:
            self.stages.append(stage_data)

        logger.info(
            "COMPONENT_STATUS component=%s stage=%s status=%s %s",
            self.component_id,
            stage_id,
            status,
            " ".join(f"{k}={v}" for k, v in metadata.items()),
        )

    def set_metadata(self, **metadata: Any) -> None:
        """Set component-level metadata."""
        self.metadata.update(metadata)

    def save(self) -> None:
        """Write the final status to the artifact."""
        if not self._enabled:
            return

        self.artifact_path.mkdir(parents=True, exist_ok=True)

        data = {
            "component_id": self.component_id,
            "started_at": self.started_at,
            "completed_at": self._utc_now_iso(),
            "stages": self.stages,
            "metadata": self.metadata,
        }

        output_file = self.artifact_path / COMPONENT_STATUS_FILENAME
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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

    def mark_active_failed(self, error: str) -> None:
        """Mark the in-progress stage as failed, or the last open stage if none is active."""
        active_statuses = (STATUS_STARTED, STATUS_RUNNING)
        for stage in reversed(self.stages):
            if stage.get("status") in active_statuses:
                self.record(stage["id"], STATUS_FAILED, error=error)
                return

        if self.stages and self.stages[-1].get("status") != STATUS_COMPLETED:
            self.record(self.stages[-1]["id"], STATUS_FAILED, error=error)
            return

        self.set_metadata(status=STATUS_FAILED, error=error)

    @contextmanager
    def stage(self, stage_id: str, **start_metadata: Any) -> Iterator[None]:
        """Record stage started/completed, or failed when an exception escapes the block."""
        self.record(stage_id, STATUS_STARTED, **start_metadata)
        try:
            yield
        except Exception as exc:
            self.record(stage_id, STATUS_FAILED, error=str(exc))
            raise
        else:
            self.record(stage_id, STATUS_COMPLETED)

    def __enter__(self) -> ComponentStatusTracker:
        """Enter context: return this tracker."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        """On exit, mark active stage failed and save status best-effort."""
        if exc is not None:
            self.mark_active_failed(str(exc))
        self.save_best_effort()
        return False


def component_status_tracker(component_status: Any, component_id: str) -> ComponentStatusTracker:
    """Build a tracker from an optional KFP ``component_status`` output artifact."""
    artifact_path = component_status.path if component_status is not None else None
    return ComponentStatusTracker(artifact_path, component_id)


def load_component_status(artifact_path: str) -> dict[str, Any]:
    """Load component status from an artifact directory."""
    status_file = Path(artifact_path) / COMPONENT_STATUS_FILENAME
    if not status_file.exists():
        return {}

    with status_file.open("r", encoding="utf-8") as f:
        return json.load(f)
