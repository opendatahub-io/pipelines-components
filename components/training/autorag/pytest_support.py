"""Test-only helpers for AutoRAG training component unit tests."""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from unittest import mock


def autorag_shared_dir() -> Path:
    """Path to ``components/training/autorag/shared`` embedded in training components."""
    return Path(__file__).resolve().parent / "shared"


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
    shared_dir = autorag_shared_dir()
    embed_root = embedded_path or str(shared_dir)

    def wrapper(*args, **kwargs):
        if "embedded_artifact" in signature.parameters and kwargs.get("embedded_artifact") is None:
            embedded = mock.MagicMock()
            embedded.path = embed_root
            kwargs["embedded_artifact"] = embedded
        if "component_status" in signature.parameters and kwargs.get("component_status") is None:
            status = mock.MagicMock()
            status.path = str(tmp_path / "component_status_out")
            status.metadata = {}
            kwargs["component_status"] = status
        return original(*args, **kwargs)

    wrapper = functools.wraps(original)(wrapper)
    monkeypatch.setattr(component, "python_func", wrapper)
