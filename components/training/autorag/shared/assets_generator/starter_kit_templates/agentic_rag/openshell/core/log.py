"""Logging setup and the single subprocess runner used by the kit CLI.

The CLI keeps the shell-era output shape (``==> step``, indented details,
``WARNING:``/``ERROR:`` on stderr) but routes everything through ``logging``.
Pass ``--verbose`` (or ``KIT_VERBOSE=1``) to log every external command.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

_LOGGER = logging.getLogger("kit")
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_STATE = {"verbose": False}
_SECRET_ARG_RE = re.compile(
    r"^((?:--from-literal=)?[A-Za-z0-9_]*(?:KEY|TOKEN|PASSWORD|SECRET)[A-Za-z0-9_]*)=.*$",
    re.IGNORECASE | re.DOTALL,
)


def _redact(argv: Sequence[str]) -> str:
    """Render *argv* as a shell line with credential values masked."""
    return shlex.join(_SECRET_ARG_RE.sub(r"\1=***", str(part)) for part in argv)


class _MaxLevel(logging.Filter):
    """Route records up to ``level`` to the stdout handler."""

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.level


class KitError(RuntimeError):
    """Raised when a kit command cannot continue."""


def configure(verbose: bool = False) -> None:
    """Configure the ``kit`` logger once, without timestamps or logger names."""
    _STATE["verbose"] = verbose or os.environ.get("KIT_VERBOSE", "").strip().lower() in _TRUTHY
    _LOGGER.setLevel(logging.DEBUG if _STATE["verbose"] else logging.INFO)
    if _LOGGER.handlers:
        return
    _LOGGER.propagate = False
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG)
    stdout.addFilter(_MaxLevel(logging.INFO))
    stdout.setFormatter(logging.Formatter("%(message)s"))
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    _LOGGER.addHandler(stdout)
    _LOGGER.addHandler(stderr)


def is_verbose() -> bool:
    """Return True when command tracing is enabled."""
    return bool(_STATE["verbose"])


def dry_run() -> bool:
    """Return True when ``KIT_DRY_RUN`` asks for a command preview."""
    return os.environ.get("KIT_DRY_RUN", "").strip().lower() in _TRUTHY


def info(message: str = "") -> None:
    """Write a plain line to stdout."""
    _LOGGER.info(message)


def step(message: str) -> None:
    """Write a top-level progress line."""
    _LOGGER.info("==> %s", message)


def detail(message: str) -> None:
    """Write an indented detail line."""
    _LOGGER.info("    %s", message)


def warn(message: str) -> None:
    """Write a warning to stderr."""
    _LOGGER.warning(message)


def error(message: str) -> None:
    """Write an error to stderr."""
    _LOGGER.error(message)


def run(
    cmd: Sequence[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: Mapping[str, str] | None = None,
    stdin_text: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and return the completed process.

    ``KIT_DRY_RUN=1`` logs the command instead of executing it, and
    ``--verbose`` logs every command that is really executed.
    """
    argv = [str(part) for part in cmd]
    if dry_run():
        detail("+ " + _redact(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    if is_verbose():
        _LOGGER.debug("+ %s", _redact(argv))

    run_env = dict(os.environ)
    if env:
        run_env.update({key: str(value) for key, value in env.items()})
    try:
        result = subprocess.run(
            argv,
            check=False,
            text=True,
            input=stdin_text,
            capture_output=capture,
            stdin=subprocess.DEVNULL if stdin_text is None else None,
            env=run_env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        expired_stdout = expired.stdout or ""
        expired_stderr = expired.stderr or ""
        if isinstance(expired_stdout, bytes):
            expired_stdout = expired_stdout.decode("utf-8", "replace")
        if isinstance(expired_stderr, bytes):
            expired_stderr = expired_stderr.decode("utf-8", "replace")
        return subprocess.CompletedProcess(argv, 124, stdout=expired_stdout, stderr=expired_stderr)
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        message = f"command failed ({result.returncode}): {_redact(argv)}"
        raise KitError(f"{message}\n{stderr}" if stderr else message)
    return result


def output(cmd: Sequence[str], *, check: bool = True) -> str:
    """Run ``cmd`` and return its trimmed stdout."""
    return (run(cmd, check=check, capture=True).stdout or "").strip()
