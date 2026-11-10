"""Resolve kit deployment inputs from the environment, ``.env``, and ``values.yaml``.

Precedence matches the previous shell implementation: variables defined in
``.env`` win over the process environment because ``.env`` used to be sourced
with ``set -a``; both win over ``values.yaml`` and the built-in defaults.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

_SECTION_RE = re.compile(r"^([A-Za-z0-9_.-]+):\s*(.*)$")
_COMMENT_RE = re.compile(r"\s+#")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _clean(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in "\"'" and value[0] in value[1:]:
        end = value.index(value[0], 1)
        return value[1:end]
    return _COMMENT_RE.split(value, maxsplit=1)[0].strip()


def parse_values(text: str) -> dict[str, dict[str, str]]:
    """Parse the flat ``section:`` + ``key: value`` subset used by ``values.yaml``."""
    data: dict[str, dict[str, str]] = {}
    section = ""
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indented = raw[0] in " \t"
        line = raw.strip()
        if not indented:
            match = _SECTION_RE.match(line)
            if match and not match.group(2).strip():
                section = match.group(1)
                data.setdefault(section, {})
                continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        data.setdefault(section, {})[key.strip()] = _clean(value)
    return data


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse the ``KEY=VALUE`` subset of a ``.env`` file."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        values[key.strip()] = _clean(value)
    return values


def slugify(value: str) -> str:
    """Normalize a value into a DNS-1123 label usable for images, routes, and SAs."""
    return _SLUG_RE.sub("-", value.lower()).strip("-")[:63].strip("-")


def _default_namespace() -> str:
    from openshell.core import log  # imported lazily to keep config importable without a cluster

    try:
        return log.output(["oc", "project", "-q"], check=False)
    except OSError:
        return ""


def _load_sources(
    root: Path | None,
    environ: Mapping[str, str] | None,
) -> tuple[Path, dict[str, str], dict[str, dict[str, str]]]:
    """Return the kit root, the merged environment, and parsed ``values.yaml``."""
    kit_root = Path(root) if root else Path(__file__).resolve().parents[2]
    env = dict(os.environ if environ is None else environ)
    dotenv = kit_root / ".env"
    if dotenv.is_file():
        env.update(parse_dotenv(dotenv.read_text(encoding="utf-8")))
    values_path = kit_root / "values.yaml"
    values = parse_values(values_path.read_text(encoding="utf-8")) if values_path.is_file() else {}
    return kit_root, env, values


def _app_name(kit_root: Path, env: dict[str, str], values: dict[str, dict[str, str]]) -> str:
    """Resolve the application name from env, ``values.yaml``, or the directory name."""
    raw = (
        env.get("APP_NAME")
        or values.get("deployment", {}).get("app_name", "")
        or values.get("", {}).get("nameOverride", "")
        or kit_root.name
    )
    return slugify(raw) or slugify(kit_root.name)


@dataclass
class KitConfig:  # pylint: disable=too-many-instance-attributes
    """Every deployment input plus the names derived from the application name."""

    root: Path
    app_name: str
    namespace: str
    workspace: str
    gateway_name: str
    openshell_namespace: str
    image: str
    image_ref: str
    sandbox_name: str
    route_name: str
    agent_sa_name: str
    agent_token_secret: str
    sandbox_sa: str
    cli_state: Path
    create_attempts: int
    values: dict[str, dict[str, str]] = field(default_factory=dict)
    environ: dict[str, str] = field(default_factory=dict)

    @property
    def sandbox_image(self) -> str:
        """Image reference the sandbox is created from."""
        if self.image_ref:
            return self.image_ref
        return f"image-registry.openshift-image-registry.svc:5000/{self.namespace}/{self.image}:latest"

    @property
    def local_cli(self) -> str:
        """Return the installed ``openshell`` command, or an empty string."""
        return shutil.which("openshell") or ""

    def value(self, name: str, section: str = "", key: str = "", default: str = "") -> str:
        """Return an environment value, then ``values.yaml``, then ``default``."""
        for candidate in (self.environ.get(name), self.values.get(section, {}).get(key)):
            if candidate:
                return candidate
        return default

    @classmethod
    def load(
        cls,
        root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        lookup_namespace: Callable[[], str] | None = None,
    ) -> "KitConfig":
        """Build the configuration for the kit rooted at ``root``."""
        kit_root, env, values = _load_sources(root, environ)

        def value(name: str, section: str, key: str, default: str = "") -> str:
            for candidate in (env.get(name), values.get(section, {}).get(key)):
                if candidate:
                    return candidate
            return default

        app_name = _app_name(kit_root, env, values)
        namespace = value("APP_NAMESPACE", "deployment", "namespace") or (
            (lookup_namespace or _default_namespace)() or "default"
        )
        sandbox_name = slugify(value("OPENSHELL_SANDBOX_NAME", "deployment", "sandbox_name", app_name)) or app_name
        agent_sa_name = slugify(value("AGENT_SA_NAME", "", "", app_name)) or app_name
        attempts = env.get("SANDBOX_CREATE_ATTEMPTS", "").strip()
        return cls(
            root=kit_root,
            app_name=app_name,
            namespace=namespace,
            workspace=value("OPENSHELL_WORKSPACE", "deployment", "workspace", "default"),
            gateway_name=value("OPENSHELL_GATEWAY_NAME", "deployment", "gateway_name", "openshift"),
            openshell_namespace=env.get("OPENSHELL_NAMESPACE") or "openshell",
            image=slugify(env.get("OPENSHELL_IMAGE") or app_name),
            image_ref=value("OPENSHELL_IMAGE_REF", "deployment", "image"),
            sandbox_name=sandbox_name,
            route_name=slugify(env.get("OPENSHELL_ROUTE_NAME") or app_name),
            agent_sa_name=agent_sa_name,
            agent_token_secret=env.get("AGENT_TOKEN_SECRET") or f"{agent_sa_name}-token",
            sandbox_sa=env.get("OPENSHELL_SANDBOX_SA") or "openshell-sandbox",
            cli_state=Path(env.get("OPENSHELL_CLI_STATE") or Path.home() / ".config" / "openshell"),
            create_attempts=int(attempts) if attempts.isdigit() else 3,
            values=values,
            environ=env,
        )

    def as_dict(self) -> dict[str, str]:
        """Flatten the configuration for debugging output."""
        return {
            "app_name": self.app_name,
            "namespace": self.namespace,
            "workspace": self.workspace,
            "gateway_name": self.gateway_name,
            "openshell_namespace": self.openshell_namespace,
            "image": self.image,
            "image_ref": self.image_ref,
            "sandbox_name": self.sandbox_name,
            "sandbox_image": self.sandbox_image,
            "route_name": self.route_name,
            "agent_sa_name": self.agent_sa_name,
            "agent_token_secret": self.agent_token_secret,
            "sandbox_sa": self.sandbox_sa,
        }
