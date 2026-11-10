"""OpenShift helpers: images, BuildConfigs, routes, and secrets."""

from __future__ import annotations

import base64
import json
import re
import time

from openshell.core import log
from openshell.core.config import KitConfig
from openshell.core.log import KitError

VECTOR_STORE_KEYS = (
    "MILVUS_URI",
    "MILVUS_TOKEN",
    "MILVUS_CA_CERT",
    "PGVECTOR_HOST",
    "PGVECTOR_PORT",
    "PGVECTOR_DB",
    "PGVECTOR_USER",
    "PGVECTOR_PASSWORD",
    "PGVECTOR_CA_CERT",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "NEO4J_CA_CERT",
)
_PEM_MARKER = "-----BEGIN CERTIFICATE-----"
_BASE_IMAGE_ARG_RE = re.compile(r"^\s*ARG\s+BASE_IMAGE=(.+?)\s*$", re.MULTILINE)


def oc(*args: str, check: bool = True, capture: bool = False, stdin_text: str | None = None):
    """Run ``oc`` with the given arguments."""
    return log.run(["oc", *args], check=check, capture=capture, stdin_text=stdin_text)


def oc_output(*args: str, check: bool = True) -> str:
    """Return trimmed stdout of ``oc``."""
    return log.output(["oc", *args], check=check)


def current_namespace() -> str:
    """Return the namespace selected in the local kubeconfig."""
    return oc_output("project", "-q", check=False)


def sandbox_base_image(config: KitConfig) -> str:
    """Resolve the sandbox base image from ``OPENSHELL_BASE_IMAGE``."""
    image = config.environ.get("OPENSHELL_BASE_IMAGE", "")
    if not image:
        log.detail("Using the Containerfile BASE_IMAGE default.")
        return ""
    log.detail(f"Using sandbox base image: {image}")
    return image


def containerfile_base_image(config: KitConfig) -> str:
    """Return the Containerfile's default ``BASE_IMAGE`` value."""
    path = config.root / "Containerfile.openshell"
    if not path.is_file():
        return ""
    match = _BASE_IMAGE_ARG_RE.search(path.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else ""


def cluster_domain() -> str:
    """Return the cluster applications domain."""
    return oc_output("get", "ingresses.config", "cluster", "-o", "jsonpath={.spec.domain}")


def api_server_url() -> str:
    """Return the Kubernetes API URL used by the local client."""
    return oc_output("whoami", "--show-server")


def route_host(namespace: str, name: str) -> str:
    """Return a Route hostname, or an empty string when the route is absent."""
    result = oc("get", "route", name, "-n", namespace, "-o", "jsonpath={.spec.host}", check=False, capture=True)
    return (result.stdout or "").strip()


def latest_build(namespace: str, build_config: str) -> str:
    """Return the newest build name for a BuildConfig."""
    result = oc(
        "get",
        "builds",
        "-n",
        namespace,
        "-l",
        f"build={build_config}",
        "--sort-by=.metadata.creationTimestamp",
        "-o",
        "name",
        check=False,
        capture=True,
    )
    names = (result.stdout or "").strip().splitlines()
    return names[-1].strip() if names else ""


def _patch_build_args(config: KitConfig, base_image: str) -> None:
    patch = json.dumps(
        {"spec": {"strategy": {"dockerStrategy": {"buildArgs": [{"name": "BASE_IMAGE", "value": base_image}]}}}}
    )
    oc("patch", "bc", config.image, "-n", config.namespace, "--type=merge", "-p", patch)


def _clear_build_args(config: KitConfig) -> None:
    """Drop a previously pinned BASE_IMAGE so the Containerfile default applies."""
    oc(
        "patch",
        "bc",
        config.image,
        "-n",
        config.namespace,
        "--type=merge",
        "-p",
        '{"spec":{"strategy":{"dockerStrategy":{"buildArgs":[]}}}}',
    )


def _run_build(config: KitConfig) -> tuple[str, str]:
    if log.dry_run():
        log.run(["oc", "start-build", config.image, "-n", config.namespace, "--from-dir", str(config.root), "--follow"])
        return "Complete", ""
    before = latest_build(config.namespace, config.image)
    log.run(
        ["oc", "start-build", config.image, "-n", config.namespace, "--from-dir", str(config.root), "--follow"],
        check=False,
    )
    build_name = latest_build(config.namespace, config.image)
    if not build_name:
        raise KitError("OpenShift did not create a new build.")
    if build_name == before:
        raise KitError("OpenShift did not create a new build.")
    phase = ""
    for _ in range(120):
        phase = oc_output("get", build_name, "-n", config.namespace, "-o", "jsonpath={.status.phase}", check=False)
        if phase in ("Complete", "Failed", "Cancelled", "Error"):
            break
        time.sleep(5)
    reason = oc_output("get", build_name, "-n", config.namespace, "-o", "jsonpath={.status.reason}", check=False)
    log.detail(f"Build {build_name} finished with phase {phase or 'Unknown'}{f' ({reason})' if reason else ''}.")
    return phase or "Unknown", reason


def build_openshell(config: KitConfig) -> None:
    """Create or update the sandbox BuildConfig and run a binary build."""
    log.run(
        [
            "oc",
            "new-build",
            "-n",
            config.namespace,
            "--strategy=docker",
            "--binary",
            f"--name={config.image}",
            f"--to={config.image}:latest",
        ],
        check=False,
        capture=True,
    )
    oc(
        "patch",
        "bc",
        config.image,
        "-n",
        config.namespace,
        "-p",
        '{"spec":{"strategy":{"dockerStrategy":{"dockerfilePath":"Containerfile.openshell"}}}}',
    )
    base_image = sandbox_base_image(config)
    if base_image:
        _patch_build_args(config, base_image)
    else:
        _clear_build_args(config)
    resolved_base = base_image or containerfile_base_image(config)
    phase, reason = _run_build(config)
    if phase == "Failed" and "Pull" in reason:
        fallback = resolved_base.replace("registry.redhat.io", "quay.io")
        if fallback != resolved_base:
            log.detail(f"Base image pull failed; retrying with {fallback}")
            _patch_build_args(config, fallback)
            phase, reason = _run_build(config)
    if phase != "Complete":
        raise KitError(f"OpenShift image build did not complete (phase {phase}, reason {reason or 'unknown'}).")
    log.info("")
    log.info(f"Image ready: {config.sandbox_image}")
    log.info("Next: make deploy-openshell")


def refresh_certs(config: KitConfig, cli) -> None:
    """Write mTLS certificates and register the cluster gateway."""
    log.step("Refreshing openShell client certificates...")
    mtls_dir = config.cli_state / "gateways" / config.gateway_name / "mtls"
    mtls_dir.mkdir(parents=True, exist_ok=True)
    for key in ("ca.crt", "tls.crt", "tls.key"):
        escaped = key.replace(".", "\\.")
        encoded = (
            oc(
                "get",
                "secret",
                "openshell-client-tls",
                "-n",
                config.openshell_namespace,
                "-o",
                f"jsonpath={{.data.{escaped}}}",
                capture=True,
            ).stdout
            or ""
        )
        if log.dry_run():
            continue
        (mtls_dir / key).write_bytes(base64.b64decode(encoded))
    if not log.dry_run():
        (mtls_dir / "tls.key").chmod(0o600)
    log.detail("Certificates refreshed.")

    if not cli.connected():
        host = route_host(config.openshell_namespace, "openshell-gateway")
        if host:
            log.step(f"Registering gateway '{config.gateway_name}' at https://{host}...")
            cli.gateway(["gateway", "remove", config.gateway_name], check=False)
            cli.gateway(["gateway", "add", f"https://{host}", "--local", "--name", config.gateway_name])
    if not cli.connected():
        raise KitError(
            f"ERROR: gateway '{config.gateway_name}' is not registered or not reachable over mTLS.\n"
            "       The gateway is a cluster prerequisite: run scripts/setup-openshell-gateway.sh."
        )


def grant_image_pullers(config: KitConfig) -> None:
    """Allow the OpenShell namespace to pull the kit image."""
    log.step(f"Granting image-pull to {config.openshell_namespace} namespace...")
    for service_account in (config.sandbox_sa, "default"):
        oc(
            "policy",
            "add-role-to-user",
            "system:image-puller",
            f"system:serviceaccount:{config.openshell_namespace}:{service_account}",
            "-n",
            config.namespace,
            check=False,
            capture=True,
        )


def read_secret(namespace: str, name: str, key: str) -> str:
    """Return a decoded secret entry, or an empty string when absent."""
    result = oc("get", "secret", name, "-n", namespace, "-o", "json", check=False, capture=True)
    try:
        data = json.loads(result.stdout or "{}").get("data", {})
    except json.JSONDecodeError:
        return ""
    if key not in data:
        return ""
    try:
        return base64.b64decode(data[key]).decode("utf-8").strip()
    except (ValueError, UnicodeDecodeError):
        return ""


def load_secrets(config: KitConfig) -> dict[str, str]:
    """Read MaaS and vector database settings from the configured secrets."""
    env = dict(config.environ)
    maas_name = config.values.get("secrets", {}).get("maas_secret_name", "")
    db_name = env.get("DB_SECRET_NAME") or config.values.get("secrets", {}).get("db_secret_name", "")
    if maas_name and maas_name != "__MAAS_SECRET_NAME__":
        for key in ("MAAS_BASE_URL", "MAAS_API_KEY"):
            value = read_secret(config.namespace, maas_name, key)
            if value:
                env[key] = value
    if db_name and db_name != "__DB_SECRET_NAME__":
        for key in VECTOR_STORE_KEYS:
            if env.get(key):
                continue
            value = read_secret(config.namespace, db_name, key)
            if value:
                env[key] = value
    for key in ("MILVUS_CA_CERT", "PGVECTOR_CA_CERT", "NEO4J_CA_CERT"):
        value = env.get(key, "")
        if value and _PEM_MARKER in value:
            env[key] = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return env


def apply_route(config: KitConfig, host: str) -> None:
    """Create or update the passthrough Route that fronts the sandbox service."""
    manifest = (
        "apiVersion: route.openshift.io/v1\n"
        "kind: Route\n"
        "metadata:\n"
        f"  name: {config.route_name}\n"
        f"  namespace: {config.openshell_namespace}\n"
        "spec:\n"
        f'  host: "{host}"\n'
        "  to:\n"
        "    kind: Service\n"
        "    name: openshell\n"
        "  port:\n"
        "    targetPort: grpc\n"
        "  tls:\n"
        "    termination: passthrough\n"
    )
    oc("apply", "-f", "-", stdin_text=manifest)


def delete_route(config: KitConfig) -> None:
    """Delete the kit Route, ignoring a missing object."""
    oc("delete", "route", config.route_name, "-n", config.openshell_namespace, check=False, capture=True)


def config_map_value(namespace: str, name: str, key: str) -> str:
    """Return a ConfigMap entry, or an empty string when absent."""
    result = oc("get", "configmap", name, "-n", namespace, "-o", "json", check=False, capture=True)
    try:
        return json.loads(result.stdout or "{}").get("data", {}).get(key, "")
    except json.JSONDecodeError:
        return ""


def ensure_service_account(namespace: str, name: str) -> None:
    """Create a ServiceAccount if it does not exist yet."""
    oc("create", "sa", name, "-n", namespace, check=False, capture=True)


def create_token(namespace: str, name: str, duration: str = "24h") -> str:
    """Create a bound ServiceAccount token (default: 24h).

    For production use, rotate tokens daily via ``make rotate-tokens``.
    """
    return oc_output("create", "token", name, "-n", namespace, f"--duration={duration}")


def upsert_token_secret(config: KitConfig, token: str) -> None:
    """Store the agent client token in the kit namespace."""
    oc("delete", "secret", config.agent_token_secret, "-n", config.namespace, check=False, capture=True)
    oc(
        "create",
        "secret",
        "generic",
        config.agent_token_secret,
        "-n",
        config.namespace,
        f"--from-literal=token={token}",
    )
