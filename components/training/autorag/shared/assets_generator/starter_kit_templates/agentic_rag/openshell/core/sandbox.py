"""OpenShell sandbox lifecycle: CLI container, create, egress, agent, route."""

from __future__ import annotations

import base64
import json
import re
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from openshell.core import cluster, log
from openshell.core.config import KitConfig
from openshell.core.log import KitError

PYTHON_BIN = "/usr/bin/python3.12"
SANDBOX_PYTHONPATH = (
    "/sandbox/.local/lib/python3.12/site-packages:/sandbox/.local/lib64/python3.12/site-packages:/sandbox"
)
WAIT_MAX_ATTEMPTS = 30
WAIT_POLL_INTERVAL_SECONDS = 5
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_POLICY_FILE = ".openshell-policy.yaml"
_POLICY_MOUNTED = "/work/" + _POLICY_FILE
_ENV_KEYS = ("MAAS_BASE_URL", "MAAS_API_KEY", "AGENT_CONFIG_B64", *cluster.VECTOR_STORE_KEYS)
_STOP_SCRIPT = (
    "import os,signal; from pathlib import Path; pid=Path('/sandbox/agent.pid'); "
    "os.kill(int(pid.read_text().strip()),signal.SIGTERM) if pid.is_file() else None"
)
_HEALTH_SCRIPT = (
    "import sys,urllib.request; response=urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3); "
    "sys.exit(0 if response.status == 200 else 1)"
)
_LAUNCH_SCRIPT = (
    "import subprocess,sys,time; log=open('/sandbox/agent.log','ab',buffering=0); "
    "cmd=['/usr/bin/python3.12','/sandbox/proxychains_entrypoint.py','/usr/bin/python3.12','-m','uvicorn',"
    "'auth_wrapper:app','--host','0.0.0.0','--port','8080']; "
    "process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,"
    "start_new_session=True,close_fds=True); open('/sandbox/agent.pid','w').write(str(process.pid)); "
    "time.sleep(3); alive=process.poll() is None; "
    "print('Agent process alive' if alive else 'ERROR: Agent died on startup'); sys.exit(0 if alive else 1)"
)


@dataclass
class Context:
    """Resolved runtime inputs shared by the deploy steps."""

    config: KitConfig
    env: dict[str, str]
    agent_config_b64: str
    provider: str

    def get(self, key: str, default: str = "") -> str:
        """Return a resolved environment value."""
        if key == "AGENT_CONFIG_B64":
            return self.agent_config_b64 or self.env.get(key) or default
        return self.env.get(key) or default


class OpenShell:
    """Thin wrapper around the OpenShell CLI, local or containerized."""

    def __init__(self, config: KitConfig) -> None:
        self.config = config
        self._image: str | None = None

    def _cli_image(self) -> str:
        if self._image is None:
            self._image = self._resolve_cli_image()
        return self._image

    def _resolve_cli_image(self) -> str:
        image = cluster.openshell_cli_image(self.config)
        fallback_registry = (self.config.environ.get("OPENSHELL_FALLBACK_REGISTRY") or "quay.io/rhoai").rstrip("/")
        fallback_namespace = fallback_registry.rsplit("/", 1)[-1]
        path = "/".join(image.split("/")[1:])
        if path.startswith(f"{fallback_namespace}/"):
            path = path.split("/", 1)[1]
        fallback = f"{fallback_registry}/{path}"
        if self._podman_image_exists(image):
            return image
        if image != fallback and self._podman_image_exists(fallback):
            return fallback
        if not self._podman_pull(image):
            if image == fallback:
                raise KitError(f"unable to pull OpenShell CLI image {image}.")
            log.warn(f"OpenShell CLI: {image} is not pullable here; using {fallback}...")
            if not self._podman_pull(fallback):
                raise KitError(f"unable to pull OpenShell CLI image {fallback}.")
            return fallback
        return image

    def _podman_image_exists(self, image: str) -> bool:
        return log.run(["podman", "image", "exists", image], check=False, capture=True).returncode == 0

    def _podman_pull(self, image: str) -> bool:
        return log.run(["podman", "pull", image], check=False, capture=True).returncode == 0

    def _container_prefix(self) -> list[str]:
        config = self.config
        config.cli_state.mkdir(parents=True, exist_ok=True)
        return [
            "podman",
            "run",
            "--rm",
            "-i",
            *shlex.split(config.podman_userns),
            "-e",
            "XDG_CONFIG_HOME=/cli",
            "-e",
            "HOME=/cli",
            "-v",
            f"{config.cli_state}:/cli/openshell",
            "-v",
            f"{config.root}:/work:ro",
            self._cli_image(),
        ]

    def command(self, args: list[str], *, with_gateway: bool = True) -> list[str]:
        """Build the full argv for the OpenShell CLI."""
        if not self.config.uses_container_cli:
            prefix = shlex.split(self.config.openshell_cli)
        else:
            prefix = self._container_prefix()
        if with_gateway:
            prefix = prefix + ["-g", self.config.gateway_name]
        return prefix + args

    def call(
        self,
        args: list[str],
        *,
        with_gateway: bool = True,
        check: bool = True,
        capture: bool = False,
        stdin_text: str | None = None,
        timeout: float | None = None,
    ):
        """Run the CLI, defaulting to ``-g <gateway>``."""
        return log.run(
            self.command(args, with_gateway=with_gateway),
            check=check,
            capture=capture,
            stdin_text=stdin_text,
            timeout=timeout,
        )

    def connected(self) -> bool:
        """Return True when ``openshell status`` reports a connection."""
        if log.dry_run():
            self.call(["status"], check=False, capture=True)
            return True
        result = self.call(["status"], check=False, capture=True)
        return "Connected" in (result.stdout or "")


def prepare(config: KitConfig) -> Context:
    """Load ``agent_config.json`` and resolve every secret-backed setting."""
    config_path = config.root / "agent_config.json"
    if not config_path.is_file():
        raise KitError("agent_config.json not found.")
    raw = config_path.read_bytes()
    agent_config = json.loads(raw.decode("utf-8"))
    provider = agent_config["vector_store"]["provider_type"]
    env = cluster.load_secrets(config)
    env["AGENT_CONFIG_B64"] = base64.b64encode(raw).decode("ascii")
    if provider == "pgvector":
        if not env.get("PGVECTOR_HOST"):
            raise KitError("PGVECTOR_HOST is required for pgvector deployment.")
        env["PGVECTOR_PORT"] = env.get("PGVECTOR_PORT") or "5432"
        env["PGVECTOR_DB"] = env.get("PGVECTOR_DB") or "postgres"
        env["PGVECTOR_USER"] = env.get("PGVECTOR_USER") or "postgres"
    elif provider == "neo4j":
        if not env.get("NEO4J_URI"):
            raise KitError("NEO4J_URI is required for Neo4j deployment.")
        if not env.get("NEO4J_PASSWORD"):
            raise KitError("NEO4J_PASSWORD is required for Neo4j deployment.")
        env["NEO4J_USERNAME"] = env.get("NEO4J_USERNAME") or "neo4j"
        env["NEO4J_DATABASE"] = env.get("NEO4J_DATABASE") or "neo4j"
    return Context(config=config, env=env, agent_config_b64=env["AGENT_CONFIG_B64"], provider=provider)


def _sleep(seconds: float) -> None:
    if not log.dry_run():
        time.sleep(seconds)


def _sandbox_resource(config: KitConfig) -> str:
    return f"{config.workspace}--{config.sandbox_name}"


def _wait_gone(config: KitConfig, _cli: OpenShell) -> None:
    if log.dry_run():
        log.run(["oc", "-n", config.openshell_namespace, "get", "sandbox", _sandbox_resource(config)], check=False)
        log.run(["oc", "-n", config.namespace, "get", "pod", config.sandbox_name], check=False)
        return
    for _ in range(30):
        sandbox_present = (
            log.run(
                ["oc", "-n", config.openshell_namespace, "get", "sandbox", _sandbox_resource(config)],
                check=False,
                capture=True,
            ).returncode
            == 0
        )
        pod_present = (
            log.run(
                ["oc", "-n", config.namespace, "get", "pod", config.sandbox_name],
                check=False,
                capture=True,
            ).returncode
            == 0
        )
        if not sandbox_present and not pod_present:
            return
        _sleep(2)
    return


def _create_args(ctx: Context) -> list[str]:
    config = ctx.config
    args = ["sandbox", "create", "--name", config.sandbox_name, "--from", config.sandbox_image]
    for key in _ENV_KEYS:
        value = ctx.get(key)
        if not value:
            continue
        if "\n" in value or "\r" in value:
            raise KitError(f"{key} contains newline characters; pass it base64-encoded instead.")
        args += ["--env", f"{key}={value}"]
    args += ["--env", "PORT=8080"]
    return args


def _run_create(ctx: Context, cli: OpenShell) -> tuple[int, str]:
    result = cli.call(_create_args(ctx), check=False, capture=True, timeout=80)
    combined = f"{result.stdout or ''}{result.stderr or ''}"
    return result.returncode, combined


def create(config: KitConfig, cli: OpenShell, ctx: Context) -> None:
    """Recreate the sandbox, retrying when Kyverno rejects the driver Pod update."""
    if not cli.connected():
        raise KitError(
            f"ERROR: OpenShell gateway '{config.gateway_name}' is not available from this machine.\n"
            "       Run 'make refresh-certs' to fetch the mTLS client certificates and register\n"
            "       the cluster gateway, or ask an administrator to run scripts/setup-openshell-gateway.sh."
        )
    cluster.grant_image_pullers(config)
    log.step(f"Deleting old sandbox {config.sandbox_name}...")
    cli.call(["sandbox", "delete", config.sandbox_name], check=False, capture=True)
    log.step("Waiting for the previous sandbox to be removed...")
    _wait_gone(config, cli)
    log.step(f"Creating sandbox {config.sandbox_name} from {config.sandbox_image}...")
    log.detail("(timeout 120s — this may take a minute while the image is pulled)...")
    attempt = 1
    rc = 0
    output = ""
    while True:
        log.detail(f"Attempt {attempt}/{config.create_attempts}...")
        rc, output = _run_create(ctx, cli)
        if rc == 0:
            break
        if "pod updates may not change fields" not in output:
            break
        if attempt >= config.create_attempts:
            break
        log.detail("Sandbox Pod update was rejected by the cluster Kyverno imagePullSecrets policy")
        log.detail("(duplicate pull secret). Recreating the sandbox and retrying...")
        cli.call(["sandbox", "delete", config.sandbox_name], check=False, capture=True)
        _wait_gone(config, cli)
        _sleep(5)
        attempt += 1
    if rc != 0:
        log.info(output)
        log.detail(f"sandbox create exited with {rc} (may still be starting — wait-sandbox will poll)")
    else:
        log.detail("Sandbox created.")


def wait(config: KitConfig, cli: OpenShell, _ctx: Context) -> None:
    """Poll the sandbox list until the sandbox reports ``Ready``."""
    log.step(f"Waiting for sandbox {config.sandbox_name} to be ready...")
    if log.dry_run():
        cli.call(["sandbox", "list"], check=False, capture=True)
        return
    timeout_seconds = WAIT_MAX_ATTEMPTS * WAIT_POLL_INTERVAL_SECONDS
    for attempt in range(1, WAIT_MAX_ATTEMPTS + 1):
        result = cli.call(["sandbox", "list"], check=False, capture=True)
        phase = ""
        for line in (result.stdout or "").splitlines():
            if config.sandbox_name in line:
                fields = _ANSI_RE.sub("", line).split()
                if fields:
                    phase = fields[-1]
        if phase == "Ready":
            log.detail("Sandbox ready.")
            return
        log.detail(f"status: {phase} (attempt {attempt}/{WAIT_MAX_ATTEMPTS}, waiting {WAIT_POLL_INTERVAL_SECONDS}s...)")
        _sleep(WAIT_POLL_INTERVAL_SECONDS)
    raise KitError(f"sandbox not ready after {timeout_seconds}s")


def _endpoint(uri: str, default_port: int) -> tuple[str, str]:
    parsed = urlsplit(uri if "//" in uri else f"//{uri}")
    return (parsed.hostname or "", str(parsed.port or default_port))


def _db_endpoint(ctx: Context) -> tuple[str, str]:
    if ctx.provider == "pgvector":
        return ctx.get("PGVECTOR_HOST", "localhost"), ctx.get("PGVECTOR_PORT", "5432")
    if ctx.provider == "milvus":
        return _endpoint(ctx.get("MILVUS_URI"), 443)
    if ctx.provider == "neo4j":
        return _endpoint(ctx.get("NEO4J_URI"), 7687)
    if log.dry_run():
        return "<vector-db-host>", "443"
    raise KitError(f"Unsupported deployment provider {ctx.provider}")


def egress(config: KitConfig, cli: OpenShell, ctx: Context) -> None:
    """Allow the agent to reach MaaS, the K8s API, and the vector database."""
    log.step(f"Using agent Python binary: {PYTHON_BIN}")
    log.step("Adding egress policy for MaaS...")
    maas_host = urlsplit(ctx.get("MAAS_BASE_URL")).hostname or ctx.get("MAAS_BASE_URL")
    cli.call(
        [
            "policy",
            "update",
            config.sandbox_name,
            "--add-endpoint",
            f"{maas_host}:443",
            "--binary",
            PYTHON_BIN,
            "--wait",
        ]
    )
    log.step(f"Adding egress policies for K8s API + {ctx.provider}...")
    db_host, db_port = _db_endpoint(ctx)
    if not db_host or not db_port:
        if not log.dry_run():
            raise KitError("Vector database host and port are required.")
        db_host, db_port = "<vector-db-host>", "443"
    policy_path = config.root / _POLICY_FILE
    api_url = cluster.api_server_url()
    api = urlsplit(api_url)
    result = cli.call(["policy", "get", config.sandbox_name, "--full", "-o", "json"], check=False, capture=True)
    raw = result.stdout or ""
    begin = raw.find("{")
    policy = json.loads(raw[begin:])["policy"] if begin >= 0 else {}
    network = policy.setdefault("network_policies", {})
    network["k8s_api"] = {
        "name": "k8s-api",
        "endpoints": [{"host": api.hostname, "port": api.port or 443, "tls": "skip"}],
        "binaries": [{"path": PYTHON_BIN}],
    }
    network["vector_db"] = {
        "name": "vector-db",
        "endpoints": [{"host": db_host, "port": int(db_port), "tls": "skip"}],
        "binaries": [{"path": PYTHON_BIN}],
    }
    if not log.dry_run():
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    policy_arg = _POLICY_MOUNTED if config.uses_container_cli else str(policy_path)
    try:
        cli.call(["policy", "set", config.sandbox_name, "--policy", policy_arg, "--wait"])
    finally:
        if policy_path.exists() and not log.dry_run():
            policy_path.unlink()
    log.info("Done!")


def _k8s_ca(config: KitConfig, api_url: str) -> tuple[str, str]:
    if log.dry_run():
        log.run(
            ["oc", "get", "configmap", "trusted-ca-bundle", "-n", "openshift-config-managed", "-o", "json"], check=False
        )
        return "dry-run", ""
    candidates = {
        "trusted-ca-bundle": cluster.config_map_value("openshift-config-managed", "trusted-ca-bundle", "ca-bundle.crt"),
        "kube-root-ca": cluster.config_map_value(config.namespace, "kube-root-ca.crt", "ca.crt"),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        for source, content in candidates.items():
            if not content.strip():
                continue
            path = Path(tmpdir) / f"{source}.crt"
            path.write_text(content, encoding="utf-8")
            check = log.run(
                ["curl", "-sS", "--max-time", "10", "--cacert", str(path), "-o", "/dev/null", f"{api_url}/healthz"],
                check=False,
                capture=True,
            )
            if check.returncode == 0:
                return source, content
        combined = candidates["trusted-ca-bundle"] + candidates["kube-root-ca"]
        if combined.strip():
            return f"combined bundles, none validated {api_url}", combined
    raise KitError("Kubernetes API CA certificate not found.")


def _decode_cert(value: str) -> str:
    if not value or "-----BEGIN CERTIFICATE-----" in value:
        return value
    try:
        return base64.b64decode(value).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return value


def _agent_script(config: KitConfig, ctx: Context, api_url: str, reviewer_token: str, ca_path: str) -> str:
    env = {
        "MAAS_API_KEY": ctx.get("MAAS_API_KEY"),
        "MAAS_BASE_URL": ctx.get("MAAS_BASE_URL"),
        "AGENT_CONFIG_B64": ctx.agent_config_b64,
        "K8S_API_URL": api_url,
        "K8S_REVIEWER_TOKEN": reviewer_token,
        "K8S_CA_PATH": ca_path,
        "ALLOWED_SA_USERNAME": f"system:serviceaccount:{config.namespace}:{config.agent_sa_name}",
        "OPENSHELL_GRPC_PROXY_PORT": "18080",
        "PYTHONPATH": SANDBOX_PYTHONPATH,
    }
    if ctx.provider == "milvus":
        env.update(
            {
                "MILVUS_URI": ctx.get("MILVUS_URI"),
                "MILVUS_TOKEN": ctx.get("MILVUS_TOKEN"),
                "MILVUS_CA_CERT": _decode_cert(ctx.get("MILVUS_CA_CERT")),
            }
        )
    elif ctx.provider == "pgvector":
        env.update(
            {
                "PGVECTOR_HOST": ctx.get("PGVECTOR_HOST"),
                "PGVECTOR_PORT": ctx.get("PGVECTOR_PORT"),
                "PGVECTOR_DB": ctx.get("PGVECTOR_DB"),
                "PGVECTOR_USER": ctx.get("PGVECTOR_USER"),
                "PGVECTOR_PASSWORD": ctx.get("PGVECTOR_PASSWORD"),
                "PGVECTOR_CA_CERT": _decode_cert(ctx.get("PGVECTOR_CA_CERT")),
            }
        )
    elif ctx.provider == "neo4j":
        env.update(
            {
                "NEO4J_URI": ctx.get("NEO4J_URI"),
                "NEO4J_USERNAME": ctx.get("NEO4J_USERNAME"),
                "NEO4J_PASSWORD": ctx.get("NEO4J_PASSWORD"),
                "NEO4J_DATABASE": ctx.get("NEO4J_DATABASE"),
                "NEO4J_CA_CERT": _decode_cert(ctx.get("NEO4J_CA_CERT")),
            }
        )
    lines = ["set -e"]
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(value or '')}")
    lines += [
        "cd /sandbox",
        ": > /sandbox/agent.log",
        "rm -f /sandbox/agent.pid",
        f"/usr/bin/python3.12 -c {shlex.quote(_LAUNCH_SCRIPT)}",
    ]
    return "\n".join(lines) + "\n"


def start(config: KitConfig, cli: OpenShell, ctx: Context) -> None:
    """Start the agent inside the sandbox with ServiceAccount auth enabled."""
    log.step(f"Creating {config.agent_sa_name} ServiceAccount and token...")
    cluster.ensure_service_account(config.namespace, config.agent_sa_name)
    client_token = cluster.create_token(config.namespace, config.agent_sa_name)
    cluster.upsert_token_secret(config, client_token)
    log.step("Generating sandbox reviewer token...")
    api_url = cluster.api_server_url()
    reviewer_token = cluster.create_token(config.openshell_namespace, config.sandbox_sa)
    ca_source, ca_content = _k8s_ca(config, api_url)
    log.step("Stopping any running agent...")
    cli.call(
        ["sandbox", "exec", "-n", config.sandbox_name, "--", PYTHON_BIN, "-c", _STOP_SCRIPT], check=False, capture=True
    )
    _sleep(5)
    log.step(f"Uploading the cluster API CA ({ca_source})...")
    cli.call(
        ["sandbox", "exec", "-n", config.sandbox_name, "--", "bash", "-c", "cat > /sandbox/k8s-ca.crt"],
        stdin_text=ca_content,
    )
    log.step("Starting agent (auth: enabled)...")
    script = _agent_script(config, ctx, api_url, reviewer_token, "/sandbox/k8s-ca.crt")
    result = cli.call(
        ["sandbox", "exec", "-n", config.sandbox_name, "--", "bash", "-s"], check=False, stdin_text=script
    )
    if result.returncode != 0:
        log.detail("Checking agent health after OpenShell exec transport error...")
        health = cli.call(
            ["sandbox", "exec", "-n", config.sandbox_name, "--", PYTHON_BIN, "-c", _HEALTH_SCRIPT],
            check=False,
            capture=True,
        )
        if health.returncode != 0:
            raise KitError("agent did not start; inspect the sandbox log at /sandbox/agent.log.")
    log.detail("Agent started in background.")


def expose(config: KitConfig, cli: OpenShell, _ctx: Context) -> None:
    """Expose the agent service and front it with a passthrough Route."""
    log.step("Exposing service URL...")
    cli.call(["service", "expose", config.sandbox_name, "8080", "agent"], check=False, capture=True)
    _sleep(2)
    domain = cluster.cluster_domain()
    host = f"{config.workspace}--{config.sandbox_name}--agent.{config.openshell_namespace}.{domain}"
    cluster.apply_route(config, host)
    log.step("Waiting for agent to initialize (10s)...")
    _sleep(10)
    log.step("Checking agent health...")
    health = log.run(["curl", "-sk", "-m", "3", f"https://{host}/health"], check=False, capture=True)
    body = (health.stdout or "").strip()
    log.info(body if body else "Agent not ready yet (check logs)")
    log.info("")
    log.info(f"Agent URL: https://{host}")
    log.info("")
    log.info("Get token:")
    log.info(
        f"  TOKEN=$(oc get secret {config.agent_token_secret} -n {config.namespace} "
        "-o jsonpath='{.data.token}' | base64 -d)"
    )
    log.info("")
    log.info("Query:")
    log.info(f"  curl -sk https://{host}/v1/responses \\")
    log.info('    -H "X-Api-Key: $TOKEN" \\')
    log.info("    -H 'Content-Type: application/json' \\")
    log.info('    -d \'{"input":[{"role":"user","content":"Hello"}]}\'')


def undeploy(config: KitConfig, cli: OpenShell) -> None:
    """Remove the sandbox and its Route."""
    cli.call(["sandbox", "delete", config.sandbox_name], check=False, capture=True)
    cluster.delete_route(config)
    log.info(f"Sandbox {config.sandbox_name} and route {config.route_name} removed.")
    _sleep(10)


def rotate_tokens(config: KitConfig, cli: OpenShell) -> None:
    """Rotate ServiceAccount tokens and restart the agent.

    Generates new client and reviewer tokens (24h duration), updates the
    stored secret, and restarts the agent with fresh credentials.
    """
    ctx = prepare(config)
    log.step("Rotating ServiceAccount tokens...")
    log.step(f"Creating new {config.agent_sa_name} client token...")
    client_token = cluster.create_token(config.namespace, config.agent_sa_name)
    cluster.upsert_token_secret(config, client_token)
    log.step("Generating new sandbox reviewer token...")
    reviewer_token = cluster.create_token(config.openshell_namespace, config.sandbox_sa)
    api_url = cluster.api_server_url()
    ca_source, ca_content = _k8s_ca(config, api_url)
    log.step("Stopping agent...")
    cli.call(
        ["sandbox", "exec", "-n", config.sandbox_name, "--", PYTHON_BIN, "-c", _STOP_SCRIPT], check=False, capture=True
    )
    _sleep(5)
    log.step(f"Uploading the cluster API CA ({ca_source})...")
    cli.call(
        ["sandbox", "exec", "-n", config.sandbox_name, "--", "bash", "-c", "cat > /sandbox/k8s-ca.crt"],
        stdin_text=ca_content,
    )
    log.step("Starting agent with new tokens...")
    script = _agent_script(config, ctx, api_url, reviewer_token, "/sandbox/k8s-ca.crt")
    cli.call(["sandbox", "exec", "-n", config.sandbox_name, "--", "bash", "-c", script])
    _sleep(2)
    pid_check = cli.call(
        ["sandbox", "exec", "-n", config.sandbox_name, "--", "bash", "-c", "[ -f /sandbox/agent.pid ]"],
        check=False,
        capture=True,
    )
    if pid_check.returncode == 0:
        log.detail("Agent process alive")
    log.info("Tokens rotated and agent restarted.")


def deploy(config: KitConfig, cli: OpenShell) -> None:
    """Run the full deployment chain."""
    ctx = prepare(config)
    cluster.refresh_certs(config, cli)
    create(config, cli, ctx)
    wait(config, cli, ctx)
    egress(config, cli, ctx)
    start(config, cli, ctx)
    expose(config, cli, ctx)


def refresh_certs(config: KitConfig, cli: OpenShell) -> None:
    """Delegate certificate refresh and gateway registration."""
    cluster.refresh_certs(config, cli)
