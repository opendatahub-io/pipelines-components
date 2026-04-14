"""Setup utilities: logging, K8s initialization, environment config, HF token."""

import logging
import os
import sys
from typing import Dict, Optional


def create_logger(name: str = "train_model") -> logging.Logger:
    """Create and configure a logger for training components.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        lg.addHandler(h)
    return lg


def init_k8s(log: logging.Logger) -> Optional[object]:
    """Initialize a Kubernetes API client for Kubeflow Trainer / Training APIs.

    Pipeline steps that run **on-cluster** should use the pod's in-cluster
    ServiceAccount (default). That avoids ``401 Unauthorized`` from stale or
    incorrect tokens in a ``kubernetes-credentials`` secret.

    **Credential order** (unless ``KUBERNETES_PREFER_ENV_CREDENTIALS`` is truthy):

    1. In-cluster config (``/var/run/secrets/kubernetes.io/serviceaccount/...``)
    2. ``KUBERNETES_SERVER_URL`` + ``KUBERNETES_AUTH_TOKEN`` (e.g. local dev or
       cross-cluster)

    Set ``KUBERNETES_PREFER_ENV_CREDENTIALS=1`` to use env vars first when both
    are available.

    Args:
        log: Logger instance.

    Returns:
        Kubernetes ApiClient, or None if initialization fails.
    """
    try:
        from kubernetes import client as k8s
        from kubernetes import config as kube_config

        prefer_env = os.environ.get("KUBERNETES_PREFER_ENV_CREDENTIALS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        def _api_from_incluster() -> object:
            kube_config.load_incluster_config()
            log.info("Initializing Kubernetes client via in-cluster ServiceAccount")
            return k8s.ApiClient()

        def _api_from_env() -> Optional[object]:
            srv = os.environ.get("KUBERNETES_SERVER_URL", "").strip()
            tok = os.environ.get("KUBERNETES_AUTH_TOKEN", "").strip()
            if not srv or not tok:
                return None
            log.info("Initializing Kubernetes client from KUBERNETES_SERVER_URL / KUBERNETES_AUTH_TOKEN")
            cfg = k8s.Configuration()
            cfg.host, cfg.verify_ssl = srv, False
            cfg.api_key = {"authorization": f"Bearer {tok}"}
            k8s.Configuration.set_default(cfg)

            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            return k8s.ApiClient(cfg)

        if prefer_env:
            client = _api_from_env()
            if client is not None:
                return client
            log.warning("Env credentials missing or incomplete; falling back to in-cluster config")
            return _api_from_incluster()

        try:
            return _api_from_incluster()
        except Exception as ic_err:
            log.warning(f"In-cluster Kubernetes config unavailable ({ic_err}); trying env credentials")

        client = _api_from_env()
        if client is not None:
            return client

        raise RuntimeError(
            "Kubernetes client could not be configured: in-cluster config failed and "
            "KUBERNETES_SERVER_URL / KUBERNETES_AUTH_TOKEN are missing or incomplete."
        )
    except Exception as e:
        log.warning(f"K8s client init failed: {e}")
        return None


def parse_kv(s: str) -> Dict[str, str]:
    """Parse comma-separated key=value pairs.

    Args:
        s: String containing key=value pairs.

    Returns:
        Dictionary of parsed key-value pairs.
    """
    out = {}
    if not s:
        return out
    for it in s.split(","):
        it = it.strip()
        if not it:
            continue
        if "=" not in it:
            raise ValueError(f"Invalid kv: {it}")
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"Empty key: {it}")
        out[k] = v
    return out


def configure_env(csv: str, base: Dict[str, str], log: logging.Logger) -> Dict[str, str]:
    """Configure environment variables.

    Args:
        csv: Comma-separated key=value pairs.
        base: Base environment dictionary.
        log: Logger instance.

    Returns:
        Merged environment dictionary.
    """
    m = {**base, **parse_kv(csv)}
    for k, v in m.items():
        os.environ[k] = v
    log.info(f"Env: {sorted(m.keys())}")
    return m


def setup_hf_token(menv: Dict[str, str], training_base_model: str, log: logging.Logger) -> None:
    """Setup HuggingFace token if available.

    Args:
        menv: Environment dictionary to update.
        training_base_model: Base model path/ID.
        log: Logger instance.
    """
    hf_tok = os.environ.get("HF_TOKEN", "").strip()
    if hf_tok:
        menv["HF_TOKEN"] = hf_tok
        os.environ["HF_TOKEN"] = hf_tok
        log.info("HF_TOKEN propagated")
    elif isinstance(training_base_model, str):
        b = training_base_model.strip()
        if b.startswith("hf://") or ("/" in b and not b.startswith("oci://") and not os.path.exists(b)):
            log.warning(f"HF_TOKEN not set; only public models accessible for '{training_base_model}'")
