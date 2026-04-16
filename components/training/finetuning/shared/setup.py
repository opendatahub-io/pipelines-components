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
    """Initialize Kubernetes client from environment variables.

    Args:
        log: Logger instance.

    Returns:
        Kubernetes ApiClient, or None if initialization fails.
    """
    try:
        from kubernetes import client as k8s

        srv = os.environ.get("KUBERNETES_SERVER_URL", "").strip()
        tok = os.environ.get("KUBERNETES_AUTH_TOKEN", "").strip()
        if not srv or not tok:
            raise RuntimeError(
                "Kubernetes credentials missing or incomplete: both KUBERNETES_SERVER_URL and "
                "KUBERNETES_AUTH_TOKEN must be set and non-empty. Ensure the 'kubernetes-credentials' "
                "secret is configured and mounted into the training task."
            )

        log.info("Initializing Kubernetes client from environment variables")
        _in_cluster_ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        cfg = k8s.Configuration()
        cfg.host = srv
        cfg.verify_ssl = True
        if not os.path.isfile(_in_cluster_ca):
            raise RuntimeError(
                f"In-cluster CA certificate not found at {_in_cluster_ca}. Ensure the service account token is mounted."
            )
        cfg.ssl_ca_cert = _in_cluster_ca
        cfg.api_key = {"authorization": f"Bearer {tok}"}
        k8s.Configuration.set_default(cfg)

        return k8s.ApiClient(cfg)
    except RuntimeError:
        raise
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
