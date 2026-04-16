"""Unit tests for the shared setup utilities (K8s SSL and configuration)."""

import logging
from unittest import mock

import pytest

from ..setup import init_k8s


@pytest.fixture
def log():
    """Create a test logger."""
    return logging.getLogger("test_setup")


class TestInitK8sSSL:
    """Tests for init_k8s SSL verification and error propagation."""

    def test_missing_credentials_raises_runtime_error(self, log):
        """RuntimeError must propagate when KUBERNETES_SERVER_URL is missing."""
        with mock.patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="Kubernetes credentials missing"):
                init_k8s(log)

    def test_missing_token_raises_runtime_error(self, log):
        """RuntimeError must propagate when KUBERNETES_AUTH_TOKEN is missing."""
        env = {"KUBERNETES_SERVER_URL": "https://api.example.com"}
        with mock.patch.dict("os.environ", env, clear=True):
            with pytest.raises(RuntimeError, match="Kubernetes credentials missing"):
                init_k8s(log)

    def test_missing_ca_cert_raises_runtime_error(self, log):
        """RuntimeError must propagate when in-cluster CA cert file is absent."""
        env = {
            "KUBERNETES_SERVER_URL": "https://api.example.com",
            "KUBERNETES_AUTH_TOKEN": "test-token",
        }
        with mock.patch.dict("os.environ", env, clear=True), mock.patch("os.path.isfile", return_value=False):
            with pytest.raises(RuntimeError, match="In-cluster CA certificate not found"):
                init_k8s(log)

    def test_ssl_enabled_with_ca_cert(self, log):
        """Verify verify_ssl=True and ssl_ca_cert is set when CA file exists."""
        env = {
            "KUBERNETES_SERVER_URL": "https://api.example.com",
            "KUBERNETES_AUTH_TOKEN": "test-token",
        }
        mock_cfg = mock.MagicMock()
        mock_k8s = mock.MagicMock()
        mock_k8s.Configuration.return_value = mock_cfg
        mock_kubernetes = mock.MagicMock()
        mock_kubernetes.client = mock_k8s

        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch("os.path.isfile", return_value=True),
            mock.patch.dict("sys.modules", {"kubernetes": mock_kubernetes, "kubernetes.client": mock_k8s}),
        ):
            result = init_k8s(log)

        assert mock_cfg.host == "https://api.example.com"
        assert mock_cfg.verify_ssl is True
        assert mock_cfg.ssl_ca_cert == "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        assert result is not None

    def test_import_error_returns_none(self, log):
        """Non-RuntimeError exceptions return None (e.g., missing kubernetes package)."""
        with mock.patch.dict("sys.modules", {"kubernetes": None, "kubernetes.client": None}):
            result = init_k8s(log)
        assert result is None
