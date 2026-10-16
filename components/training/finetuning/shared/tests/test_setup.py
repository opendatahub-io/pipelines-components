"""Unit tests for the shared setup utilities."""

import logging
import os
from unittest import mock

import pytest

from ..setup import configure_env, create_logger, init_k8s, parse_kv, setup_hf_token


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


class TestCreateLogger:
    """Tests for create_logger."""

    def test_returns_logger_with_correct_name(self):
        """Logger is created with the specified name."""
        logger = create_logger("my_component")
        assert logger.name == "my_component"

    def test_default_name(self):
        """Default logger name is 'train_model'."""
        logger = create_logger()
        assert logger.name == "train_model"

    def test_logger_level_is_info(self):
        """Logger level is set to INFO."""
        logger = create_logger("test_level")
        assert logger.level == logging.INFO

    def test_logger_has_stdout_handler(self):
        """Logger has exactly one StreamHandler."""
        name = "test_handler_check"
        logging.getLogger(name).handlers.clear()
        logger = create_logger(name)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_no_duplicate_handlers_on_repeated_calls(self):
        """Repeated calls do not add duplicate handlers."""
        name = "test_no_dup"
        logging.getLogger(name).handlers.clear()
        create_logger(name)
        create_logger(name)
        assert len(logging.getLogger(name).handlers) == 1


class TestParseKv:
    """Tests for parse_kv."""

    def test_empty_string(self):
        """Empty string yields empty dict."""
        assert parse_kv("") == {}

    def test_single_pair(self):
        """Single key=value pair is parsed."""
        assert parse_kv("FOO=bar") == {"FOO": "bar"}

    def test_multiple_pairs(self):
        """Multiple comma-separated pairs are parsed."""
        assert parse_kv("A=1,B=2,C=3") == {"A": "1", "B": "2", "C": "3"}

    def test_strips_whitespace(self):
        """Whitespace around keys and values is stripped."""
        assert parse_kv("  A = 1 , B = 2 ") == {"A": "1", "B": "2"}

    def test_value_contains_equals(self):
        """Values containing '=' are preserved after the first split."""
        assert parse_kv("URL=http://host?a=b") == {"URL": "http://host?a=b"}

    def test_trailing_comma_ignored(self):
        """Trailing comma is ignored."""
        assert parse_kv("A=1,") == {"A": "1"}

    def test_missing_equals_raises(self):
        """Missing '=' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid kv"):
            parse_kv("NOEQUALS")

    def test_empty_key_raises(self):
        """Empty key raises ValueError."""
        with pytest.raises(ValueError, match="Empty key"):
            parse_kv("=value")

    def test_empty_value_allowed(self):
        """Empty value is allowed."""
        assert parse_kv("KEY=") == {"KEY": ""}


class TestConfigureEnv:
    """Tests for configure_env."""

    def test_merges_base_and_csv(self, log):
        """CSV pairs are merged with base dict."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = configure_env("X=1,Y=2", {"BASE": "val"}, log)
        assert result == {"BASE": "val", "X": "1", "Y": "2"}

    def test_csv_overrides_base(self, log):
        """CSV values override base dict values."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = configure_env("K=new", {"K": "old"}, log)
        assert result == {"K": "new"}

    def test_sets_os_environ(self, log):
        """Merged values are set in os.environ."""
        with mock.patch.dict(os.environ, {}, clear=True):
            configure_env("MY_VAR=hello", {}, log)
            assert os.environ["MY_VAR"] == "hello"

    def test_empty_csv(self, log):
        """Empty CSV returns only base dict values."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = configure_env("", {"ONLY": "base"}, log)
        assert result == {"ONLY": "base"}


class TestSetupHfToken:
    """Tests for setup_hf_token."""

    def test_propagates_existing_token(self, log):
        """Existing HF_TOKEN is propagated to menv."""
        menv = {}
        with mock.patch.dict(os.environ, {"HF_TOKEN": "tok123"}, clear=True):
            setup_hf_token(menv, "some/model", log)
        assert menv["HF_TOKEN"] == "tok123"

    def test_propagates_token_without_logging_value(self):
        """Token propagation logs status only, never the token value."""
        menv = {}
        log = mock.Mock()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "tok123"}, clear=True):
            setup_hf_token(menv, "some/model", log)

        assert "tok123" not in repr(log.mock_calls)
        log.info.assert_called_once_with("HF_TOKEN propagated")

    def test_no_token_warns_for_hf_model(self, log):
        """Warning is emitted when HF_TOKEN is missing for an HF model."""
        menv = {}
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(log, "warning") as mock_warn,
        ):
            setup_hf_token(menv, "meta-llama/Llama-2-7b", log)
        mock_warn.assert_called_once()
        assert "HF_TOKEN not set" in mock_warn.call_args[0][0]

    def test_no_token_warns_for_hf_prefix(self, log):
        """Warning is emitted for hf:// prefixed models without token."""
        menv = {}
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(log, "warning") as mock_warn,
        ):
            setup_hf_token(menv, "hf://some/model", log)
        mock_warn.assert_called_once()

    def test_no_token_no_warning_for_local_path(self, log, tmp_path):
        """No warning is emitted for local filesystem paths."""
        menv = {}
        local_dir = str(tmp_path)
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(log, "warning") as mock_warn,
        ):
            setup_hf_token(menv, local_dir, log)
        mock_warn.assert_not_called()

    def test_no_token_no_warning_for_oci_model(self, log):
        """No warning is emitted for OCI model references."""
        menv = {}
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(log, "warning") as mock_warn,
        ):
            setup_hf_token(menv, "oci://registry/model:tag", log)
        mock_warn.assert_not_called()
