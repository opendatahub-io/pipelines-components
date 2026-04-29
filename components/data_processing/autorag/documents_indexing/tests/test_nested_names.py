"""Unit tests for nested helper functions from documents_indexing component."""

import ssl
from unittest.mock import MagicMock, patch

import httpx
import pytest
from llama_stack_client import APIConnectionError as LSAPIConnectionError

from components.data_processing.autorag.documents_indexing.tests.nested_names import (
    _create_llama_stack_client,
    _is_ssl_error,
)


class TestIsSSLError:
    """Unit tests for _is_ssl_error helper function."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("CERTIFICATE_VERIFY_FAILED", True),
            ("certificate_verify_failed", True),  # case insensitive
            ("SSL error occurred", True),
            ("ssl verification", True),
            ("Connection timeout", False),
            ("Generic error", False),
            ("", False),
        ],
    )
    def test_ssl_error_detection_in_message(self, message, expected):
        """Test SSL error detection from exception message."""
        exc = Exception(message)
        assert _is_ssl_error(exc) == expected

    def test_ssl_error_in_cause_chain(self):
        """Test SSL error detection in __cause__ chain."""
        inner = Exception("CERTIFICATE_VERIFY_FAILED")
        outer = Exception("Connection failed")
        outer.__cause__ = inner
        assert _is_ssl_error(outer) is True

    def test_ssl_error_in_context_chain(self):
        """Test SSL error detection in __context__ chain."""
        inner = Exception("SSL verification failed")
        outer = Exception("Connection failed")
        outer.__context__ = inner
        assert _is_ssl_error(outer) is True

    def test_circular_reference_handling(self):
        """Test that circular references don't cause infinite loop."""
        exc1 = Exception("error1")
        exc2 = Exception("error2")
        exc1.__cause__ = exc2
        exc2.__cause__ = exc1
        assert _is_ssl_error(exc1) is False

    def test_empty_exception_chain(self):
        """Test exception with no cause or context."""
        exc = Exception("No chain")
        assert _is_ssl_error(exc) is False

    def test_non_ssl_exception_returns_false(self):
        """Test that non-SSL exceptions return False."""
        exc = ValueError("Invalid parameter")
        assert _is_ssl_error(exc) is False


class TestCreateLlamaStackClient:
    """Unit tests for _create_llama_stack_client helper function."""

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    def test_successful_connection(self, mock_client_class):
        """Test successful client creation without SSL issues."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == mock_client
        mock_client_class.assert_called_once_with(base_url="https://api.example.com")

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.httpx.Client")
    def test_ssl_error_fallback(self, mock_httpx_client, mock_client_class):
        """Test fallback to unverified client on SSL error."""
        failing_client = MagicMock()
        failing_client.models.list.side_effect = ssl.SSLCertVerificationError("SSL error")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_client_class.side_effect = [failing_client, success_client]

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == success_client
        assert mock_client_class.call_count == 2
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.httpx.Client")
    def test_ls_api_connection_error_wrapping_ssl_retries(self, mock_httpx_client, mock_client_class):
        """Test LSAPIConnectionError wrapping SSL error triggers retry."""
        ssl_err = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")
        mock_request = MagicMock(spec=httpx.Request)
        api_err = LSAPIConnectionError(message="Connection error", request=mock_request)
        api_err.__cause__ = ssl_err

        failing_client = MagicMock()
        failing_client.models.list.side_effect = api_err

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_client_class.side_effect = [failing_client, success_client]

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == success_client
        assert mock_client_class.call_count == 2
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.httpx.Client")
    def test_httpx_connect_error_with_ssl_message_triggers_retry(self, mock_httpx_client, mock_client_class):
        """Test httpx.ConnectError with SSL message triggers retry."""
        failing_client = MagicMock()
        failing_client.models.list.side_effect = httpx.ConnectError("SSL verification failed")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_client_class.side_effect = [failing_client, success_client]

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == success_client
        assert mock_client_class.call_count == 2
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    def test_non_ssl_error_is_reraised(self, mock_client_class):
        """Test non-SSL error is re-raised without retry."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = TimeoutError("Request timed out")
        mock_client_class.return_value = mock_client

        with pytest.raises(TimeoutError, match="Request timed out"):
            _create_llama_stack_client(base_url="https://api.example.com")

        assert mock_client_class.call_count == 1  # No retry

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    def test_ls_api_connection_error_with_non_ssl_cause_is_reraised(self, mock_client_class):
        """Test LSAPIConnectionError with non-SSL cause is re-raised."""
        timeout_err = TimeoutError("timeout")
        mock_request = MagicMock(spec=httpx.Request)
        api_err = LSAPIConnectionError(message="Connection error", request=mock_request)
        api_err.__cause__ = timeout_err

        mock_client = MagicMock()
        mock_client.models.list.side_effect = api_err
        mock_client_class.return_value = mock_client

        with pytest.raises(LSAPIConnectionError):
            _create_llama_stack_client(base_url="https://api.example.com")

        assert mock_client_class.call_count == 1  # No retry

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    def test_kwargs_propagation(self, mock_client_class):
        """Test that all kwargs are passed to client constructor."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        _create_llama_stack_client(base_url="https://api.example.com", api_key="test", timeout=30)

        mock_client_class.assert_called_once_with(base_url="https://api.example.com", api_key="test", timeout=30)

    @patch("components.data_processing.autorag.documents_indexing.tests.nested_names.LlamaStackClient")
    def test_client_is_returned_and_functional(self, mock_client_class):
        """Test returned client has expected interface."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert hasattr(result, "models")
        assert result == mock_client
