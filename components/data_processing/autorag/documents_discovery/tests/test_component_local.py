"""Local runner tests for the documents_discovery component."""

import pytest

from ..component import documents_discovery


class TestDocumentsDiscoveryLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    @pytest.mark.skip(reason="Requires S3 credentials (AWS_* env); run E2E in pipeline with secrets")
    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution with LocalRunner."""
        result = documents_discovery(
            input_data_bucket_name="test-bucket",
            input_data_keys=["test-prefix/"],
        )
        assert result is not None
