"""Unit tests for the shared user-provided test dataset helpers."""

import os
from unittest import mock

import pytest
from kfp_components.components.training.automl.shared.user_test_data import resolve_s3_env_credentials

TRAIN_ENV = {
    "AWS_ACCESS_KEY_ID": "train_key",
    "AWS_SECRET_ACCESS_KEY": "train_secret",
    "AWS_S3_ENDPOINT": "https://train-s3.example.local",
    "AWS_DEFAULT_REGION": "us-east-1",
}

TEST_ENV = {
    "TEST_DATA_AWS_ACCESS_KEY_ID": "test_key",
    "TEST_DATA_AWS_SECRET_ACCESS_KEY": "test_secret",
    "TEST_DATA_AWS_S3_ENDPOINT": "https://test-s3.example.local",
    "TEST_DATA_AWS_DEFAULT_REGION": "eu-west-1",
}


class TestResolveS3EnvCredentials:
    """The test-data credential scope must never blend the two identities."""

    @mock.patch.dict(os.environ, TRAIN_ENV, clear=True)
    def test_training_scope_reads_aws_variables(self):
        """Training access always uses the plain ``AWS_*`` variables."""
        assert resolve_s3_env_credentials() == {
            "access_key": "train_key",
            "secret_key": "train_secret",
            "endpoint_url": "https://train-s3.example.local",
            "region_name": "us-east-1",
        }

    @mock.patch.dict(os.environ, TRAIN_ENV, clear=True)
    def test_test_scope_falls_back_to_training_when_no_test_secret(self):
        """With no ``TEST_DATA_AWS_*`` injected, the training credentials are reused."""
        assert resolve_s3_env_credentials(for_test_data=True) == {
            "access_key": "train_key",
            "secret_key": "train_secret",
            "endpoint_url": "https://train-s3.example.local",
            "region_name": "us-east-1",
        }

    @mock.patch.dict(os.environ, {**TRAIN_ENV, **TEST_ENV}, clear=True)
    def test_test_scope_prefers_test_secret(self):
        """A fully populated test secret overrides every training value."""
        assert resolve_s3_env_credentials(for_test_data=True) == {
            "access_key": "test_key",
            "secret_key": "test_secret",
            "endpoint_url": "https://test-s3.example.local",
            "region_name": "eu-west-1",
        }

    @pytest.mark.parametrize(
        "partial",
        [
            {"TEST_DATA_AWS_ACCESS_KEY_ID": "test_key"},
            {"TEST_DATA_AWS_SECRET_ACCESS_KEY": "test_secret"},
        ],
    )
    def test_partial_test_key_pair_raises(self, partial):
        """Half a test secret must fail loudly, not borrow the other half from training.

        Per-variable fallback would pair a test access key with the training secret key.
        Both slots end up non-empty, so ``validate_s3_env_credentials`` cannot see it and
        S3 fails much later with an opaque ``SignatureDoesNotMatch``.
        """
        with mock.patch.dict(os.environ, {**TRAIN_ENV, **partial}, clear=True):
            with pytest.raises(ValueError, match="must both be set, or both be unset"):
                resolve_s3_env_credentials(for_test_data=True)

    @mock.patch.dict(
        os.environ,
        {
            **TRAIN_ENV,
            "TEST_DATA_AWS_ACCESS_KEY_ID": "test_key",
            "TEST_DATA_AWS_SECRET_ACCESS_KEY": "test_secret",
        },
        clear=True,
    )
    def test_test_key_pair_without_endpoint_reuses_training_endpoint(self):
        """Separate credentials against a shared endpoint is a real configuration."""
        assert resolve_s3_env_credentials(for_test_data=True) == {
            "access_key": "test_key",
            "secret_key": "test_secret",
            "endpoint_url": "https://train-s3.example.local",
            "region_name": "us-east-1",
        }
