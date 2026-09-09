"""Unit tests for the shared user-provided test dataset helpers."""

import os
from unittest import mock

from kfp_components.components.training.automl.shared.user_test_data import resolve_s3_env_credentials

TRAIN_ENV = {
    "AWS_ACCESS_KEY_ID": "train_key",
    "AWS_SECRET_ACCESS_KEY": "train_secret",
    "AWS_S3_ENDPOINT": "https://train-s3.example.local",
    "AWS_DEFAULT_REGION": "us-east-1",
}


class TestResolveS3EnvCredentials:
    """S3 credentials are read from the standard ``AWS_*`` environment variables."""

    @mock.patch.dict(os.environ, TRAIN_ENV, clear=True)
    def test_reads_aws_variables(self):
        """Training and test data both resolve credentials from ``AWS_*``."""
        assert resolve_s3_env_credentials() == {
            "access_key": "train_key",
            "secret_key": "train_secret",
            "endpoint_url": "https://train-s3.example.local",
            "region_name": "us-east-1",
        }
