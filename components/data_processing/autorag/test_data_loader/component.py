from json import JSONDecodeError

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def test_data_loader(
    test_data_bucket_name: str,
    test_data_path: str,
    test_data: dsl.Output[dsl.Artifact] = None,
    data_source: str = "s3",
    pvc_data_path: str = "",
):
    """Download test data json file from S3 or PVC into a KFP artifact.

    **Data Source Configuration:**
    - When ``data_source="s3"``: downloads JSON from S3 using AWS credentials
    - When ``data_source="pvc"``: loads JSON from PVC workspace filesystem

    Args:
        test_data_bucket_name: S3 bucket containing the test data file (required when data_source="s3").
        test_data_path: S3 object key to the JSON test data file (required when data_source="s3").
        test_data: Output artifact that receives the downloaded file.
        data_source: Data source type ("s3" or "pvc"). Default is "s3".
        pvc_data_path: Path to JSON file on PVC (required when data_source="pvc").
            Can be absolute or relative to current directory.

    Environment variables (required when data_source="s3"):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.

    Raises:
        ValueError: If data_source is invalid, S3 credentials are missing, or required parameters are missing.
        FileNotFoundError: If PVC file not found when data_source="pvc".
        Exception: If the download fails or the path is not a JSON file.
    """
    import json
    import logging
    import os
    import sys

    import boto3
    from botocore.exceptions import ClientError, SSLError

    logger = logging.getLogger("Test Data Loader component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    # Data source validation
    VALID_DATA_SOURCES = {"s3", "pvc"}
    if data_source not in VALID_DATA_SOURCES:
        raise ValueError(f"data_source must be one of {VALID_DATA_SOURCES}; got {data_source!r}.")

    if data_source == "pvc":
        if not pvc_data_path or not pvc_data_path.strip():
            raise ValueError("pvc_data_path must be provided when data_source='pvc'")
    elif data_source == "s3":
        if not test_data_bucket_name:
            raise TypeError("test_data_bucket_name must be a non-empty string when data_source='s3'")

    def get_test_data_s3():
        """Validate S3 credentials and download the JSON test data file."""

        class TestDataLoaderException(Exception):
            pass

        s3_creds = {k: os.environ.get(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]}
        for k, v in s3_creds.items():
            if v is None:
                raise ValueError(
                    "%s environment variable not set. Check if kubernetes secret was configured properly" % k
                )
        s3_creds["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION")

        def _make_s3_client(verify=True):
            return boto3.client(
                "s3",
                endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
                region_name=s3_creds["AWS_DEFAULT_REGION"],
                aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
                verify=verify,
            )

        s3_client = _make_s3_client()

        logger.info(f"Fetching test data from S3: bucket={test_data_bucket_name}, path={test_data_path}")
        try:
            logger.info(f"Starting download to {test_data.path}")
            s3_client.download_file(test_data_bucket_name, test_data_path, test_data.path)
            logger.info("Download completed successfully")
        except SSLError:
            logger.warning(
                "SSL error when downloading %s, retrying with verify=False",
                test_data_path,
            )
            s3_client = _make_s3_client(verify=False)
            s3_client.download_file(test_data_bucket_name, test_data_path, test_data.path)
            logger.info("Download completed successfully with verify=False")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                raise FileNotFoundError(
                    "Test data object not found in S3. bucket=%r, key=%r. "
                    "Check that test_data_key (pipeline parameter) is the full object key to an existing JSON file."
                    % (test_data_bucket_name, test_data_path)
                ) from e
            else:
                raise TestDataLoaderException("Failed to fetch %s: %s", test_data_path, e) from e
        except Exception as e:
            raise TestDataLoaderException("Failed to fetch %s: %s", test_data_path, e) from e

        try:
            with open(test_data.path, "r") as f:
                json.load(f)
        except JSONDecodeError as e:
            raise TestDataLoaderException("test_data_path must point to a valid JSON file.") from e

    def _load_test_data_pvc(pvc_path, output_path):
        """Load test data from PVC and validate JSON format."""
        import shutil

        if not os.path.exists(pvc_path):
            raise FileNotFoundError(f"PVC test data file not found: {pvc_path}")

        # Validate JSON format
        with open(pvc_path, "r") as f:
            try:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} test questions from PVC")
            except JSONDecodeError as e:
                raise ValueError(f"PVC file {pvc_path} is not valid JSON") from e

        # Copy to output artifact
        shutil.copy2(pvc_path, output_path)
        logger.info(f"Copied test data from PVC: {pvc_path} -> {output_path}")

    # Load data from S3 or PVC based on data_source
    if data_source == "s3":
        get_test_data_s3()
    elif data_source == "pvc":
        _load_test_data_pvc(pvc_data_path, test_data.path)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        test_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
