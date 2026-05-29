from json import JSONDecodeError

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def test_data_loader(
    test_data_bucket_name: str,
    test_data_path: str,
    benchmark_sample_size: int = 25,
    test_data: dsl.Output[dsl.Artifact] = None,
):
    """Download test data JSON from S3 and sample it for benchmarking.

    The component reads S3-compatible credentials from environment variables
    (injected by the pipeline from a Kubernetes secret), downloads a JSON
    test data file, and randomly samples up to ``benchmark_sample_size``
    records to limit evaluation cost in downstream components.

    Args:
        test_data_bucket_name: S3 (or compatible) bucket that contains the test
            data file.
        test_data_path: S3 object key to the JSON test data file.
        benchmark_sample_size: Maximum number of records to keep from the test
            data. When the dataset exceeds this limit, a reproducible random
            sample is drawn (seed 42). Set to 0 to disable sampling and keep
            all records.
        test_data: Output artifact that receives the (possibly sampled) file.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.

    Raises:
        ValueError: If S3 credentials are missing or misconfigured.
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

    if not test_data_bucket_name:
        raise TypeError("test_data_bucket_name must be a non-empty string")

    benchmark_record_keys = {"question", "correct_answers", "correct_answer_document_ids"}

    class TestDataLoaderException(Exception):
        pass

    s3_creds = {k: os.environ.get(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]}
    missing_creds = [k for k, v in s3_creds.items() if v is None]

    if missing_creds:
        raise ValueError(
            f"Missing environment variable(s): {missing_creds}. Check if kubernetes secret was configured properly."
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

    logger.info("Fetching test data from S3: bucket='%s', path='%s'.", test_data_bucket_name, test_data_path)
    try:
        logger.info("Downloading test data...")
        test_data_response = s3_client.get_object(Bucket=test_data_bucket_name, Key=test_data_path)
        logger.info("Download completed successfully.")
    except SSLError:
        logger.warning("SSL error when downloading %s, retrying with verify=False.", test_data_path)
        s3_client = _make_s3_client(verify=False)
        test_data_response = s3_client.get_object(Bucket=test_data_bucket_name, Key=test_data_path)
        logger.info("Download completed successfully with verify=False.")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            raise FileNotFoundError(
                "Test data object not found in S3. bucket=%r, key=%r. "
                "Check that test_data_key (pipeline parameter) is the full object key to an existing JSON file."
                % (test_data_bucket_name, test_data_path)
            ) from e
        else:
            raise TestDataLoaderException(f"Failed to fetch {test_data_path}: {e}") from e
    except Exception as e:
        raise TestDataLoaderException(f"Failed to fetch {test_data_path}: {e}") from e

    test_data_raw = test_data_response["Body"].read().decode("utf-8")

    try:
        benchmark_data = json.loads(test_data_raw)
    except JSONDecodeError as e:
        raise TestDataLoaderException("test_data_path must point to a valid JSON file.") from e

    if not isinstance(benchmark_data, list):
        raise TestDataLoaderException("Test data file content must be a list with benchmark records.")

    for idx, benchmark_record in enumerate(benchmark_data):
        if not isinstance(benchmark_record, dict):
            raise TestDataLoaderException(
                f"Expected a dict at index {idx}, got {type(benchmark_record).__name__}: {benchmark_record!r}"
            )
        if set(benchmark_record.keys()) != benchmark_record_keys:
            raise TestDataLoaderException(
                f"Incorrect or incomplete keys in test data record. "
                f"Make sure that each test data records contains following keys: {benchmark_record_keys}."
            )

    if 0 < benchmark_sample_size < len(benchmark_data) and isinstance(benchmark_data, list):
        import random

        original_count = len(benchmark_data)
        rng = random.Random(42)
        data = rng.sample(benchmark_data, benchmark_sample_size)
        with open(test_data.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Sampled %d records from %d total.", benchmark_sample_size, original_count)
    else:
        with open(test_data.path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
        record_count = len(benchmark_data)
        logger.info("No sampling applied; record count: %s.", record_count)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        test_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
