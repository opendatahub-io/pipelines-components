import logging
import os

logger = logging.getLogger(__name__)


def _make_kfp_client(config):
    """Create a KFP client from a config dict; returns None if config is None."""
    if config is None:
        logger.info("Skipping KFP client creation due to missing config.")
        return None
    import kfp

    host = config["rhoai_kfp_url"]
    if not host.endswith("/"):
        host = host + "/"
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return kfp.Client(
        host=host,
        namespace=config["rhoai_project"],
        existing_token=config.get("rhoai_token"),
        verify_ssl=verify_ssl,
    )


def _make_s3_client(config):
    """Create a boto3 S3 client from a config dict; returns None if not configured."""
    if config is None or not config.get("s3_endpoint"):
        logger.info("Skipping S3 client creation due to missing config.")
        return None
    try:
        import boto3
    except ImportError:
        logger.info("Skipping S3 client creation due to missing 'boto3' package.")
        return None
    return boto3.client(
        "s3",
        endpoint_url=config["s3_endpoint"],
        aws_access_key_id=config["s3_access_key"],
        aws_secret_access_key=config["s3_secret_key"],
        region_name=config["s3_region"],
    )
