"""Shared errors for the AutoML data loaders."""


def s3_tls_validation_error(bucket: str, key: str, ssl_cert_path: str | None) -> RuntimeError:
    """Build the guided error for S3 TLS certificate validation failures."""
    certificate_source = ssl_cert_path or "the system trust bundle"
    return RuntimeError(
        "S3 TLS certificate validation failed when downloading "
        f"s3://{bucket}/{key}. Verify that ssl_cert_path="
        f"{certificate_source!r} is a PEM CA bundle that trusts the S3 endpoint. "
        "Do not disable certificate verification."
    )
