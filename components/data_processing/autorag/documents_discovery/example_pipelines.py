"""Example pipelines demonstrating usage of documents_discovery."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.documents_discovery import documents_discovery


@dsl.pipeline(name="documents-discovery-example")
def example_pipeline(
    input_data_bucket_name: str = "my-bucket",
    input_data_keys: list[str] = ["documents/"],
    sampling_enabled: bool = True,
    sampling_max_size: float = 1,
):
    """Example pipeline using documents_discovery.

    Args:
        input_data_bucket_name: S3 bucket containing input documents.
        input_data_keys: Up to ten path prefixes within the bucket; all are merged
            into one deduplicated corpus.
        sampling_enabled: Whether to enable sampling.
        sampling_max_size: Maximum sample size in GB.
    """
    documents_discovery(
        input_data_bucket_name=input_data_bucket_name,
        input_data_keys=input_data_keys,
        sampling_enabled=sampling_enabled,
        sampling_max_size=sampling_max_size,
    )
