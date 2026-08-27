"""Example pipelines demonstrating usage of rag_templates_optimization."""

from kfp import dsl
from kfp_components.components.training.autorag.rag_templates_optimization import (
    rag_templates_optimization,
)


@dsl.pipeline(name="rag-templates-optimization-example")
def example_pipeline(
    test_data_key: str = "questions",
    maas_secret_name: str = "maas-connection",
    vector_db_secret_name: str = "vector-db-connection",
    input_data_secret_name: str = "s3-input-connection",
    input_data_bucket_name: str = "my-bucket",
    input_data_key: str = "",
):
    """Example pipeline using rag_templates_optimization.

    Args:
        test_data_key: Key for the test data.
        maas_secret_name: Name of the K8s secret with MaaS inference credentials.
        vector_db_secret_name: Name of the K8s secret with the vector database
            configuration (MILVUS_* selects Milvus, PGVECTOR_* selects PGVector).
        input_data_secret_name: Name of the K8s secret with S3 credentials.
        input_data_bucket_name: S3 bucket containing input documents.
        input_data_key: Key for the input data.
    """
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    search_space_mps_report = dsl.importer(
        artifact_uri="gs://placeholder/search_space_mps_report",
        artifact_class=dsl.Artifact,
    )
    rag_templates_optimization(
        extracted_text=extracted_text.output,
        test_data=test_data.output,
        search_space_mps_report=search_space_mps_report.output,
        test_data_key=test_data_key,
        maas_secret_name=maas_secret_name,
        vector_db_secret_name=vector_db_secret_name,
        input_data_secret_name=input_data_secret_name,
        input_data_bucket_name=input_data_bucket_name,
        input_data_key=input_data_key,
    )
