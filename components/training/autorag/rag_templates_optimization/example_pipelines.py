"""Example pipelines demonstrating usage of rag_templates_optimization."""

from kfp import dsl
from kfp_components.components.training.autorag.rag_templates_optimization import (
    rag_templates_optimization,
)


@dsl.pipeline(name="rag-templates-optimization-example")
def example_pipeline(
    test_data_key: str = "questions",
    vector_io_provider_id: str = "milvus",
    input_data_key: str = "",
):
    """Example pipeline using rag_templates_optimization.

    Args:
        test_data_key: Key for the test data.
        vector_io_provider_id: Vector I/O provider identifier.
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
    search_space_prep_report = dsl.importer(
        artifact_uri="gs://placeholder/search_space_prep_report",
        artifact_class=dsl.Artifact,
    )
    rag_templates_optimization(
        extracted_text=extracted_text.output,
        test_data=test_data.output,
        search_space_prep_report=search_space_prep_report.output,
        test_data_key=test_data_key,
        vector_io_provider_id=vector_io_provider_id,
        input_data_key=input_data_key,
    )
