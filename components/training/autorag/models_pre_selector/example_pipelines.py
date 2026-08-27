"""Example pipelines demonstrating usage of models_pre_selector."""

from kfp import dsl
from kfp_components.components.training.autorag.models_pre_selector import models_pre_selector


@dsl.pipeline(name="models-pre-selector-example")
def example_pipeline():
    """Example pipeline using models_pre_selector."""
    search_space_report = dsl.importer(
        artifact_uri="gs://placeholder/search_space_report",
        artifact_class=dsl.Artifact,
    )
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    models_pre_selector(
        search_space_report=search_space_report.output,
        extracted_text=extracted_text.output,
        test_data=test_data.output,
    )
