"""Example pipelines demonstrating usage of search_space_preparation."""

from kfp import dsl
from kfp_components.components.training.autorag.search_space_preparation import search_space_preparation


@dsl.pipeline(name="search-space-preparation-example")
def example_pipeline():
    """Example pipeline using search_space_preparation."""
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    search_space_preparation(
        test_data=test_data.output,
        embedding_models=["ibm-granite/granite-embedding-278m-multilingual"],
        generation_models=["ibm-granite/granite-3.3-8b-instruct"],
    )
