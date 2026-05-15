"""Example pipelines demonstrating usage of documents_indexing."""

from kfp import dsl
from kfp_components.components.data_processing.autorag.documents_indexing import documents_indexing


@dsl.pipeline(name="documents-indexing-example")
def example_pipeline(
    embedding_model_id: str = "all-MiniLM-L6-v2",
    vector_io_provider_id: str = "milvus",
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
):
    """Example pipeline using documents_indexing.

    Args:
        embedding_model_id: ID of the embedding model.
        vector_io_provider_id: OGX provider ID for the vector database.
        distance_metric: Distance metric for similarity search.
        chunking_method: Method for text chunking.
        chunk_size: Size of each text chunk.
        chunk_overlap: Overlap between chunks.
        batch_size: Number of documents per batch.
    """
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    documents_indexing(
        embedding_model_id=embedding_model_id,
        extracted_text=extracted_text.output,
        vector_io_provider_id=vector_io_provider_id,
        distance_metric=distance_metric,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
    )
