from typing import Optional

from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.autorag.documents_discovery.component import documents_discovery
from kfp_components.components.data_processing.autorag.documents_indexing.component import documents_indexing
from kfp_components.components.data_processing.autorag.text_extraction.component import text_extraction


@dsl.pipeline(
    name="AutoRAG Documents Indexing Pipeline",
    description="Pipeline to load test data, discover and extract documents, then index them into a vector store.",
)
def documents_indexing_pipeline(
    ogx_secret_name: str,
    embedding_model_id: str,
    vector_io_provider_id: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    input_data_key: Optional[str] = None,
    collection_name: str = None,
    embedding_params: Optional[dict] = None,
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
):
    """Defines a pipeline to load, sample, extract text, and index documents for AutoRAG.

    Args:
        ogx_secret_name: Name of the secret with OGX credentials
            ("OGX_CLIENT_BASE_URL", "OGX_CLIENT_API_KEY").
        embedding_model_id: Embedding model ID for the vector store.
        vector_io_provider_id: Optional OGX provider ID.
        input_data_secret_name: Name of the secret with S3 credentials for input data
            ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION").
        input_data_bucket_name: Name of the S3 bucket containing input data.
        input_data_key: Path to folder with input documents within bucket.
        collection_name: Optional name of the collection to reuse; omit to create a new one.
        embedding_params: Dict passed to OGXEmbeddingParams (default: {}).
        distance_metric: Vector distance metric (e.g. "cosine").
        chunking_method: Chunking method (e.g. "recursive").
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        batch_size: Number of documents per batch (0 = process all at once).
    """
    documents_discovery_task = documents_discovery(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_key,
    )

    text_extraction_task = text_extraction(
        documents_descriptor=documents_discovery_task.outputs["discovered_documents"],
    )

    documents_indexing_task = documents_indexing(
        embedding_params=embedding_params,
        embedding_model_id=embedding_model_id,
        extracted_text=text_extraction_task.outputs["extracted_text"],
        vector_io_provider_id=vector_io_provider_id,
        distance_metric=distance_metric,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
        collection_name=collection_name,
    )

    def set_input_data_secrets(task, secret_name):
        use_secret_as_env(
            task,
            secret_name=secret_name,
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
                "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
                "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            },
        )

    set_input_data_secrets(documents_discovery_task, input_data_secret_name)
    set_input_data_secrets(text_extraction_task, input_data_secret_name)

    use_secret_as_env(
        documents_indexing_task,
        secret_name=ogx_secret_name,
        secret_key_to_env={
            "OGX_CLIENT_BASE_URL": "OGX_CLIENT_BASE_URL",
            "OGX_CLIENT_API_KEY": "OGX_CLIENT_API_KEY",
        },
    )


if __name__ == "__main__":
    import pathlib

    from kfp.compiler import Compiler

    output_path = pathlib.Path(__file__).with_name("documents_indexing_pipeline.yaml")
    Compiler().compile(pipeline_func=documents_indexing_pipeline, package_path=str(output_path))
    print(f"Pipeline compiled to {output_path}")
