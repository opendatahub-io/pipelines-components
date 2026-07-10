from typing import Optional

from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.autorag.documents_discovery.component import documents_discovery
from kfp_components.components.data_processing.autorag.documents_indexing.component import documents_indexing
from kfp_components.components.data_processing.autorag.text_extraction.component import text_extraction

MAX_CPUS = "32"
MAX_MEMORY = "64Gi"

# Must match metadata.yaml ``name`` (and managed-pipelines catalog when registered).
PIPELINE_NAME = "documents-indexing-pipeline"


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "AutoRAG pipeline for building a production vector index from your documents. Powered by ai4rag, "
        "it discovers documents, extracts text, and indexes chunks into OGX using settings from an optimized "
        "RAG pattern. Delivers a named collection, indexing artifacts, and component status for Dashboard "
        "deploy."
    ),
)
def documents_indexing_pipeline(
    ogx_secret_name: str,
    embedding_model_id: str,
    vector_io_provider_id: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    input_data_key: Optional[str] = None,
    vector_store_id: str = None,
    embedding_params: Optional[dict] = None,
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
):
    """Build a production vector index from documents for AutoRAG.

    Discovers documents from object storage, extracts text, and indexes chunks into OGX.
    Intended for post-optimization deploy when applying an optimized RAG pattern's indexing
    settings to a production corpus.

    Args:
        ogx_secret_name: Name of the secret with OGX credentials
            ("OGX_CLIENT_BASE_URL", "OGX_CLIENT_API_KEY").
        embedding_model_id: Embedding model ID for the vector store.
        vector_io_provider_id: Optional OGX provider ID.
        input_data_secret_name: Name of the secret with S3 credentials for input data
            ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION").
        input_data_bucket_name: Name of the S3 bucket containing input data.
        input_data_key: Path to folder with input documents within bucket.
        vector_store_id: OGX vector store / collection id to reuse (aligned with
            ``pattern.json`` ``settings.vector_store_binding.vector_store_id``).
            Omit to create a new collection.
        embedding_params: Dict passed to OGXEmbeddingParams (default: {}).
        distance_metric: Vector distance metric (e.g. "cosine").
        chunking_method: Chunking method (e.g. "recursive").
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        batch_size: Number of documents per batch. Defaults to ``20``; ``0`` processes all
            documents in a single batch.
    """
    documents_discovery_task = documents_discovery(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_key,
    )
    documents_discovery_task.set_caching_options(False)
    documents_discovery_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )

    text_extraction_task = text_extraction(
        documents_descriptor=documents_discovery_task.outputs["discovered_documents"],
    )
    text_extraction_task.set_caching_options(False)
    text_extraction_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
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
        vector_store_id=vector_store_id,
    )
    documents_indexing_task.set_caching_options(False)
    documents_indexing_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
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
