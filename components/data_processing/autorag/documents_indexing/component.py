from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def documents_indexing(
    embedding_model_id: str,
    extracted_text: dsl.Input[dsl.Artifact],
    vector_io_provider_id: str,
    embedding_params: Optional[dict] = None,
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
    vector_store_id: Optional[str] = None,
):
    """Index extracted text into a vector store with optional batch processing.

    Thin wrapper that delegates to ``ai4rag.components.data.indexing.index_documents``.

    Args:
        embedding_model_id: Embedding model ID used for the vector store.
        extracted_text: Input artifact (folder) containing DoclingDocument JSON
            files from text extraction.
        vector_io_provider_id: OGX provider ID for the vector database.
        embedding_params: Optional embedding parameters.
        distance_metric: Vector distance metric (e.g. "cosine").
        chunking_method: Chunking method.
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        batch_size: Number of documents per batch. Defaults to ``20``; ``0`` processes all
            documents in a single batch.
        vector_store_id: OGX vector store / collection id to reuse (matches
            ``pattern.json`` ``settings.vector_store_binding.vector_store_id``).
            Omit to create a new collection.
    """
    import logging
    import os

    from ai4rag.components.data.indexing import index_documents
    from ai4rag.components.utils.ogx_client import create_ogx_client

    logging.basicConfig(level=logging.INFO)

    ogx_client = create_ogx_client(
        base_url=os.environ["OGX_CLIENT_BASE_URL"],
        api_key=os.environ["OGX_CLIENT_API_KEY"],
    )

    index_documents(
        extracted_text_dir=extracted_text.path,
        embedding_model_id=embedding_model_id,
        vector_io_provider_id=vector_io_provider_id,
        ogx_client=ogx_client,
        embedding_params=embedding_params,
        distance_metric=distance_metric,
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
        collection_name=vector_store_id,
    )
