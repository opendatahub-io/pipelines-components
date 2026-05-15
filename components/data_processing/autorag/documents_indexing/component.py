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
    collection_name: Optional[str] = None,
):
    """Index extracted text into a vector store with optional batch processing.

    Reads markdown files from extracted_text, chunks them, embeds via OGX,
    and adds them to the vector store. When batch_size > 0, processes documents
    in batches to limit memory use and allow progress on large inputs.

    Args:
        embedding_model_id: Embedding model ID used for the vector store.
        extracted_text: Input artifact (folder) containing .md files from text extraction.
        vector_io_provider_id: OGX provider ID for the vector database.
        embedding_params: Optional embedding parameters.
        distance_metric: Vector distance metric (e.g. "cosine").
        chunking_method: Chunking method.
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        batch_size: Number of documents per batch; 0 means process all in one batch.
        collection_name: Optional name of the collection to reuse; omit to create a new one.
    """
    import logging
    import os
    import ssl
    import sys
    from pathlib import Path

    import httpx
    from ai4rag.rag.chunking import LangChainChunker
    from ai4rag.rag.embedding.ogx import OGXEmbeddingModel, OGXEmbeddingParams
    from ai4rag.rag.vector_store.ogx import OGXVectorStore
    from langchain_core.documents import Document
    from ogx_client import APIConnectionError as OGXAPIConnectionError
    from ogx_client import OgxClient

    def _is_ssl_error(exc: BaseException) -> bool:
        """Check whether an exception (or its cause/context chain) is an SSL verification failure."""
        seen = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            msg = str(current).upper()
            if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg:
                return True
            current = current.__cause__ or current.__context__
        return False

    def _create_ogx_client(**kwargs) -> OgxClient:
        """Create OgxClient, falling back to SSL-unverified if self-signed cert detected."""
        client = OgxClient(**kwargs)
        try:
            client.models.list()
        except (ssl.SSLCertVerificationError, httpx.ConnectError, OGXAPIConnectionError) as exc:
            if _is_ssl_error(exc):
                logger.warning(
                    "SSL verification failed for OgxClient — retrying with verify=False. ",
                )
                client = OgxClient(
                    **kwargs,
                    http_client=httpx.Client(verify=False),
                )
            else:
                raise
        return client

    logger = logging.getLogger("Document Loader component logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    supported_distance_metrics = ("cosine", "euclidean")
    supported_chunking_methods = ("recursive",)
    supported_chunks_sizes_range = (128, 2048)

    if not vector_io_provider_id or not vector_io_provider_id.strip():
        raise ValueError("vector_io_provider_id must be a non-empty string.")

    if not embedding_model_id:
        raise ValueError("embedding_model_id must be a non-empty string.")

    if distance_metric not in supported_distance_metrics:
        raise ValueError(
            f"distance metric {distance_metric} is not supported, supported types are {supported_distance_metrics}."
        )

    if chunking_method not in supported_chunking_methods:
        raise ValueError(f"chunking_method is not supported, supported methods are {supported_chunking_methods}.")

    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")
    else:
        if not (supported_chunks_sizes_range[0] <= chunk_size <= supported_chunks_sizes_range[1]):
            raise ValueError(
                f"chunk_size must be an integer in the range"
                f" {supported_chunks_sizes_range[0]} to {supported_chunks_sizes_range[1]}."
            )

    if not isinstance(chunk_overlap, (int, float)):
        raise TypeError("chunk_overlap must be a numerical value.")

    if embedding_params is None:
        embedding_params = {}
    else:
        if not isinstance(embedding_params, dict):
            raise TypeError("embedding_params must be a dictionary.")

    params = OGXEmbeddingParams(**embedding_params)

    ogx_base_url = os.getenv("OGX_CLIENT_BASE_URL")
    ogx_api_key = os.getenv("OGX_CLIENT_API_KEY")
    missing = [
        name for name, val in (("OGX_CLIENT_BASE_URL", ogx_base_url), ("OGX_CLIENT_API_KEY", ogx_api_key)) if not val
    ]
    if missing:
        raise RuntimeError(f"Required environment variable(s) not set: {', '.join(missing)}")

    client = _create_ogx_client(
        base_url=ogx_base_url,
        api_key=ogx_api_key,
    )

    paths = sorted(Path(extracted_text.path).glob("*.md"))
    total_documents = len(paths)
    logger.info("Found %s documents to index", total_documents)

    if total_documents == 0:
        logger.warning("No documents found in %s", extracted_text.path)
        return

    chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding_model = OGXEmbeddingModel(client=client, model_id=embedding_model_id, params=params)

    collection_name_param = {"reuse_collection_name": collection_name} if collection_name is not None else {}
    ogx_vectorstore = OGXVectorStore(
        embedding_model=embedding_model,
        client=client,
        provider_id=vector_io_provider_id,
        distance_metric=distance_metric,
        **collection_name_param,
    )

    effective_batch_size = batch_size if batch_size > 0 else total_documents
    total_chunks = 0

    for start in range(0, total_documents, effective_batch_size):
        batch_paths = paths[start : start + effective_batch_size]
        batch_documents = [
            Document(
                page_content=p.read_text(encoding="utf-8", errors="replace"),
                metadata={"document_id": p.stem},
            )
            for p in batch_paths
        ]
        batch_chunks = chunker.split_documents(batch_documents)
        ogx_vectorstore.add_documents(batch_chunks)
        total_chunks += len(batch_chunks)
        batch_num = start // effective_batch_size + 1
        num_batches = (total_documents + effective_batch_size - 1) // effective_batch_size
        logger.info(
            "Batch %s/%s: indexed %s documents (%s chunks), total chunks so far: %s",
            batch_num,
            num_batches,
            len(batch_documents),
            len(batch_chunks),
            total_chunks,
        )

    logger.info(
        "Documents indexing finished: %s documents, %s chunks",
        total_documents,
        total_chunks,
    )
