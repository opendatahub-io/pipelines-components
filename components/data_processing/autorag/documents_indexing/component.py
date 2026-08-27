from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_TEMPLATE_PATH = Path(__file__).parent / "indexing_report_template.html"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_TEMPLATE_PATH),
    install_kfp_package=False,
)
def documents_indexing(
    embedding_model_id: str,
    extracted_text: dsl.Input[dsl.Artifact],
    indexing_report: dsl.Output[dsl.Artifact],
    indexing_report_html: dsl.Output[dsl.HTML],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    embedding_params: Optional[dict] = None,
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
    collection_name: Optional[str] = None,
):
    """Chunk, embed, and index extracted documents into a vector store.

    Reads DoclingDocument JSON files from the *extracted_text* artifact,
    splits them into chunks, computes embeddings via MaaS, and inserts the
    resulting vectors into the configured vector store.  Documents are
    processed in batches to bound memory consumption.

    The vector store backend (Milvus or PGVector) is resolved at runtime from
    the environment injected by the vector-database secret: ``MILVUS_*`` keys
    select Milvus, ``PGVECTOR_*`` keys select PGVector.

    Individual document failures (corrupt JSON, chunking errors) are
    recorded in the indexing report and skipped — they do not abort the
    pipeline.  Systemic failures (MaaS API unreachable, vector database
    unreachable, embedding model errors) propagate normally.

    Args:
        embedding_model_id: Embedding model ID served by MaaS.
        extracted_text: Input artifact (directory) containing DoclingDocument
            JSON files from text extraction.
        indexing_report: Output artifact containing ``indexing_report.json``
            with per-document indexing status and pipeline settings.
        indexing_report_html: Output HTML artifact containing a styled rendering of
            the indexing results (summary stats, settings, per-document table).
        embedded_artifact: Embedded HTML report template injected by KFP
            at runtime from ``indexing_report_template.html``.
        embedding_params: Optional parameters forwarded to
            :class:`OpenAIEmbeddingParams` (e.g. ``embedding_dimension``,
            ``context_length``).
        chunking_method: Chunking strategy: ``"recursive"`` (LangChain) or
            ``"hybrid"`` (Docling structure-aware).
        chunk_size: Maximum chunk size in tokens (128--2048).
        chunk_overlap: Token overlap between consecutive chunks (recursive
            method only).
        batch_size: Number of documents loaded and processed per batch.
            Controls peak memory usage, not API payload sizes. Defaults to
            ``20``; ``0`` processes all documents in a single batch.
        collection_name: Vector store collection to reuse (matches
            ``pattern.json`` ``settings.vector_store_binding.collection_name``).
            Omit to create a new collection.

    Environment variables (required):
        MAAS_BASE_URL, MAAS_API_KEY for inference. Plus the vector database
        configuration injected from the vector-database secret: ``MILVUS_*``
        keys (at least ``MILVUS_URI``) select Milvus, ``PGVECTOR_*`` keys
        select PGVector.
    """
    import html as html_mod
    import json
    import logging
    import os
    from dataclasses import asdict
    from pathlib import Path

    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()

    from ai4rag.components.utils import create_maas_client
    from ai4rag.rag.chunking import DoclingChunker, LangChainChunker
    from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel, OpenAIEmbeddingParams
    from ai4rag.rag.vector_store import get_vector_store, get_vector_store_config
    from ai4rag.utils.constants import ChunkingConstraints
    from docling_core.types.doc.document import DoclingDocument

    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger(__name__)

    # --- Validate inputs before making any API calls ---

    if not embedding_model_id or not embedding_model_id.strip():
        raise ValueError("embedding_model_id must be a non-empty string.")

    if chunking_method not in ChunkingConstraints.METHODS:
        raise ValueError(
            f"chunking_method {chunking_method!r} is not supported, "
            f"supported methods are {ChunkingConstraints.METHODS}."
        )

    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")

    lo, hi = ChunkingConstraints.MIN_CHUNK_SIZE, ChunkingConstraints.MAX_CHUNK_SIZE
    if not lo <= chunk_size <= hi:
        raise ValueError(f"chunk_size must be an integer in the range {lo} to {hi}.")

    if not isinstance(chunk_overlap, int):
        raise TypeError("chunk_overlap must be an integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size < 0:
        raise ValueError("batch_size must be a non-negative integer.")

    maas_client = create_maas_client(
        base_url=os.environ["MAAS_BASE_URL"],
        api_key=os.environ["MAAS_API_KEY"],
    )

    if any(k.startswith("MILVUS") for k in os.environ):
        provider = "milvus"
    elif any(k.startswith("PGVECTOR") for k in os.environ):
        provider = "pgvector"
    else:
        raise ValueError(
            "No vector database configuration found. Expected MILVUS_* or PGVECTOR_* "
            "environment variables injected from the vector-database secret."
        )

    vector_store_config = get_vector_store_config(provider)
    _logger.info("Detected %s database provider from secret.", provider)

    params = OpenAIEmbeddingParams(**(embedding_params or {}))

    base = Path(extracted_text.path)
    paths = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    total_documents = len(paths)
    _logger.info("Found %d documents to index", total_documents)

    def write_report(total_documents, total_chunks, entries, settings):
        completed = sum(1 for e in entries if e.get("status") == "completed")
        failed = sum(1 for e in entries if e.get("status") == "failed")
        report = {
            "total_documents": total_documents,
            "completed": completed,
            "failed": failed,
            "total_chunks": total_chunks,
            "settings": settings,
            "documents": entries,
        }
        Path(indexing_report.path).parent.mkdir(parents=True, exist_ok=True)
        with open(indexing_report.path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        indexing_report.metadata["display_name"] = "Documents Indexing Report"
        indexing_report.metadata["total_documents"] = total_documents
        indexing_report.metadata["completed"] = completed
        indexing_report.metadata["failed"] = failed
        indexing_report.metadata["total_chunks"] = total_chunks

    def write_html(total_documents, total_chunks, completed, failed, entries, settings):
        esc = html_mod.escape

        # --- Build documents table fragment ---
        if entries:
            rows = []
            for i, entry in enumerate(entries, 1):
                status = entry.get("status", "")
                if status == "completed":
                    badge = '<span class="badge badge-success">completed</span>'
                else:
                    badge = '<span class="badge badge-danger">failed</span>'
                chunks = entry.get("chunks", "—")
                rows.append(
                    f"<tr><td>{i}</td><td>{esc(entry.get('file', ''))}</td><td>{badge}</td><td>{chunks}</td></tr>"
                )
            table_body = "\n".join(rows)
            table_html = (
                '<div class="card">\n'
                "  <h2>Documents</h2>\n"
                '  <div class="table-wrap">\n'
                "    <table>\n"
                "      <thead>\n"
                "        <tr>\n"
                "          <th>#</th><th>File</th><th>Status</th>\n"
                "          <th>Chunks</th>\n"
                "        </tr>\n"
                "      </thead>\n"
                f"      <tbody>{table_body}</tbody>\n"
                "    </table>\n"
                "  </div>\n"
                "</div>"
            )
        else:
            table_html = (
                '<div class="card">\n'
                "  <h2>Documents</h2>\n"
                '  <p class="empty-msg">No documents were found for indexing.</p>\n'
                "</div>"
            )

        # --- Build settings grid fragment ---
        vsb = settings.get("vector_store_binding", {})
        chk = settings.get("chunking", {})
        emb = settings.get("embedding", {})
        emb_params = emb.get("embedding_params", {})
        if emb_params:
            emb_params_html = ", ".join(f"<strong>{esc(str(k))}</strong>: {esc(str(v))}" for k, v in emb_params.items())
        else:
            emb_params_html = '<span class="muted">none</span>'

        collection = vsb.get("collection_name")
        collection_display = esc(str(collection)) if collection is not None else '<span class="muted">N/A</span>'

        settings_html = (
            '<div class="settings-group">\n'
            "  <h3>Vector Store</h3>\n"
            "  <dl>\n"
            "    <dt>Provider</dt>\n"
            f"    <dd>{esc(str(vsb.get('provider_type', '')))}</dd>\n"
            "    <dt>Collection Name</dt>\n"
            f"    <dd>{collection_display}</dd>\n"
            "  </dl>\n"
            "</div>\n"
            '<div class="settings-group">\n'
            "  <h3>Chunking</h3>\n"
            "  <dl>\n"
            "    <dt>Method</dt>\n"
            f"    <dd>{esc(str(chk.get('method', '')))}</dd>\n"
            "    <dt>Chunk Size</dt>\n"
            f"    <dd>{chk.get('chunk_size', '')}</dd>\n"
            "    <dt>Chunk Overlap</dt>\n"
            f"    <dd>{chk.get('chunk_overlap', '')}</dd>\n"
            "  </dl>\n"
            "</div>\n"
            '<div class="settings-group">\n'
            "  <h3>Embedding</h3>\n"
            "  <dl>\n"
            "    <dt>Model ID</dt>\n"
            f"    <dd>{esc(str(emb.get('model_id', '')))}</dd>\n"
            "    <dt>Parameters</dt>\n"
            f"    <dd>{emb_params_html}</dd>\n"
            "  </dl>\n"
            "</div>"
        )

        # --- Load template and substitute placeholders ---
        if embedded_artifact is None:
            raise ValueError(
                "embedded_artifact was not injected by the pipeline runtime; "
                "ensure the component is compiled with embedded_artifact_path "
                "pointing to a valid HTML template"
            )
        _embedded_path = Path(embedded_artifact.path)
        _template_file = (
            _embedded_path if _embedded_path.is_file() else _embedded_path / "indexing_report_template.html"
        )
        with open(_template_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        html_content = (
            html_content.replace("__TOTAL_DOCUMENTS__", str(total_documents))
            .replace("__COMPLETED__", str(completed))
            .replace("__FAILED__", str(failed))
            .replace("__FAILED_CLASS__", " danger" if failed else "")
            .replace("__TOTAL_CHUNKS__", str(total_chunks))
            .replace("__SETTINGS_HTML__", settings_html)
            .replace("__TABLE_HTML__", table_html)
        )

        Path(indexing_report_html.path).parent.mkdir(parents=True, exist_ok=True)
        with open(indexing_report_html.path, "w", encoding="utf-8") as f:
            f.write(html_content)
        indexing_report_html.metadata["display_name"] = "Documents Indexing Report"

    report_entries = []

    if total_documents == 0:
        _logger.warning("No documents found in %s", extracted_text.path)
        settings = {
            "vector_store_binding": {
                "provider_type": provider,
                "collection_name": collection_name,
            },
            "chunking": {
                "method": chunking_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "embedding": {
                "model_id": embedding_model_id,
                "embedding_params": embedding_params or {},
            },
        }
        write_report(total_documents=0, total_chunks=0, entries=report_entries, settings=settings)
        write_html(
            total_documents=0,
            total_chunks=0,
            completed=0,
            failed=0,
            entries=report_entries,
            settings=settings,
        )
        return

    if chunking_method == "hybrid":
        chunker = DoclingChunker(max_tokens=chunk_size)
    else:
        chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embedding_model = OpenAIEmbeddingModel(client=maas_client, model_id=embedding_model_id, params=params)

    effective_batch_size = batch_size if batch_size > 0 else total_documents
    total_chunks = 0
    num_batches = (total_documents + effective_batch_size - 1) // effective_batch_size

    with get_vector_store(
        embedding_model=embedding_model,
        config=vector_store_config,
        collection_name=collection_name,
    ) as vector_store:
        settings = {
            "vector_store_binding": {
                "provider_type": provider,
                "collection_name": vector_store.collection_name,
            },
            "chunking": {
                "method": chunking_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "embedding": {
                "model_id": embedding_model_id,
                "embedding_params": asdict(embedding_model.params),
            },
        }

        for start in range(0, total_documents, effective_batch_size):
            batch_paths = paths[start : start + effective_batch_size]
            batch_chunks = []

            for p in batch_paths:
                try:
                    doc = DoclingDocument.load_from_json(p)
                    chunks = chunker.split_documents([doc])
                    batch_chunks.extend(chunks)
                    report_entries.append({"file": p.name, "status": "completed", "chunks": len(chunks)})
                except Exception as exc:
                    _logger.warning("Skipping %s: %s", p.name, exc)
                    report_entries.append({"file": p.name, "status": "failed", "error": str(exc)})

            if batch_chunks:
                vector_store.add_documents(batch_chunks)
            total_chunks += len(batch_chunks)

            batch_num = start // effective_batch_size + 1
            _logger.info(
                "Batch %d/%d: indexed %d documents (%d chunks), total chunks so far: %d",
                batch_num,
                num_batches,
                len(batch_paths),
                len(batch_chunks),
                total_chunks,
            )

    completed_count = sum(1 for e in report_entries if e["status"] == "completed")
    failed_count = sum(1 for e in report_entries if e["status"] == "failed")
    _logger.info(
        "Documents indexing finished: %d documents, %d chunks, %d completed, %d failed",
        total_documents,
        total_chunks,
        completed_count,
        failed_count,
    )

    write_report(
        total_documents=total_documents,
        total_chunks=total_chunks,
        entries=report_entries,
        settings=settings,
    )
    write_html(
        total_documents=total_documents,
        total_chunks=total_chunks,
        completed=completed_count,
        failed=failed_count,
        entries=report_entries,
        settings=settings,
    )
