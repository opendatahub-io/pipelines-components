from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def text_extraction(
    documents_descriptor: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Output[dsl.Artifact],
    error_tolerance: Optional[float] = None,
    max_extraction_workers: Optional[int] = None,
):
    """Text Extraction component.

    Reads the documents_descriptor JSON (from documents_discovery), fetches
    the listed documents from S3, and extracts text using the docling library.

    Args:
        documents_descriptor: Input artifact containing
            documents_descriptor.json with bucket, prefix, and documents list.
        extracted_text: Output artifact where the extracted text content will be stored.
        error_tolerance: Fraction of documents (0.0–1.0) allowed to fail without
            raising an error. None (the default) means zero tolerance — any failure
            raises immediately after all documents are processed. 0.1 means up to
            10 % of documents may fail. Exceeding the threshold raises RuntimeError
            with a summary of up to 10 failing documents.
        max_extraction_workers: Number of parallel worker processes used for text
            extraction. Each worker loads a full docling DocumentConverter into memory
            (ONNX models, layout detection, etc.), so this should be kept low to avoid
            out-of-memory issues. Defaults to 4. Set to None to use all available CPU
            cores. Set to 1 to disable parallelism.
    """
    import json
    import logging
    import os
    import sys
    import tempfile
    import time
    import traceback
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from multiprocessing.pool import AsyncResult
    from pathlib import Path

    import boto3
    import multiprocess as multiprocessing
    from botocore.exceptions import SSLError

    DOCUMENTS_DESCRIPTOR_FILENAME = "documents_descriptor.json"
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}
    DOWNLOAD_MAX_THREADS = 8

    descriptor_path = Path(documents_descriptor.path) / DOCUMENTS_DESCRIPTOR_FILENAME
    if not descriptor_path.exists():
        raise FileNotFoundError(f"documents_descriptor.json not found at {descriptor_path}")

    s3_creds = {k: os.environ.get(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]}
    for k, v in s3_creds.items():
        if v is None:
            raise ValueError(f"{k} environment variable not set. Check if kubernetes secret was configured properly.")
    s3_creds["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION")

    logger = logging.getLogger("Text Extraction component logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    output_dir = Path(extracted_text.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_s3_client(verify=True):
        """Create a new boto3 S3 client from the environment credentials.

        A fresh client is created on every call so it is safe to use from
        multiple threads without sharing state. Pass verify=False to skip
        TLS certificate verification (used as a fallback when an SSLError
        is encountered during download).
        """
        session = boto3.session.Session(
            aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
            region_name=s3_creds.get("AWS_DEFAULT_REGION"),
        )
        return session.client(
            service_name="s3",
            endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
            verify=verify,
        )

    def download_document(doc: dict, base_path: Path) -> Path:
        """Download a single document from S3 to a local path mirroring the S3 key structure.

        On an SSLError the download is retried once with certificate verification
        disabled. Any other exception propagates to the caller.

        Args:
            doc: Document descriptor dict containing at least a "key" field with
                the S3 object key.
            base_path: Local directory under which the file is saved, preserving
                the S3 key as a relative sub-path.

        Returns:
            Path to the downloaded local file.
        """
        raw_key = doc["key"]
        safe_key = raw_key.strip().lstrip("/")
        rel = Path(safe_key)
        if not safe_key or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe S3 key (path traversal): {raw_key!r}")
        local_path = (base_path / rel).resolve()
        base_resolved = base_path.resolve()
        if not local_path.is_relative_to(base_resolved):
            raise ValueError(f"Unsafe S3 key (escapes download directory): {raw_key!r}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _dl_t0 = time.perf_counter()
        logger.info("Downloading %s", raw_key)
        try:
            make_s3_client().download_file(bucket, raw_key, str(local_path))
        except SSLError:
            logger.warning("SSL error when downloading %s, retrying with verify=False", raw_key)
            make_s3_client(verify=False).download_file(bucket, raw_key, str(local_path))
        logger.info("Download finished %s (%.1fs)", raw_key, time.perf_counter() - _dl_t0)
        return local_path

    def _docling_artifacts_path() -> Optional[Path]:
        """Local Docling models root (contains docling-project--* dirs). Set via image ENV for offline use.

        Returns the path only when it exists and contains at least one model
        directory. If the path is missing or empty, returns None so that
        docling falls back to downloading models from HuggingFace.
        """
        raw = os.environ.get("DOCLING_ARTIFACTS_PATH")
        if not raw:
            logger.info("DOCLING_ARTIFACTS_PATH not set — models will be downloaded from HuggingFace.")
            return None
        p = Path(raw)
        if not p.is_dir() or not any(p.iterdir()):
            logger.warning(
                "DOCLING_ARTIFACTS_PATH=%s is set but the directory is missing or empty "
                "— falling back to HuggingFace model download.",
                raw,
            )
            return None
        logger.info("Using local Docling artifacts from %s", p)
        return p

    def _build_docling_format_options():
        """Shared pipeline options for main-process warmup and worker processes (spawn-safe: module-level)."""
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PaginatedPipelineOptions, ThreadedPdfPipelineOptions
        from docling.document_converter import (
            HTMLFormatOption,
            MarkdownFormatOption,
            PdfFormatOption,
            PowerpointFormatOption,
            WordFormatOption,
        )

        ap = _docling_artifacts_path()
        pdf_pipeline_options = ThreadedPdfPipelineOptions(
            artifacts_path=ap,
            do_ocr=False,
            do_table_structure=False,
            accelerator_options=AcceleratorOptions(device="cpu", num_threads=2),
        )
        paginated_pipeline_options = PaginatedPipelineOptions(
            artifacts_path=ap,
            generate_page_images=False,
            accelerator_options=AcceleratorOptions(device="cpu", num_threads=2),
        )
        return {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=paginated_pipeline_options),
            InputFormat.PPTX: PowerpointFormatOption(pipeline_options=paginated_pipeline_options),
            InputFormat.HTML: HTMLFormatOption(),
            InputFormat.MD: MarkdownFormatOption(),
        }

    def _text_extraction_pool_initializer() -> None:
        """Pool initializer (top-level for multiprocessing spawn pickling)."""
        import time

        os.environ["TQDM_DISABLE"] = "1"
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        if _docling_artifacts_path() is not None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from docling.document_converter import DocumentConverter

        worker_log = logging.getLogger("text_extraction_worker")
        worker_log.setLevel(logging.INFO)
        worker_log.propagate = False
        if not worker_log.handlers:
            worker_log.addHandler(logging.StreamHandler(sys.stdout))

        worker_pid = os.getpid()
        init_start_time = time.perf_counter()
        worker_log.debug("Worker pid=%s: loading DocumentConverter.", worker_pid)

        mod = sys.modules[__name__]
        mod._mp_worker_converter = DocumentConverter(format_options=_build_docling_format_options())
        worker_log.debug(
            "Worker pid=%s: DocumentConverter ready (%.1fs)",
            worker_pid,
            time.perf_counter() - init_start_time,
        )

    def worker_process_document(file_path_str: str, output_dir_str: str) -> tuple[bool, str | None]:
        """Convert a single document to Markdown and write it to the output directory.

        Plain-text (.txt) files are copied as-is without invoking docling.
        All other supported formats are converted via the DocumentConverter
        that was created by the multiprocessing pool initializer in this process.

        Args:
            file_path_str: Absolute path to the local input file.
            output_dir_str: Absolute path to the directory where the resulting
                Markdown file will be written (named <original_filename>.md).

        Returns:
            (True, None) on success; (False, error_message) on failure where
            error_message is either a full traceback string or a plain description.
        """
        worker_log = logging.getLogger("text_extraction_worker")
        start_time = time.perf_counter()
        try:
            input_file = Path(file_path_str)
            output_dir = Path(output_dir_str)
            output_file = output_dir / f"{input_file.name}.md"

            if input_file.suffix.lower() == ".txt":
                output_file.write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")
                return True, None

            converter = getattr(sys.modules[__name__], "_mp_worker_converter", None)
            if converter is None:
                error_message = (
                    f"Worker pid={os.getpid()} has no DocumentConverter. "
                    "Pool initializer did not run or failed before setting _mp_worker_converter. "
                )
                return False, error_message

            file_size_mib = input_file.stat().st_size / (1024 * 1024) if input_file.exists() else 0.0
            worker_log.info(
                "pid=%s docling convert start: %s (%.1f MiB on disk)", os.getpid(), input_file.name, file_size_mib
            )
            conversion_result = converter.convert(input_file)
            output_file.write_text(conversion_result.document.export_to_markdown(), encoding="utf-8")
            worker_log.info(
                "pid=%s docling convert done: %s -> %s (%.1fs)",
                os.getpid(),
                input_file.name,
                output_file.name,
                time.perf_counter() - start_time,
            )
            return True, None
        except Exception:
            error_traceback = traceback.format_exc()
            worker_log.error("Failed to process %s:\n%s", file_path_str, error_traceback)
            return False, error_traceback

    def download_and_submit(
        docs: list, download_path: Path, process_pool, out_dir: Path
    ) -> tuple[list[tuple[str, AsyncResult]], list[dict]]:
        """Download all documents from S3, then submit for extraction largest-first.

        Documents with unsupported extensions are filtered out before any
        downloads begin. Supported documents are downloaded concurrently via a
        thread pool. Once all downloads complete, the local files are sorted by
        size descending before being submitted to the process pool. This ensures
        the heaviest documents are picked up by workers first, avoiding the
        straggler problem where one slow document blocks completion while other
        workers sit idle.

        Args:
            docs: List of document descriptor dicts from the documents_descriptor JSON.
            download_path: Local temporary directory where downloaded files are stored.
            process_pool: Active multiprocessing Pool to submit extraction tasks to.
            out_dir: Directory where extracted Markdown files will be written.

        Returns:
            - List of (local_file_path_str, AsyncResult) pairs, one per successfully
              downloaded and submitted document, ordered largest-first.
            - List of download error dicts, each containing 'file' (S3 key) and
              'traceback' (full exception traceback string).
        """
        download_error_details = []
        downloaded_paths = []
        skipped_docs = [doc for doc in docs if Path(doc["key"]).suffix.lower() not in SUPPORTED_EXTENSIONS]
        supported = [doc for doc in docs if Path(doc["key"]).suffix.lower() in SUPPORTED_EXTENSIONS]
        if skipped_docs:
            skipped_keys = ", ".join(doc["key"] for doc in skipped_docs)
            logger.warning("Skipping %d document(s) with unsupported extensions: %s", len(skipped_docs), skipped_keys)

        with ThreadPoolExecutor(max_workers=DOWNLOAD_MAX_THREADS) as dl_pool:
            dl_futures = {dl_pool.submit(download_document, doc, download_path): doc for doc in supported}
            for dl_future in as_completed(dl_futures):
                doc = dl_futures[dl_future]
                key = doc.get("key", "?") if isinstance(doc, dict) else "?"
                try:
                    local_path = dl_future.result()
                except Exception as exc:
                    exception_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    logger.warning("Download failed for key=%s: %s", key, exc)
                    download_error_details.append({"file": key, "traceback": exception_traceback})
                    continue
                downloaded_paths.append(local_path)

        downloaded_paths.sort(key=lambda p: p.stat().st_size, reverse=True)
        extraction_tasks = [
            (str(local_path), process_pool.apply_async(worker_process_document, (str(local_path), str(out_dir))))
            for local_path in downloaded_paths
        ]
        return extraction_tasks, download_error_details

    def raise_if_threshold_exceeded(error_details: list, total_docs: int, tolerance: Optional[float]) -> None:
        """Check whether the error count exceeds the allowed tolerance.

        Called twice during the pipeline run: once after downloads complete
        (to abort before extraction starts if too many files failed to
        download) and once after all extraction tasks finish (to report
        the combined total).

        Args:
            error_details: List of error dicts accumulated so far, each with
                'file' and 'traceback' keys.
            total_docs: Total number of documents in the original input, used
                to compute the failure percentage.
            tolerance: Fraction of total_docs (0.0–1.0) that may fail without
                raising. None means zero tolerance — any error raises.

        Raises:
            RuntimeError: When the number of errors exceeds the allowed count.
        """
        n_errors = len(error_details)
        if n_errors == 0:
            return
        allowed = 0 if tolerance is None else int(tolerance * total_docs)
        if n_errors <= allowed:
            return
        tolerance_str = "0 (none allowed)" if tolerance is None else f"{tolerance:.0%} of {total_docs}"
        shown = error_details[:10]
        lines = [
            f"Text extraction failed: {n_errors}/{total_docs} document(s) failed (tolerance: {tolerance_str}).",
            f"Showing {len(shown)} of {n_errors} error(s):",
        ]
        for i, err in enumerate(shown, 1):
            tb_lines = err["traceback"].strip().splitlines()
            snippet = "\n    ".join(tb_lines[-5:])
            lines.append(f"\n  [{i}] {err['file']}\n    {snippet}")
        raise RuntimeError("\n".join(lines))

    with open(descriptor_path) as f:
        descriptor = json.load(f)
    bucket = descriptor["bucket"]
    documents = descriptor["documents"]

    if not documents:
        logger.info("No documents to process.")
        return

    documents = sorted(documents, key=lambda d: d.get("size_bytes", 0), reverse=True)

    if max_extraction_workers is not None:
        effective_workers = max(1, max_extraction_workers)
    else:
        effective_workers = min(max(1, (os.cpu_count() or 1) // 2), 8)
    logger.info(
        "Starting text extraction for %d documents. extraction_workers=%d, download_threads=%d.",
        len(documents),
        effective_workers,
        DOWNLOAD_MAX_THREADS,
    )

    if _docling_artifacts_path() is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    multiprocessing_context = multiprocessing.get_context("spawn")
    with (
        tempfile.TemporaryDirectory() as download_dir,
        multiprocessing_context.Pool(
            processes=effective_workers,
            initializer=_text_extraction_pool_initializer,
        ) as process_pool,
    ):
        download_start_time = time.perf_counter()
        extraction_tasks, download_error_details = download_and_submit(
            documents, Path(download_dir), process_pool, output_dir
        )
        logger.info(
            "Downloads finished in %.1fs; %d file(s) queued for extraction, %d download error(s).",
            time.perf_counter() - download_start_time,
            len(extraction_tasks),
            len(download_error_details),
        )
        raise_if_threshold_exceeded(download_error_details, len(documents), error_tolerance)

        extraction_error_details = []
        processed_count = 0
        pending = list(extraction_tasks)
        completed = 0
        while pending:
            still_pending = []
            for file_path, task in pending:
                if task.ready():
                    completed += 1
                    try:
                        success, tb = task.get()
                    except Exception:
                        tb = traceback.format_exc()
                        logger.error("Worker crashed for %s:\n%s", file_path, tb)
                        success = False
                    Path(file_path).unlink(missing_ok=True)
                    if success:
                        processed_count += 1
                    else:
                        extraction_error_details.append({"file": file_path, "traceback": tb})
                    logger.info("Extraction progress %d/%d", completed, len(extraction_tasks))
                else:
                    still_pending.append((file_path, task))
            pending = still_pending
            if pending:
                time.sleep(0.01)

    all_error_details = download_error_details + extraction_error_details
    total_errors = len(all_error_details)
    total_docs = len(documents)
    logger.info(
        "Text extraction completed. Total processed: %d/%d, Errors: %d",
        processed_count,
        total_docs,
        total_errors,
    )
    raise_if_threshold_exceeded(all_error_details, total_docs, error_tolerance)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        text_extraction,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
