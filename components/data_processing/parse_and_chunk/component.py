"""KFP component: Parse PDFs with Docling and chunk with HybridChunker.

Submits a RayJob that processes PDFs using Docling's HybridChunker and
writes chunked text as JSONL files to an S3-compatible bucket (MinIO).
The PVC is mounted read-only (input PDFs only).
"""

from kfp import dsl
from kfp_components.utils.consts import RAY_RAG_BASE_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=RAY_RAG_BASE_IMAGE,
    packages_to_install=[
        "codeflare-sdk==0.35.0",
        "kubernetes>=28.1.0",
    ],
)
def parse_and_chunk(
    pvc_name: str,
    pvc_mount_path: str,
    input_path: str,
    ray_image: str,
    namespace: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_prefix: str = "chunks",
    s3_secret_name: str = "minio-secret",
    tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_max_tokens: int = 256,
    num_workers: int = 2,
    worker_cpus: int = 8,
    worker_memory_gb: int = 16,
    head_cpus: int = 2,
    head_memory_gb: int = 8,
    cpus_per_actor: int = 4,
    min_actors: int = 2,
    max_actors: int = 4,
    batch_size: int = 4,
    num_files: int = 0,
    timeout_seconds: int = 600,
    enable_profiling: bool = False,
    verbose: bool = True,
    bypass_kueue: bool = False,
) -> str:
    """Parse PDFs and write chunked JSONL files to S3.

    Args:
        pvc_name: Name of the PVC with input PDFs.
        pvc_mount_path: Mount path for the PVC inside pods.
        input_path: Relative path to input PDFs under the PVC.
        ray_image: Container image with Ray + Docling pre-installed.
        namespace: OpenShift namespace for the RayJob.
        s3_endpoint: S3-compatible endpoint URL (e.g. MinIO).
        s3_bucket: S3 bucket for output chunks.
        s3_prefix: Key prefix for chunk files in S3.
        s3_secret_name: Kubernetes Secret with S3 credentials (keys: access_key, secret_key).
        tokenizer: Tokenizer for HybridChunker (usually the embedding model name).
        chunk_max_tokens: Max tokens per chunk (HybridChunker).
        num_workers: Number of Ray worker pods.
        worker_cpus: CPUs per worker pod.
        worker_memory_gb: Memory (GB) per worker pod.
        head_cpus: CPUs for the Ray head pod.
        head_memory_gb: Memory (GB) for the Ray head pod.
        cpus_per_actor: CPUs per Docling processing actor.
        min_actors: Minimum actor pool size.
        max_actors: Maximum actor pool size.
        batch_size: Files per batch sent to each actor.
        num_files: Number of PDFs to process (0 = all).
        timeout_seconds: Per-file processing timeout.
        enable_profiling: Enable cProfile profiling (outputs profile stats).
        verbose: Enable verbose logging for debugging.
        bypass_kueue: If True, remove the Kueue queue-name label and manually
            unsuspend the RayJob, bypassing cluster quota management. Only use
            on clusters without Kueue or with sufficient free resources.

    Returns:
        The S3 URI where JSONL chunk files were written.
    """
    import base64
    import textwrap
    import time

    # Entrypoint script that runs on the Ray cluster.
    # It parses PDFs with Docling, chunks with HybridChunker,
    # and writes JSONL files to S3.
    _DOCLING_CHUNK_PROCESS_PY = textwrap.dedent('''\
        """Ray Data + Docling: parse PDFs and chunk with HybridChunker."""

        import glob
        import io
        import json
        import multiprocessing as mp
        import os
        import queue
        import time
        from typing import Dict, List

        import pandas as pd
        import ray

        MIN_ACTORS = int(os.environ.get("MIN_ACTORS", "2"))
        MAX_ACTORS = int(os.environ.get("MAX_ACTORS", "4"))
        CPUS_PER_ACTOR = int(os.environ.get("CPUS_PER_ACTOR", "4"))
        BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
        REPARTITION_FACTOR = int(os.environ.get("REPARTITION_FACTOR", "10"))

        PVC_MOUNT_PATH = os.environ.get("PVC_MOUNT_PATH", "/mnt/data")
        INPUT_PATH = os.environ.get("INPUT_PATH", "input/pdfs")
        NUM_FILES = int(os.environ.get("NUM_FILES", "0"))

        FILE_TIMEOUT = int(os.environ.get("FILE_TIMEOUT", "600"))
        MAX_ERRORED_BLOCKS = int(os.environ.get("MAX_ERRORED_BLOCKS", "50"))

        CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "256"))
        TOKENIZER = os.environ.get("TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")

        S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")
        S3_BUCKET = os.environ.get("S3_BUCKET", "rag-chunks")
        S3_PREFIX = os.environ.get("S3_PREFIX", "chunks")
        S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
        S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

        ENABLE_PROFILING = os.environ.get("ENABLE_PROFILING", "false").lower() == "true"
        VERBOSE = os.environ.get("VERBOSE", "true").lower() == "true"

        def log_verbose(msg):
            """Log message if VERBOSE is enabled."""
            if VERBOSE:
                print(f"[VERBOSE] {msg}", flush=True)


        def _get_s3_client():
            import boto3
            return boto3.client(
                "s3",
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
                region_name="us-east-1",
            )


        def _ensure_bucket(s3_client, bucket):
            try:
                s3_client.head_bucket(Bucket=bucket)
            except s3_client.exceptions.ClientError:
                s3_client.create_bucket(Bucket=bucket)
                print(f"Created S3 bucket '{bucket}'")


        def _converter_worker(
            req_q, res_q, cpus_per_actor, chunk_max_tokens, tokenizer_name,
            s3_endpoint, s3_bucket, s3_prefix, s3_access_key, s3_secret_key,
        ):
            import socket
            worker_host = socket.gethostname()
            worker_pid = os.getpid()

            log_verbose(f"[Worker {worker_pid}@{worker_host}] Initializing converter worker")
            log_verbose(f"[Worker {worker_pid}] CPUs per actor: {cpus_per_actor}")
            log_verbose(f"[Worker {worker_pid}] Chunk max tokens: {chunk_max_tokens}")
            log_verbose(f"[Worker {worker_pid}] Tokenizer: {tokenizer_name}")

            os.environ["OMP_NUM_THREADS"] = str(cpus_per_actor)
            os.environ["MKL_NUM_THREADS"] = str(cpus_per_actor)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            import boto3
            from docling.chunking import HybridChunker
            from docling.datamodel.base_models import DocumentStream, InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorOptions, PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            log_verbose(f"[Worker {worker_pid}] Creating Docling converter...")
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=cpus_per_actor, device="cpu",
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            log_verbose(f"[Worker {worker_pid}] Converter created")

            log_verbose(f"[Worker {worker_pid}] Creating HybridChunker...")
            chunker = HybridChunker(tokenizer=tokenizer_name, max_tokens=chunk_max_tokens)
            log_verbose(f"[Worker {worker_pid}] Chunker created")

            log_verbose(f"[Worker {worker_pid}] Connecting to S3: {s3_endpoint}")
            s3 = boto3.client(
                "s3", endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
                region_name="us-east-1",
            )
            log_verbose(f"[Worker {worker_pid}] S3 client connected")

            res_q.put(("ready",))
            log_verbose(f"[Worker {worker_pid}] Ready to process files")

            file_count = 0
            while True:
                msg = req_q.get()
                if msg is None:
                    log_verbose(f"[Worker {worker_pid}] Shutdown signal received, processed {file_count} files")
                    break
                file_path = msg
                file_count += 1
                fname = os.path.basename(file_path)

                try:
                    t_start = time.time()
                    log_verbose(f"[Worker {worker_pid}] Processing file {file_count}: {fname}")

                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    if len(file_bytes) == 0:
                        log_verbose(f"[Worker {worker_pid}] {fname}: EMPTY FILE")
                        res_q.put(("error", file_path, 0, 0, "File empty"))
                        continue

                    log_verbose(f"[Worker {worker_pid}] {fname}: Read {len(file_bytes)} bytes")
                    stem = os.path.splitext(fname)[0]
                    stream = DocumentStream(name=fname, stream=io.BytesIO(file_bytes))

                    t_convert_start = time.time()
                    result = converter.convert(stream)
                    t_convert = time.time() - t_convert_start
                    log_verbose(f"[Worker {worker_pid}] {fname}: Converted in {t_convert:.2f}s")

                    doc = result.document
                    pages = getattr(doc, "pages", None)
                    page_count = len(pages) if pages is not None else 0
                    log_verbose(f"[Worker {worker_pid}] {fname}: {page_count} pages")

                    t_chunk_start = time.time()
                    chunks = list(chunker.chunk(doc))
                    t_chunk = time.time() - t_chunk_start
                    log_verbose(f"[Worker {worker_pid}] {fname}: Generated {len(chunks)} chunks in {t_chunk:.2f}s")

                    if not chunks:
                        log_verbose(f"[Worker {worker_pid}] {fname}: NO CHUNKS GENERATED")
                        res_q.put(("empty", file_path, page_count, 0, "No chunks"))
                        continue

                    lines = []
                    for idx, chunk in enumerate(chunks):
                        if chunk.text.strip():
                            lines.append(json.dumps({
                                "source_file": fname,
                                "chunk_index": idx,
                                "text": chunk.text,
                            }))
                    if not lines:
                        log_verbose(f"[Worker {worker_pid}] {fname}: ALL CHUNKS EMPTY")
                        res_q.put(("empty", file_path, page_count, 0, "All chunks empty"))
                        continue

                    log_verbose(f"[Worker {worker_pid}] {fname}: {len(lines)} non-empty chunks")
                    jsonl_body = "\\n".join(lines) + "\\n"
                    s3_key = f"{s3_prefix}/{stem}.jsonl"

                    t_s3_start = time.time()
                    s3.put_object(
                        Bucket=s3_bucket, Key=s3_key,
                        Body=jsonl_body.encode("utf-8"),
                        ContentType="application/jsonl",
                    )
                    t_s3 = time.time() - t_s3_start

                    t_total = time.time() - t_start
                    log_verbose(
                        f"[Worker {worker_pid}] {fname}: Uploaded to "
                        f"s3://{s3_bucket}/{s3_key} in {t_s3:.2f}s (total: {t_total:.2f}s)"
                    )

                    res_q.put(("success", file_path, page_count, len(lines), ""))

                except Exception as e:
                    t_total = time.time() - t_start
                    error_msg = str(e)[:200]
                    log_verbose(f"[Worker {worker_pid}] {fname}: ERROR after {t_total:.2f}s: {error_msg}")
                    res_q.put(("error", file_path, 0, 0, error_msg))


        class DoclingChunkProcessor:
            def __init__(self):
                import socket
                self.hostname = socket.gethostname()
                self.batch_count = 0
                log_verbose(f"[Actor@{self.hostname}] Initializing DoclingChunkProcessor")
                self._start_worker()
                print(f"[{self.hostname}] DoclingChunkProcessor ready (pid={self._worker.pid})")

            def _start_worker(self):
                log_verbose(f"[Actor@{self.hostname}] Starting converter worker subprocess")
                self._req_q = mp.Queue()
                self._res_q = mp.Queue()
                self._worker = mp.Process(
                    target=_converter_worker,
                    args=(
                        self._req_q, self._res_q, CPUS_PER_ACTOR,
                        CHUNK_MAX_TOKENS, TOKENIZER, S3_ENDPOINT, S3_BUCKET,
                        S3_PREFIX, S3_ACCESS_KEY, S3_SECRET_KEY,
                    ),
                    daemon=True,
                )
                self._worker.start()
                log_verbose(f"[Actor@{self.hostname}] Waiting for worker to be ready...")
                msg = self._res_q.get(timeout=600)
                assert msg[0] == "ready"
                log_verbose(f"[Actor@{self.hostname}] Worker ready (pid={self._worker.pid})")

            def _restart_worker(self):
                log_verbose(f"[Actor@{self.hostname}] Restarting worker (previous pid={self._worker.pid})")
                if self._worker.is_alive():
                    self._worker.terminate()
                    self._worker.join(timeout=5)
                    if self._worker.is_alive():
                        self._worker.kill()
                        self._worker.join()
                # Drain the queue
                while True:
                    try:
                        self._res_q.get_nowait()
                    except queue.Empty:
                        break
                self._start_worker()
                print(f"[{self.hostname}] Restarted converter (new pid={self._worker.pid})")

            def __call__(self, batch):
                path_list = batch["path"]
                self.batch_count += 1
                batch_num = self.batch_count
                log_verbose(f"[Actor@{self.hostname}] Processing batch #{batch_num} with {len(path_list)} files")

                out = {k: [] for k in [
                    "source_file", "status", "page_count", "chunk_count",
                    "error", "duration_s", "actor_host",
                ]}

                for idx, file_path in enumerate(path_list, 1):
                    fname = os.path.basename(file_path)
                    log_verbose(f"[Actor@{self.hostname}] Batch #{batch_num}, file {idx}/{len(path_list)}: {fname}")

                    t0 = time.time()
                    self._req_q.put(str(file_path))
                    try:
                        result = self._res_q.get(timeout=FILE_TIMEOUT)
                        status, _, page_count, chunk_count, error_msg = result
                        elapsed = round(time.time() - t0, 3)
                        log_verbose(
                            f"[Actor@{self.hostname}] {fname}: {status} in {elapsed}s "
                            f"(pages={page_count}, chunks={chunk_count})"
                        )
                    except queue.Empty:
                        status, page_count, chunk_count = "timeout", 0, 0
                        error_msg = f"Timed out after {FILE_TIMEOUT}s"
                        elapsed = round(time.time() - t0, 3)
                        log_verbose(f"[Actor@{self.hostname}] {fname}: TIMEOUT after {elapsed}s, restarting worker")
                        self._restart_worker()

                    out["source_file"].append(fname)
                    out["status"].append(status)
                    out["page_count"].append(page_count)
                    out["chunk_count"].append(chunk_count)
                    out["error"].append(error_msg)
                    out["duration_s"].append(elapsed)
                    out["actor_host"].append(self.hostname)

                log_verbose(f"[Actor@{self.hostname}] Completed batch #{batch_num}")
                return out


        def run():
            print("=" * 80)
            print("DOCLING PDF PARSING & CHUNKING JOB")
            print("=" * 80)
            log_verbose(f"Configuration:")
            log_verbose(f"  PVC mount: {PVC_MOUNT_PATH}")
            log_verbose(f"  Input path: {INPUT_PATH}")
            log_verbose(f"  Full input path: {os.path.join(PVC_MOUNT_PATH, INPUT_PATH)}")
            log_verbose(f"  Number of files: {NUM_FILES if NUM_FILES > 0 else 'all'}")
            log_verbose(f"  Actors: {MIN_ACTORS} to {MAX_ACTORS}")
            log_verbose(f"  CPUs per actor: {CPUS_PER_ACTOR}")
            log_verbose(f"  Batch size: {BATCH_SIZE}")
            log_verbose(f"  Chunk max tokens: {CHUNK_MAX_TOKENS}")
            log_verbose(f"  Tokenizer: {TOKENIZER}")
            log_verbose(f"  File timeout: {FILE_TIMEOUT}s")
            log_verbose(f"  S3 endpoint: {S3_ENDPOINT}")
            log_verbose(f"  S3 bucket: {S3_BUCKET}")
            log_verbose(f"  S3 prefix: {S3_PREFIX}")
            log_verbose(f"  Verbose: {VERBOSE}")
            log_verbose(f"  Profiling: {ENABLE_PROFILING}")

            print(f"\\nScanning for PDFs...")
            input_full_path = os.path.join(PVC_MOUNT_PATH, INPUT_PATH)
            log_verbose(f"Globbing: {input_full_path}/**/*.pdf")
            pdf_paths = sorted(glob.glob(f"{input_full_path}/**/*.pdf", recursive=True))
            print(f"  Found {len(pdf_paths)} total PDFs")

            if NUM_FILES > 0:
                pdf_paths = pdf_paths[:NUM_FILES]
                print(f"  Limited to first {NUM_FILES} PDFs")

            print(f"  Processing {len(pdf_paths)} PDFs")

            if not pdf_paths:
                print("\\nERROR: No PDFs found. Exiting.")
                log_verbose("Checked paths:")
                log_verbose(f"  - {input_full_path}")
                log_verbose(f"  - {os.path.join(PVC_MOUNT_PATH, 'input/pdfs')}")
                return

            if VERBOSE and len(pdf_paths) <= 20:
                log_verbose("PDFs to process:")
                for i, path in enumerate(pdf_paths, 1):
                    log_verbose(f"  {i}. {path}")

            print(f"\\nSetting up S3...")
            s3 = _get_s3_client()
            _ensure_bucket(s3, S3_BUCKET)
            print(f"  Output: s3://{S3_BUCKET}/{S3_PREFIX}/")

            print(f"\\nPreparing Ray dataset...")
            target_blocks = MAX_ACTORS * REPARTITION_FACTOR
            log_verbose(f"Creating dataset from {len(pdf_paths)} paths")
            ds = ray.data.from_pandas(pd.DataFrame({"path": pdf_paths}))
            log_verbose(f"Repartitioning to {target_blocks} blocks ({MAX_ACTORS} actors × {REPARTITION_FACTOR})")
            ds = ds.repartition(target_blocks)
            print(f"  Repartitioned into {target_blocks} blocks for {MAX_ACTORS} max actors")

            print(f"\\nStarting document processing...")
            log_verbose(f"Creating actor pool strategy (min={MIN_ACTORS}, max={MAX_ACTORS})")
            results_ds = ds.map_batches(
                DoclingChunkProcessor,
                compute=ray.data.ActorPoolStrategy(
                    min_size=MIN_ACTORS, max_size=MAX_ACTORS,
                ),
                batch_size=BATCH_SIZE,
                batch_format="numpy",
                num_cpus=CPUS_PER_ACTOR,
            )

            print(f"  Actor pool: {MIN_ACTORS}-{MAX_ACTORS} actors")
            print(f"  Batch size: {BATCH_SIZE} files/batch")
            print(f"  Expected total batches: ~{len(pdf_paths) // BATCH_SIZE + 1}")
            print("")

            start_time = time.time()
            success_count = error_count = timeout_count = empty_count = 0
            total_pages = total_chunks = 0
            actor_dist = {}
            errors_list = []
            last_report_time = start_time
            report_interval = 30  # Report progress every 30 seconds

            batch_num = 0
            for batch in results_ds.iter_batches(batch_size=200, batch_format="numpy"):
                batch_num += 1
                n = len(batch["source_file"])
                log_verbose(f"Processing result batch {batch_num} with {n} files")

                for i in range(n):
                    status = str(batch["status"][i])
                    fname = str(batch["source_file"][i])
                    if status == "success":
                        success_count += 1
                    elif status == "timeout":
                        timeout_count += 1
                        errors_list.append((fname, str(batch["error"][i])))
                    elif status == "empty":
                        empty_count += 1
                    else:
                        error_count += 1
                        errors_list.append((fname, str(batch["error"][i])))
                    total_pages += int(batch["page_count"][i])
                    total_chunks += int(batch["chunk_count"][i])
                    actor = str(batch["actor_host"][i])
                    actor_dist[actor] = actor_dist.get(actor, 0) + 1

                # Progress reporting
                now = time.time()
                if now - last_report_time >= report_interval:
                    elapsed = now - start_time
                    total_processed = success_count + error_count + timeout_count + empty_count
                    pct = (total_processed / len(pdf_paths) * 100) if pdf_paths else 0
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    eta = (len(pdf_paths) - total_processed) / rate if rate > 0 else 0
                    print(f"  Progress: {total_processed}/{len(pdf_paths)} ({pct:.1f}%) | "
                          f"{rate:.1f} files/s | ETA: {eta/60:.1f}min")
                    last_report_time = now

            wall_clock = time.time() - start_time
            total_files = success_count + error_count + timeout_count + empty_count
            print(f"\\n  Completed in {wall_clock:.1f}s")

            s3_objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX + "/")
            s3_count = s3_objects.get("KeyCount", 0)

            print("\\n" + "=" * 60)
            print("PARSE & CHUNK REPORT")
            print("=" * 60)
            print(f"Actors:            {MIN_ACTORS}..{MAX_ACTORS} (CPUs/actor: {CPUS_PER_ACTOR})")
            print(f"Output:            s3://{S3_BUCKET}/{S3_PREFIX}/")
            print(f"Total files:       {total_files}")
            print(f"Success:           {success_count}")
            print(f"Empty:             {empty_count}")
            print(f"Errors:            {error_count}")
            print(f"Timeouts:          {timeout_count}")
            print(f"Total pages:       {total_pages}")
            print(f"Total chunks:      {total_chunks}")
            print(f"S3 objects:        {s3_count}")
            print(f"Wall clock:        {wall_clock:.1f}s")
            if wall_clock > 0:
                print(f"Files/second:      {success_count / wall_clock:.2f}")
                print(f"Chunks/second:     {total_chunks / wall_clock:.2f}")
            for actor, count in sorted(actor_dist.items()):
                pct = count / total_files * 100 if total_files else 0.0
                print(f"  {actor}: {count} files ({pct:.1f}%)")
            if errors_list:
                for fname, err in errors_list[:10]:
                    print(f"  {fname}: {err[:80]}")
            print("=" * 60)


        if __name__ == "__main__":
            import sys
            print(f"Python version: {sys.version}")
            print(f"Ray initializing...")
            ray.init(ignore_reinit_error=True)
            print(f"Ray initialized")

            ctx = ray.data.DataContext.get_current()
            ctx.max_errored_blocks = MAX_ERRORED_BLOCKS
            log_verbose(f"Set max_errored_blocks to {MAX_ERRORED_BLOCKS}")

            if ENABLE_PROFILING:
                import cProfile
                import pstats
                import io
                print("\\n" + "=" * 80)
                print("PROFILING ENABLED")
                print("=" * 80)
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    run()
                finally:
                    profiler.disable()
                    print("\\n" + "=" * 80)
                    print("PROFILING RESULTS")
                    print("=" * 80)
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                    ps.print_stats(50)  # Top 50 functions
                    print(s.getvalue())
                    print("=" * 80)
            else:
                run()
    ''')

    from codeflare_sdk import ManagedClusterConfig, RayJob
    from kubernetes.client import (
        V1PersistentVolumeClaimVolumeSource,
        V1Volume,
        V1VolumeMount,
    )
    from ray.runtime_env import RuntimeEnv

    rayjob_name = f"docling-chunk-{int(time.time())}"

    # Encode entrypoint script as base64 env var to avoid creating
    # Secrets or ConfigMaps (service account lacks permissions for both).
    script_b64 = base64.b64encode(_DOCLING_CHUNK_PROCESS_PY.encode()).decode()

    shared_mount = V1VolumeMount(pvc_mount_path, name="shared-data", read_only=True)
    data_volume = V1Volume(
        name="shared-data",
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name=pvc_name,
        ),
    )

    cluster_config = ManagedClusterConfig(
        head_cpu_requests=head_cpus,
        head_cpu_limits=head_cpus,
        head_memory_requests=head_memory_gb,
        head_memory_limits=head_memory_gb,
        num_workers=num_workers,
        worker_cpu_requests=worker_cpus,
        worker_cpu_limits=worker_cpus,
        worker_memory_requests=worker_memory_gb,
        worker_memory_limits=worker_memory_gb,
        volume_mounts=[shared_mount],
        volumes=[data_volume],
        image=ray_image,
        envs={
            "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": "0.1",
            "RAY_OBJECT_STORE_MEMORY": "78643200",
            "HF_HOME": "/tmp/.hf_cache",
        },
    )

    job = RayJob(
        job_name=rayjob_name,
        entrypoint="python -c \"import base64,os;exec(base64.b64decode(os.environ['ENTRYPOINT_SCRIPT']).decode())\"",
        cluster_config=cluster_config,
        namespace=namespace,
        runtime_env=RuntimeEnv(
            pip=[
                "opencv-python-headless",
                "pypdfium2",
                "boto3>=1.28.0",
            ],
            env_vars={
                "ENTRYPOINT_SCRIPT": script_b64,
                "PVC_MOUNT_PATH": pvc_mount_path,
                "INPUT_PATH": input_path,
                "NUM_FILES": str(num_files),
                "MIN_ACTORS": str(min_actors),
                "MAX_ACTORS": str(max_actors),
                "CPUS_PER_ACTOR": str(cpus_per_actor),
                "BATCH_SIZE": str(batch_size),
                "REPARTITION_FACTOR": "10",
                "FILE_TIMEOUT": str(timeout_seconds),
                "MAX_ERRORED_BLOCKS": "50",
                "CHUNK_MAX_TOKENS": str(chunk_max_tokens),
                "TOKENIZER": tokenizer,
                "S3_ENDPOINT": s3_endpoint,
                "S3_BUCKET": s3_bucket,
                "S3_PREFIX": s3_prefix,
                "ENABLE_PROFILING": "true" if enable_profiling else "false",
                "VERBOSE": "true" if verbose else "false",
            },
        ),
        ttl_seconds_after_finished=300,
    )

    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    custom_api = k8s_client.CustomObjectsApi()
    rayjob_group = "ray.io"
    rayjob_version = "v1"
    rayjob_plural = "rayjobs"

    job.submit()
    print(f"RayJob '{rayjob_name}' submitted.")

    if bypass_kueue:
        rayjob_obj = custom_api.get_namespaced_custom_object(
            group=rayjob_group,
            version=rayjob_version,
            namespace=namespace,
            plural=rayjob_plural,
            name=rayjob_name,
        )
        labels = rayjob_obj.get("metadata", {}).get("labels", {})
        if "kueue.x-k8s.io/queue-name" in labels:
            custom_api.patch_namespaced_custom_object(
                group=rayjob_group,
                version=rayjob_version,
                namespace=namespace,
                plural=rayjob_plural,
                name=rayjob_name,
                body={"metadata": {"labels": {"kueue.x-k8s.io/queue-name": None}}},
            )
            print(f"Removed kueue.x-k8s.io/queue-name label from RayJob '{rayjob_name}'.")

    # Inject S3 credentials into Ray pods via secretKeyRef so they never
    # appear as plaintext in the RayJob spec.
    s3_env_vars = [
        {
            "name": "S3_ACCESS_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": s3_secret_name, "key": "access_key"},
            },
        },
        {
            "name": "S3_SECRET_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": s3_secret_name, "key": "secret_key"},
            },
        },
    ]
    rayjob_obj = custom_api.get_namespaced_custom_object(
        group=rayjob_group,
        version=rayjob_version,
        namespace=namespace,
        plural=rayjob_plural,
        name=rayjob_name,
    )
    spec = rayjob_obj.get("spec", {})
    cluster_spec = spec.get("rayClusterSpec", {})
    head_spec = cluster_spec.get("headGroupSpec", {})
    head_containers = head_spec.get("template", {}).get("spec", {}).get("containers", [])
    for c in head_containers:
        c.setdefault("env", []).extend(s3_env_vars)
    worker_groups = cluster_spec.get("workerGroupSpecs", [])
    for wg in worker_groups:
        worker_containers = wg.get("template", {}).get("spec", {}).get("containers", [])
        for c in worker_containers:
            c.setdefault("env", []).extend(s3_env_vars)
    custom_api.patch_namespaced_custom_object(
        group=rayjob_group,
        version=rayjob_version,
        namespace=namespace,
        plural=rayjob_plural,
        name=rayjob_name,
        body={"spec": {"rayClusterSpec": cluster_spec}},
    )
    print(f"Injected S3 credentials from Secret '{s3_secret_name}' into Ray pod templates.")

    if bypass_kueue:
        custom_api.patch_namespaced_custom_object(
            group=rayjob_group,
            version=rayjob_version,
            namespace=namespace,
            plural=rayjob_plural,
            name=rayjob_name,
            body={"spec": {"suspend": False}},
        )
        print(f"Unsuspended RayJob '{rayjob_name}' - cluster will now start provisioning.")
    else:
        print(f"RayJob '{rayjob_name}' waiting for Kueue admission...")

    # Wait for Ray cluster to be fully ready before job execution
    print(f"Waiting for Ray cluster to be ready with {num_workers} workers...")
    max_wait = 600  # 10 minutes
    start_time = time.time()
    while True:
        rayjob_obj = custom_api.get_namespaced_custom_object(
            group=rayjob_group,
            version=rayjob_version,
            namespace=namespace,
            plural=rayjob_plural,
            name=rayjob_name,
        )
        ray_status = rayjob_obj.get("status", {})
        cluster_status = ray_status.get("rayClusterStatus", {})

        ready_workers = str(cluster_status.get("readyWorkerReplicas", 0))
        cluster_state = cluster_status.get("state", "creating")

        if ready_workers == str(num_workers) and cluster_state == "ready":
            print(f"Ray cluster ready: {ready_workers}/{num_workers} workers ready, state={cluster_state}")
            break

        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise TimeoutError(
                f"Ray cluster did not become ready within {max_wait}s. "
                f"Ready workers: {ready_workers}/{num_workers}, state: {cluster_state}"
            )

        print(
            f"  Waiting for cluster... ready workers: {ready_workers}/{num_workers}, "
            f"state: {cluster_state} ({elapsed:.0f}s)"
        )
        time.sleep(10)

    # Wait for job completion
    max_job_wait = 4 * 3600
    job_start = time.time()
    print(f"Waiting for RayJob to complete (timeout {max_job_wait}s)...")
    while True:
        rayjob_obj = custom_api.get_namespaced_custom_object(
            group=rayjob_group,
            version=rayjob_version,
            namespace=namespace,
            plural=rayjob_plural,
            name=rayjob_name,
        )
        job_status = rayjob_obj.get("status", {}).get("jobStatus", "")
        if job_status == "SUCCEEDED":
            print("RayJob completed successfully.")
            break
        elif job_status == "FAILED":
            raise RuntimeError(f"RayJob '{rayjob_name}' failed.")

        job_elapsed = time.time() - job_start
        if job_elapsed > max_job_wait:
            raise TimeoutError(
                f"RayJob '{rayjob_name}' did not complete within {max_job_wait}s. Last status: {job_status}"
            )
        time.sleep(30)

    return f"s3://{s3_bucket}/{s3_prefix}"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        parse_and_chunk,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
