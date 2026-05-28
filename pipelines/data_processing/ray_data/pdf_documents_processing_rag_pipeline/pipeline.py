"""KFP Pipeline: Multi-Step RAG Document Processing.

Orchestrates up to five reusable components:
1. Parse & chunk PDFs (Docling + HybridChunker via RayJob -> S3)
2. Deploy embedding model (optional — only when deploy_embedding=True)
3. Ingest into Milvus (read chunks from S3, embed locally or via service, insert)
4. Download LLM to PVC (cached — skips if already present)
5. Deploy LLM for RAG inference (vLLM InferenceService from PVC)

Intermediate chunks are stored in S3 (MinIO), not on the PVC.
The data PVC is mounted read-only for input PDFs.
The model cache PVC stores downloaded HuggingFace models.
Embedding supports two modes: local sentence-transformers or a deployed service.
S3 credentials are read from a Kubernetes Secret (not pipeline parameters).
"""

from kfp import dsl, kubernetes
from kfp_components.components.data_processing.download_model import download_model
from kfp_components.components.data_processing.ingest_to_milvus import ingest_to_milvus
from kfp_components.components.data_processing.parse_and_chunk import parse_and_chunk
from kfp_components.components.deployment.deploy_embedding_model import deploy_embedding_model
from kfp_components.components.deployment.model_deployment import model_deployment

_DEFAULT_EMBEDDING_RUNTIME_IMAGE = (
    "registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:094db84a1da5e8a575d0c9eade114fa30f4a2061064a338e3e032f3578f8082a"
)


@dsl.pipeline(
    name="RAG Multi-Step Pipeline",
    description=(
        "Multi-step RAG pipeline: parse & chunk PDFs with Docling "
        "(output to S3), ingest into Milvus with local or deployed embeddings, "
        "optionally deploy an embedding service, "
        "download and deploy an LLM for inference (cached on PVC)."
    ),
)
def rag_multistep_pipeline(
    # Shared
    pvc_name: str = "data-pvc",
    pvc_mount_path: str = "/mnt/data",
    namespace: str = "ray-docling",
    # S3 (MinIO)
    s3_endpoint: str = "http://minio-service.default.svc.cluster.local:9000",
    s3_bucket: str = "rag-chunks",
    s3_prefix: str = "chunks",
    s3_secret_name: str = "minio-secret",
    # PDF parsing
    input_path: str = "input/pdfs",
    ray_image: str = "quay.io/rhoai-szaher/docling-ray:latest",
    num_workers: int = 2,
    worker_cpus: int = 8,
    worker_memory_gb: int = 16,
    head_cpus: int = 2,
    head_memory_gb: int = 8,
    cpus_per_actor: int = 4,
    min_actors: int = 2,
    max_actors: int = 4,
    batch_size: int = 4,
    chunk_max_tokens: int = 256,
    num_files: int = 1000,
    timeout_seconds: int = 600,
    enable_profiling: bool = False,
    verbose: bool = True,
    bypass_kueue: bool = False,
    # Embedding
    deploy_embedding: bool = False,  # If True, deploys embedding model as InferenceService
    embedding_endpoint: str = "",  # If empty, uses local model; else uses deployed service
    embedding_model: str = "ibm-granite/granite-embedding-125m-english",
    embedding_dim: int = 768,
    embedding_runtime_image: str = _DEFAULT_EMBEDDING_RUNTIME_IMAGE,
    embedding_serving_runtime_name: str = "embedding-runtime",
    embedding_gpu_count: int = 1,
    embedding_min_replicas: int = 1,
    embedding_max_replicas: int = 1,
    embedding_cpu_requests: str = "2",
    embedding_cpu_limits: str = "4",
    embedding_memory_requests: str = "4Gi",
    embedding_memory_limits: str = "8Gi",
    embedding_max_model_len: int = 512,
    # Milvus
    milvus_host: str = "milvus-milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    milvus_db: str = "default",
    milvus_token: str = "",
    collection_name: str = "rag_documents",
    drop_existing: bool = True,
    embed_batch_size: int = 64,
    milvus_batch_size: int = 256,
    # LLM deployment
    hf_secret_name: str = "hf-token-secret",
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    model_cache_pvc: str = "model-cache-pvc",
    model_cache_mount: str = "/mnt/models",
    max_model_len: int = 4096,
    gpu_count: int = 1,
    llm_hardware_profile_name: str = "gpu-profile",
    llm_hardware_profile_namespace: str = "redhat-ods-applications",
    llm_min_replicas: int = 1,
    llm_max_replicas: int = 1,
    llm_cpu_requests: str = "2",
    llm_cpu_limits: str = "2",
    llm_memory_requests: str = "8Gi",
    llm_memory_limits: str = "8Gi",
    llm_force_recreate: bool = False,
):
    """Multi-step RAG pipeline: parse PDFs, ingest into Milvus, deploy LLM.

    Args:
        pvc_name: PVC containing input PDF documents.
        pvc_mount_path: Mount path for the data PVC inside pods.
        namespace: OpenShift namespace for all resources.
        s3_endpoint: S3-compatible endpoint URL (e.g. MinIO).
        s3_bucket: S3 bucket for intermediate chunk storage.
        s3_prefix: Key prefix for chunk files in S3.
        s3_secret_name: Kubernetes Secret with S3 credentials.
        input_path: Path to PDF files on the PVC.
        ray_image: Container image with Ray and Docling pre-installed.
        num_workers: Number of Ray worker pods.
        worker_cpus: CPUs per Ray worker pod.
        worker_memory_gb: Memory (GB) per Ray worker pod.
        head_cpus: CPUs for the Ray head pod.
        head_memory_gb: Memory (GB) for the Ray head pod.
        cpus_per_actor: CPUs per Docling processing actor.
        min_actors: Minimum actor pool size.
        max_actors: Maximum actor pool size.
        batch_size: Files per batch sent to each actor.
        chunk_max_tokens: Maximum tokens per chunk.
        num_files: Number of PDFs to process (0 = all).
        timeout_seconds: Per-file processing timeout in seconds.
        enable_profiling: Enable cProfile profiling output.
        verbose: Enable verbose logging.
        bypass_kueue: If True, bypass Kueue quota management for the RayJob.
        deploy_embedding: If True, deploy embedding model as InferenceService.
        embedding_endpoint: Embedding service URL (empty = local model).
        embedding_model: Embedding model name.
        embedding_dim: Embedding vector dimension.
        embedding_runtime_image: Container image for the embedding server.
        embedding_serving_runtime_name: Name of the embedding ServingRuntime CR.
        embedding_gpu_count: GPUs for the embedding service.
        embedding_min_replicas: Minimum replicas for the embedding service.
        embedding_max_replicas: Maximum replicas for the embedding service.
        embedding_cpu_requests: CPU requests for the embedding service.
        embedding_cpu_limits: CPU limits for the embedding service.
        embedding_memory_requests: Memory requests for the embedding service.
        embedding_memory_limits: Memory limits for the embedding service.
        embedding_max_model_len: Maximum sequence length for the embedding model.
        milvus_host: Milvus service hostname.
        milvus_port: Milvus gRPC port.
        milvus_db: Milvus database name.
        milvus_token: Milvus authentication token. Empty string for unauthenticated connections.
        collection_name: Milvus collection name.
        drop_existing: If True, drop and recreate the Milvus collection. If False, append.
        embed_batch_size: Batch size for embedding requests.
        milvus_batch_size: Batch size for Milvus inserts.
        llm_model_name: HuggingFace LLM model ID for inference.
        hf_secret_name: Kubernetes Secret with HuggingFace token (key: 'token').
        model_cache_pvc: PVC for cached model weights.
        model_cache_mount: Mount path for the model cache PVC.
        max_model_len: Maximum context length for the LLM.
        gpu_count: GPUs for LLM serving.
        llm_hardware_profile_name: Hardware profile name for LLM deployment.
        llm_hardware_profile_namespace: Namespace of the hardware profile.
        llm_min_replicas: Minimum replicas for the LLM service.
        llm_max_replicas: Maximum replicas for the LLM service.
        llm_cpu_requests: CPU requests for the LLM service.
        llm_cpu_limits: CPU limits for the LLM service.
        llm_memory_requests: Memory requests for the LLM service.
        llm_memory_limits: Memory limits for the LLM service.
        llm_force_recreate: If True, delete and recreate the LLM InferenceService
            (causes downtime). If False (default), patch in place.
    """
    # Step 1: Parse & chunk PDFs -> S3
    chunk_task = parse_and_chunk(
        pvc_name=pvc_name,
        pvc_mount_path=pvc_mount_path,
        input_path=input_path,
        ray_image=ray_image,
        namespace=namespace,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_secret_name=s3_secret_name,
        tokenizer=embedding_model,
        chunk_max_tokens=chunk_max_tokens,
        num_workers=num_workers,
        worker_cpus=worker_cpus,
        worker_memory_gb=worker_memory_gb,
        head_cpus=head_cpus,
        head_memory_gb=head_memory_gb,
        cpus_per_actor=cpus_per_actor,
        min_actors=min_actors,
        max_actors=max_actors,
        batch_size=batch_size,
        num_files=num_files,
        timeout_seconds=timeout_seconds,
        enable_profiling=enable_profiling,
        verbose=verbose,
        bypass_kueue=bypass_kueue,
    )
    chunk_task.set_caching_options(False)

    # Step 2 + 3: Deploy embedding (optional) then ingest into Milvus.
    # When deploy_embedding=True, deploy the embedding model first and
    # pass its endpoint URL to the ingest step.
    # When deploy_embedding=False, embedding_endpoint is used as-is
    # (empty string = local model, URL = existing service).
    with dsl.If(deploy_embedding == True):  # noqa: E712
        embed_deploy_task = deploy_embedding_model(
            model_name=embedding_model,
            namespace=namespace,
            serving_runtime_name=embedding_serving_runtime_name,
            runtime_image=embedding_runtime_image,
            min_replicas=embedding_min_replicas,
            max_replicas=embedding_max_replicas,
            cpu_requests=embedding_cpu_requests,
            cpu_limits=embedding_cpu_limits,
            memory_requests=embedding_memory_requests,
            memory_limits=embedding_memory_limits,
            gpu_count=embedding_gpu_count,
            max_model_len=embedding_max_model_len,
        )
        embed_deploy_task.after(chunk_task)
        embed_deploy_task.set_caching_options(False)

        ingest_with_service = ingest_to_milvus(
            s3_endpoint=s3_endpoint,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            milvus_db=milvus_db,
            milvus_token=milvus_token,
            collection_name=collection_name,
            drop_existing=drop_existing,
            embedding_endpoint=embed_deploy_task.output,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embed_batch_size=embed_batch_size,
            milvus_batch_size=milvus_batch_size,
        )
        kubernetes.use_secret_as_env(
            ingest_with_service,
            secret_name=s3_secret_name,
            secret_key_to_env={
                "access_key": "S3_ACCESS_KEY",
                "secret_key": "S3_SECRET_KEY",
            },
        )
        ingest_with_service.set_caching_options(False)

    with dsl.Else():
        ingest_task = ingest_to_milvus(
            s3_endpoint=s3_endpoint,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            milvus_db=milvus_db,
            milvus_token=milvus_token,
            collection_name=collection_name,
            drop_existing=drop_existing,
            embedding_endpoint=embedding_endpoint,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embed_batch_size=embed_batch_size,
            milvus_batch_size=milvus_batch_size,
        )
        kubernetes.use_secret_as_env(
            ingest_task,
            secret_name=s3_secret_name,
            secret_key_to_env={
                "access_key": "S3_ACCESS_KEY",
                "secret_key": "S3_SECRET_KEY",
            },
        )
        ingest_task.after(chunk_task)
        ingest_task.set_caching_options(False)

    # Step 4: Download LLM to PVC (skips if already cached)
    download_task = download_model(
        model_name=llm_model_name,
        model_cache_pvc=model_cache_pvc,
        model_cache_mount=model_cache_mount,
    )
    download_task.set_caching_options(False)
    kubernetes.mount_pvc(
        download_task,
        pvc_name=model_cache_pvc,
        mount_path="/mnt/models",
    )
    kubernetes.use_secret_as_env(
        download_task,
        secret_name=hf_secret_name,
        secret_key_to_env={"token": "HF_TOKEN"},
    )

    # Step 5: Deploy LLM for RAG inference (from PVC cache)
    deploy_task = model_deployment(
        model_name=llm_model_name,
        namespace=namespace,
        model_dir=download_task.output,
        model_cache_pvc=model_cache_pvc,
        hardware_profile_name=llm_hardware_profile_name,
        hardware_profile_namespace=llm_hardware_profile_namespace,
        min_replicas=llm_min_replicas,
        max_replicas=llm_max_replicas,
        gpu_count=gpu_count,
        max_model_len=max_model_len,
        cpu_requests=llm_cpu_requests,
        memory_requests=llm_memory_requests,
        cpu_limits=llm_cpu_limits,
        memory_limits=llm_memory_limits,
        force_recreate=llm_force_recreate,
    )
    deploy_task.set_caching_options(False)
    # No dependency on ingest_task — model deployment runs in parallel
    # with the data pipeline (parse → ingest). Both chains are independent:
    #   Data chain:  parse_and_chunk → ingest_to_milvus
    #   Model chain: download_model  → model_deployment


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        rag_multistep_pipeline,
        package_path="rag_multistep_pipeline.yaml",
    )
    print("Pipeline compiled to rag_multistep_pipeline.yaml")
