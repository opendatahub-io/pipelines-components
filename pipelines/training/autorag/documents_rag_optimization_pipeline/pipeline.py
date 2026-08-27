from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.autorag.documents_discovery import (
    documents_discovery,
)
from kfp_components.components.data_processing.autorag.text_extraction import (
    text_extraction,
)
from kfp_components.components.training.autorag.component_stage_map_publisher import (
    publish_component_stage_map,
)
from kfp_components.components.training.autorag.models_pre_selector.component import (
    models_pre_selector,
)
from kfp_components.components.training.autorag.rag_templates_optimization.component import (
    rag_templates_optimization,
)
from kfp_components.components.training.autorag.search_space_preparation.component import (
    search_space_preparation,
)

MAX_CPUS = "32"
MAX_MEMORY = "64Gi"

# Must match run_status_templates/pipelines/<name>.json
PIPELINE_NAME = "documents-rag-optimization-pipeline"

# Inference credentials exposed by the MaaS secret.
MAAS_SECRET_KEYS = {
    "MAAS_BASE_URL": "MAAS_BASE_URL",
    "MAAS_API_KEY": "MAAS_API_KEY",
}


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "AutoRAG pipeline for building high-quality RAG applications from your documents with minimal "
        "configuration. Powered by ai4rag, it explores and optimizes retrieval and generation design choices "
        "against your quality goals. Delivers ranked, production-ready patterns, deployment-ready settings, "
        "and a leaderboard of the best configurations."
    ),
)
def documents_rag_optimization_pipeline(
    test_data_secret_name: str,
    test_data_bucket_name: str,
    test_data_key: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    maas_secret_name: str,
    vector_db_secret_name: str,
    embedding_models: list[str],
    generation_models: list[str],
    input_data_key: str = "",
    optimization_metric: str = "overall_score",
    optimization_max_rag_patterns: int = 8,
    preset: str = "speed",
):
    """Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications.

    The Documents RAG Optimization Pipeline is an automated system for building and optimizing
    Retrieval-Augmented Generation (RAG) applications within Red Hat OpenShift AI. It leverages
    Kubeflow Pipelines to orchestrate the optimization workflow, using the ai4rag optimization
    engine to systematically explore RAG configurations and identify the best performing parameter
    settings based on an upfront-specified quality metric.

    The system integrates with MaaS (Models-as-a-Service) for inference and a vector database
    (Milvus or PGVector) for retrieval, producing optimized RAG patterns as artifacts that can
    be deployed and used for production RAG applications. Each optimized pattern contains a
    ``pattern.json`` (with deployment settings), executable notebooks, and evaluation results.

    Args:
        test_data_secret_name: Name of the Kubernetes secret holding S3-compatible credentials for
            test data access. The following environment variables are required:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
            AWS_DEFAULT_REGION is optional.
        test_data_bucket_name: S3 (or compatible) bucket name for the test data file.
        test_data_key: Object key (path) of the test data JSON file in the test data bucket.
        input_data_secret_name: Name of the Kubernetes secret holding S3-compatible credentials
            for input document data access. The following environment variables are required:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
            AWS_DEFAULT_REGION is optional.
        input_data_bucket_name: S3 (or compatible) bucket name for the input documents.
        maas_secret_name: Name of the Kubernetes secret for the MaaS inference connection.
            The secret must define: MAAS_BASE_URL, MAAS_API_KEY.
        vector_db_secret_name: Name of the Kubernetes secret carrying the vector database
            configuration. The env-var prefix selects the backend: ``MILVUS_*`` keys (at least
            ``MILVUS_URI``) select Milvus, ``PGVECTOR_*`` keys select PGVector.
        embedding_models: List of embedding model identifiers to use in the search space.
            Required: MaaS exposes no metadata to distinguish model types, so embedding
            models can no longer be inferred and must be declared explicitly.
        generation_models: List of foundation/generation model identifiers to use in the
            search space. Required: MaaS exposes no metadata to distinguish model types, so
            generation models can no longer be inferred and must be declared explicitly.
        input_data_key: Object key (path) of the input documents in the input data bucket.
        optimization_metric: Quality metric used to rank RAG patterns. Supported values:
            "faithfulness", "answer_correctness", "context_correctness", "answer_relevance",
            and "overall_score" (default). "faithfulness", "answer_correctness", and
            "context_correctness" are deterministic Unitxt metrics; choosing one as the
            optimization metric keeps the experiment deterministic. The LLM-judge metric
            "answer_relevance" is always computed but only drives optimization when selected
            (or via "overall_score", which aggregates all metrics).
        optimization_max_rag_patterns: Maximum number of RAG patterns to generate. Passed to ai4rag
            (max_number_of_rag_patterns). Defaults to 8.
        preset: Pipeline quality tier. "speed" (default) uses recursive chunking,
            no table structure parsing, and no contextual enrichment. "balanced"
            enables Docling table layout parsing, hybrid chunking, and LLM
            contextual enrichment. Both presets use the same resource tier.
    """
    component_stage_map_task = publish_component_stage_map(
        pipeline_id=PIPELINE_NAME,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    )
    component_stage_map_task.set_caching_options(False)
    component_stage_map_task.set_cpu_request("0.5").set_memory_request("512Mi").set_cpu_limit("1").set_memory_limit(
        "1Gi"
    )

    documents_discovery_task = documents_discovery(
        input_data_bucket_name=input_data_bucket_name,
        test_data_bucket_name=test_data_bucket_name,
        test_data_path_key=test_data_key,
        input_data_path=input_data_key,
    )
    documents_discovery_task.after(component_stage_map_task)

    documents_discovery_task.set_caching_options(False)
    documents_discovery_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )

    search_space_preparation_task = search_space_preparation(
        test_data=documents_discovery_task.outputs["test_data"],
        embedding_models=embedding_models,
        generation_models=generation_models,
        preset=preset,
    )

    search_space_preparation_task.set_caching_options(False)
    search_space_preparation_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(
        MAX_CPUS
    ).set_memory_limit(MAX_MEMORY)

    text_extraction_task = text_extraction(
        documents_descriptor=documents_discovery_task.outputs["discovered_documents"],
        preset=preset,
    )
    text_extraction_task.after(search_space_preparation_task)

    text_extraction_task.set_caching_options(False)
    text_extraction_task.set_cpu_request("4").set_memory_request("16Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )

    models_pre_selector_task = models_pre_selector(
        search_space_report=search_space_preparation_task.outputs["search_space_report"],
        extracted_text=text_extraction_task.outputs["extracted_text"],
        test_data=documents_discovery_task.outputs["test_data"],
        preset=preset,
    )

    models_pre_selector_task.set_caching_options(False)
    models_pre_selector_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )

    rag_optimization_task = rag_templates_optimization(
        extracted_text=text_extraction_task.outputs["extracted_text"],
        test_data=documents_discovery_task.outputs["test_data"],
        search_space_mps_report=models_pre_selector_task.outputs["search_space_mps_report"],
        maas_secret_name=maas_secret_name,
        vector_db_secret_name=vector_db_secret_name,
        input_data_secret_name=input_data_secret_name,
        input_data_bucket_name=input_data_bucket_name,
        optimization_settings={
            "metric": optimization_metric,
            "max_number_of_rag_patterns": optimization_max_rag_patterns,
        },
        test_data_key=test_data_key,
        input_data_key=input_data_key,
        preset=preset,
        llm_judge_mode="ragas",
    )

    rag_optimization_task.set_caching_options(False)
    rag_optimization_task.set_cpu_request("4").set_memory_request("16Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )

    # Object storage credentials for document discovery and text extraction.
    use_secret_as_env(
        documents_discovery_task,
        secret_name=input_data_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "INPUT_DATA_AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "INPUT_DATA_AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "INPUT_DATA_AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "INPUT_DATA_AWS_DEFAULT_REGION",
        },
        optional=True,
    )
    use_secret_as_env(
        documents_discovery_task,
        secret_name=test_data_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "TEST_DATA_AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "TEST_DATA_AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "TEST_DATA_AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "TEST_DATA_AWS_DEFAULT_REGION",
        },
        optional=True,
    )
    use_secret_as_env(
        text_extraction_task,
        secret_name=input_data_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
        optional=True,
    )

    use_secret_as_env(search_space_preparation_task, maas_secret_name, MAAS_SECRET_KEYS)
    use_secret_as_env(models_pre_selector_task, maas_secret_name, MAAS_SECRET_KEYS)
    use_secret_as_env(rag_optimization_task, maas_secret_name, MAAS_SECRET_KEYS)

    use_secret_as_env(
        rag_optimization_task,
        vector_db_secret_name,
        secret_key_to_env={
            "MILVUS_URI": "MILVUS_URI",
            "MILVUS_TOKEN": "MILVUS_TOKEN",
            "MILVUS_SERVER_CERT": "MILVUS_SERVER_CERT",
            "PGVECTOR_HOST": "PGVECTOR_HOST",
            "PGVECTOR_PORT": "PGVECTOR_PORT",
            "PGVECTOR_DB": "PGVECTOR_DB",
            "PGVECTOR_USER": "PGVECTOR_USER",
            "PGVECTOR_PASSWORD": "PGVECTOR_PASSWORD",
        },
        optional=True,
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_rag_optimization_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
