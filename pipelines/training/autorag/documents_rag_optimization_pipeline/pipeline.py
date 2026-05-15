from typing import List, Optional

from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.autorag.documents_discovery import (
    documents_discovery,
)
from kfp_components.components.data_processing.autorag.test_data_loader import (
    test_data_loader,
)
from kfp_components.components.data_processing.autorag.text_extraction import (
    text_extraction,
)
from kfp_components.components.deployment.autorag.build_responses_request_bodies.component import (
    prepare_responses_api_requests,
)
from kfp_components.components.training.autorag.leaderboard_evaluation import (
    leaderboard_evaluation,
)
from kfp_components.components.training.autorag.rag_templates_optimization.component import (
    rag_templates_optimization,
)
from kfp_components.components.training.autorag.search_space_preparation.component import (
    search_space_preparation,
)

MAX_CPUS = "32"
MAX_MEMORY = "64Gi"

SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})


@dsl.pipeline(
    name="documents-rag-optimization-pipeline",
    description="Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications",
)
def documents_rag_optimization_pipeline(
    test_data_secret_name: str,
    test_data_bucket_name: str,
    test_data_key: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    ogx_secret_name: str,
    vector_io_provider_id: str,
    input_data_key: str = "",
    embedding_models: Optional[List] = None,
    generation_models: Optional[List] = None,
    optimization_metric: str = "faithfulness",
    optimization_max_rag_patterns: int = 8,
):
    """Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications.

    The Documents RAG Optimization Pipeline is an automated system for building and optimizing
    Retrieval-Augmented Generation (RAG) applications within Red Hat OpenShift AI. It leverages
    Kubeflow Pipelines to orchestrate the optimization workflow, using the ai4rag optimization
    engine to systematically explore RAG configurations and identify the best performing parameter
    settings based on an upfront-specified quality metric.

    The system integrates with OGX API for inference and vector database operations,
    producing optimized RAG patterns as artifacts that can be deployed and used for production
    RAG applications. After optimization, request JSON bodies for OGX ``/v1/responses`` are
    emitted per pattern (``prepare_responses_api_requests``).

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
        ogx_secret_name: Name of the Kubernetes secret for OGX API connection.
            The secret must define: OGX_CLIENT_API_KEY, OGX_CLIENT_BASE_URL.
        vector_io_provider_id: Vector I/O provider id (e.g., registered in OGX Milvus).
        input_data_key: Object key (path) of the input documents in the input data bucket.
        embedding_models: Optional list of embedding model identifiers to use in the search space.
        generation_models: Optional list of foundation/generation model identifiers to use in the
            search space.
        optimization_metric: Quality metric used to optimize RAG patterns. Supported values:
            "faithfulness", "answer_correctness", "context_correctness".
        optimization_max_rag_patterns: Maximum number of RAG patterns to generate. Passed to ai4rag
            (max_number_of_rag_patterns). Defaults to 8.
    """
    test_data_loader_task = test_data_loader(
        test_data_bucket_name=test_data_bucket_name,
        test_data_path=test_data_key,
    )

    test_data_loader_task.set_caching_options(False)
    test_data_loader_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )

    documents_discovery_task = documents_discovery(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_key,
        test_data=test_data_loader_task.outputs["test_data"],
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

    for task, secret_name in zip(
        [test_data_loader_task, documents_discovery_task, text_extraction_task],
        [test_data_secret_name, input_data_secret_name, input_data_secret_name],
    ):
        use_secret_as_env(
            task,
            secret_name=secret_name,
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
                "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
                "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            },
            optional=True,
        )

    mps_task = search_space_preparation(
        test_data=test_data_loader_task.outputs["test_data"],
        extracted_text=text_extraction_task.outputs["extracted_text"],
        embedding_models=embedding_models,
        generation_models=generation_models,
    )

    mps_task.set_caching_options(False)
    mps_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(MAX_MEMORY)

    hpo_task = rag_templates_optimization(
        extracted_text=text_extraction_task.outputs["extracted_text"],
        test_data=test_data_loader_task.outputs["test_data"],
        search_space_prep_report=mps_task.outputs["search_space_prep_report"],
        vector_io_provider_id=vector_io_provider_id,
        optimization_settings={
            "metric": optimization_metric,
            "max_number_of_rag_patterns": optimization_max_rag_patterns,
        },
        test_data_key=test_data_key,
        input_data_key=input_data_key,
    )

    hpo_task.set_caching_options(False)
    hpo_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(MAX_MEMORY)

    use_secret_as_env(
        mps_task,
        ogx_secret_name,
        {
            "OGX_CLIENT_BASE_URL": "OGX_CLIENT_BASE_URL",
            "OGX_CLIENT_API_KEY": "OGX_CLIENT_API_KEY",
        },
    )
    use_secret_as_env(
        hpo_task,
        ogx_secret_name,
        {
            "OGX_CLIENT_BASE_URL": "OGX_CLIENT_BASE_URL",
            "OGX_CLIENT_API_KEY": "OGX_CLIENT_API_KEY",
        },
    )

    prepare_responses_api_requests_task = prepare_responses_api_requests(
        rag_patterns=hpo_task.outputs["rag_patterns"],
    )
    prepare_responses_api_requests_task.set_caching_options(False)
    prepare_responses_api_requests_task.set_cpu_request("500m").set_memory_request("2Gi").set_cpu_limit(
        MAX_CPUS
    ).set_memory_limit(MAX_MEMORY)
    use_secret_as_env(
        prepare_responses_api_requests_task,
        ogx_secret_name,
        {
            "OGX_CLIENT_BASE_URL": "OGX_CLIENT_BASE_URL",
            "OGX_CLIENT_API_KEY": "OGX_CLIENT_API_KEY",
        },
    )

    leaderboard_evaluation_task = leaderboard_evaluation(rag_patterns=hpo_task.outputs["rag_patterns"])
    leaderboard_evaluation_task.set_caching_options(False)
    leaderboard_evaluation_task.set_cpu_request("1").set_memory_request("4Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_rag_optimization_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
