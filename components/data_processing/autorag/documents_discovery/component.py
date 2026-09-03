from pathlib import Path

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[3] / "training" / "autorag" / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def documents_discovery(
    input_data_bucket_name: str,
    input_data_path: str = "",
    test_data_bucket_name: str = "",
    test_data_path_key: str = "",
    benchmark_sample_size: int = 25,
    sampling_enabled: bool = True,
    sampling_max_size: float = 1,
    discovered_documents: dsl.Output[dsl.Artifact] = None,
    test_data: dsl.Output[dsl.Artifact] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
):
    """Documents discovery component with optional benchmark test data loading.

    Discovers input documents in S3 and optionally downloads benchmark test data
    for document-prioritised sampling.  When ``test_data_bucket_name`` is provided,
    the component first downloads and samples the benchmark JSON, then uses the
    referenced document IDs to prioritise discovery.

    Args:
        input_data_bucket_name: S3 (or compatible) bucket containing input documents.
        input_data_path: Path to folder with input documents within the bucket.
        test_data_bucket_name: S3 bucket containing the test data file.  Leave empty
            to skip test data loading (e.g. for the indexing pipeline).
        test_data_path_key: S3 object key to the JSON test data file.
        benchmark_sample_size: Maximum number of benchmark records to keep.
            When the dataset exceeds this limit, a reproducible random sample is
            drawn (seed 42). Set to 0 to disable sampling.
        sampling_enabled: Whether to enable document size-based sampling.
        sampling_max_size: Maximum size of sampled documents (in gigabytes).
        discovered_documents: Output artifact containing the documents descriptor JSON file.
        test_data: Output artifact containing the (possibly sampled) test data JSON.
            Empty when test data loading is skipped.
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT for input documents
        (or INPUT_DATA_AWS_* prefixed variants when both secrets are injected).
        TEST_DATA_AWS_ACCESS_KEY_ID, TEST_DATA_AWS_SECRET_ACCESS_KEY,
        TEST_DATA_AWS_S3_ENDPOINT for test data (only when test_data_bucket_name is set).
        AWS_DEFAULT_REGION / *_DEFAULT_REGION variants are optional.
    """
    import importlib.util
    import json
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.clients.s3 import create_s3_client
    from ai4rag.utils.data import discover_documents

    logging.basicConfig(level=logging.INFO)

    if component_status is None:
        from kfp_components.components.training.autorag.shared.component_status import (  # pyright: ignore[reportMissingImports]
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
    else:
        _embedded_path = Path(embedded_artifact.path)
        _module_path = _embedded_path if _embedded_path.is_file() else _embedded_path / "component_status.py"
        _spec = importlib.util.spec_from_file_location("_autorag_component_status", _module_path)
        if _spec is None or _spec.loader is None:
            raise ValueError(f"Cannot load embedded module from {_module_path}")
        _status_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_status_module)
        status = _status_module.bootstrap_status_tracker(embedded_artifact, component_status, "documents_discovery")
    with status:
        if component_status is not None:
            status.set_metadata(display_name="Documents Discovery Status")
            component_status.metadata["display_name"] = "Documents Discovery Status"

        test_data_doc_names = None
        load_benchmark = bool(test_data_bucket_name and test_data_path_key)

        if load_benchmark:
            from ai4rag.utils.data import load_test_data

            with status.stage("load_benchmark"):
                test_data_s3_client = create_s3_client(
                    endpoint_url=os.environ.get("TEST_DATA_AWS_S3_ENDPOINT"),
                    access_key_id=os.environ.get("TEST_DATA_AWS_ACCESS_KEY_ID"),
                    secret_access_key=os.environ.get("TEST_DATA_AWS_SECRET_ACCESS_KEY"),
                    region_name=os.environ.get("TEST_DATA_AWS_DEFAULT_REGION"),
                )

                test_data_result = load_test_data(
                    bucket_name=test_data_bucket_name,
                    key=test_data_path_key,
                    benchmark_sample_size=benchmark_sample_size,
                    s3_client=test_data_s3_client,
                )

                with open(test_data.path, "w", encoding="utf-8") as f:
                    json.dump(test_data_result.data, f, indent=2, ensure_ascii=False)

                test_data_doc_names = list(
                    {doc_key for r in test_data_result.data for doc_key in r.get("correct_answer_document_keys", [])}
                )

        with status.stage("discover_documents"):
            input_endpoint = os.environ.get("INPUT_DATA_AWS_S3_ENDPOINT")
            if input_endpoint:
                input_s3_client = create_s3_client(
                    endpoint_url=input_endpoint,
                    access_key_id=os.environ.get("INPUT_DATA_AWS_ACCESS_KEY_ID"),
                    secret_access_key=os.environ.get("INPUT_DATA_AWS_SECRET_ACCESS_KEY"),
                    region_name=os.environ.get("INPUT_DATA_AWS_DEFAULT_REGION"),
                )
            else:
                input_s3_client = create_s3_client()

            result = discover_documents(
                bucket_name=input_data_bucket_name,
                prefix=input_data_path,
                test_data_doc_names=test_data_doc_names,
                sampling_enabled=sampling_enabled,
                sampling_max_size_gb=sampling_max_size,
                s3_client=input_s3_client,
            )

            output_dir = Path(discovered_documents.path)
            output_dir.mkdir(parents=True, exist_ok=True)
            result.save(path=output_dir, filename="documents_descriptor.json")


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_discovery,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
