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
    test_data_path: str = "",
    benchmark_sample_size: int = 25,
    sampling_enabled: bool = True,
    sampling_max_size: float = 1,
    discovered_documents: dsl.Output[dsl.Artifact] = None,
    test_data: dsl.Output[dsl.Artifact] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
):
    """Documents discovery component.

    Optionally downloads benchmark test data from S3, then lists and samples input
    documents. Delegates to ``ai4rag.components.data.test_data_loader.load_test_data``
    and ``ai4rag.components.data.documents_discovery.discover_documents``.

    Args:
        input_data_bucket_name: S3 (or compatible) bucket containing input data.
        input_data_path: Path to folder with input documents within the bucket.
        test_data_bucket_name: S3 bucket containing benchmark test data JSON. When
            set together with ``test_data_path``, test data is downloaded and used
            for document sampling.
        test_data_path: S3 object key to the benchmark test data JSON file.
        benchmark_sample_size: Maximum number of benchmark records to keep. When the
            dataset exceeds this limit, a reproducible random sample is drawn (seed 42).
            Set to 0 to disable sampling and keep all records.
        sampling_enabled: Whether to enable sampling or not.
        sampling_max_size: Maximum size of sampled documents (in gigabytes).
        discovered_documents: Output artifact containing the documents descriptor JSON file.
        test_data: Output artifact with the (possibly sampled) benchmark JSON when
            ``test_data_bucket_name`` and ``test_data_path`` are provided.
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.
    """
    import importlib.util
    import json
    import logging
    from pathlib import Path

    from ai4rag.components.data.documents_discovery import discover_documents
    from ai4rag.components.data.test_data_loader import load_test_data
    from ai4rag.components.utils.s3 import create_s3_client

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
        s3_client = create_s3_client()
        benchmark_records = None

        if test_data_bucket_name and test_data_path:
            with status.stage("load_benchmark"):
                result = load_test_data(
                    bucket_name=test_data_bucket_name,
                    key=test_data_path,
                    benchmark_sample_size=benchmark_sample_size,
                    s3_client=s3_client,
                )
                benchmark_records = result.data
                if test_data is not None:
                    with open(test_data.path, "w", encoding="utf-8") as f:
                        json.dump(benchmark_records, f, indent=2, ensure_ascii=False)

        with status.stage("discover_documents"):
            test_data_doc_names = None
            if benchmark_records is not None:
                test_data_doc_names = list(
                    {
                        doc_id
                        for record in benchmark_records
                        for doc_id in record.get("correct_answer_document_ids", [])
                    }
                )

            result = discover_documents(
                bucket_name=input_data_bucket_name,
                prefix=input_data_path,
                test_data_doc_names=test_data_doc_names,
                sampling_enabled=sampling_enabled,
                sampling_max_size_gb=sampling_max_size,
                s3_client=s3_client,
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
