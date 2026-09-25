"""Unit tests for the documents_indexing_pipeline pipeline."""

import inspect
import tempfile
from pathlib import Path

import pytest
from kfp import compiler
from kfp_components.components.data_processing.autorag.documents_indexing.component import documents_indexing
from kfp_components.utils.autorag_extraction import normalize_extraction_preset
from kfp_components.utils.pipeline_dag_tasks import (
    assert_compiled_pipeline_root_dag_task_ids,
    load_pipeline_spec_document,
)

from ..pipeline import documents_indexing_pipeline

_EXPECTED_ROOT_DAG_TASK_IDS = (
    "condition-branches-1",
    "documents-discovery",
    "documents-indexing",
    "normalize-extraction-preset",
)


class TestDocumentsIndexingPipelineUnit:
    """Unit tests for pipeline structure, wiring, and compile stability."""

    def test_pipeline_is_callable(self):
        """Pipeline is a GraphComponent with expected inputs."""
        assert callable(documents_indexing_pipeline)
        assert hasattr(documents_indexing_pipeline, "_component_inputs")

    def test_pipeline_required_parameters(self):
        """Pipeline declares S3, MaaS, vector-DB, and indexing parameters."""
        inputs = getattr(documents_indexing_pipeline, "_component_inputs", set())
        for name in (
            "maas_secret_name",
            "vector_db_secret_name",
            "embedding_model_id",
            "input_data_secret_name",
            "input_data_bucket_name",
            "chunk_size",
            "chunk_overlap",
            "collection_name",
            "ocr_lang",
            "preset",
        ):
            assert name in inputs

    def test_pipeline_compiles(self):
        """Pipeline compiles to a valid YAML package."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            compiler.Compiler().compile(
                pipeline_func=documents_indexing_pipeline,
                package_path=tmp_path,
            )
            assert Path(tmp_path).is_file()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_compiled_pipeline_root_dag_task_ids(self):
        """Root DAG task IDs match discovery → extraction → indexing order."""
        assert_compiled_pipeline_root_dag_task_ids(
            pipeline_func=documents_indexing_pipeline,
            expected_task_ids=_EXPECTED_ROOT_DAG_TASK_IDS,
        )

    def test_compiled_pipeline_task_dependencies(self):
        """Indexing consumes the conditional extraction artifact after validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            compiler.Compiler().compile(
                pipeline_func=documents_indexing_pipeline,
                package_path=tmp_path,
            )
            spec = load_pipeline_spec_document(Path(tmp_path))
            tasks = spec["root"]["dag"]["tasks"]
            assert set(tasks["condition-branches-1"]["dependentTasks"]) == {
                "documents-discovery",
                "normalize-extraction-preset",
            }
            assert tasks["documents-indexing"]["dependentTasks"] == ["condition-branches-1"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_compiled_pipeline_wires_indexing_parameters(self):
        """Chunking and embedding pipeline inputs reach the indexing component."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            compiler.Compiler().compile(
                pipeline_func=documents_indexing_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "componentInputParameter: chunk_size" in content
        assert "componentInputParameter: chunk_overlap" in content
        assert "componentInputParameter: embedding_model_id" in content
        assert "componentInputParameter: collection_name" in content
        assert "componentInputParameter: preset" in content
        assert "comp-documents-indexing:" in content

    def test_compiled_pipeline_wires_ocr_lang_to_text_extraction(self):
        """ocr_lang reaches both extraction branches so indexing OCRs the corpus the way the experiment did."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            compiler.Compiler().compile(
                pipeline_func=documents_indexing_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "componentInputParameter: ocr_lang" in content

    def test_compiled_pipeline_declares_gpu_extraction_resources(self):
        """Only the gpu_accelerated extraction branch requests an NVIDIA GPU."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            compiler.Compiler().compile(
                pipeline_func=documents_indexing_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "resourceType: nvidia.com/gpu" in content
        assert "resourceCount: '1'" in content

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            (None, "speed"),
            ("", "speed"),
            ("speed", "speed"),
            ("balanced", "balanced"),
            ("gpu_accelerated", "gpu_accelerated"),
        ],
    )
    def test_normalizes_extraction_preset(self, preset, expected):
        """Missing and empty presets keep existing indexing runs at the speed CPU tier."""
        assert normalize_extraction_preset.python_func(preset=preset) == expected

    def test_rejects_invalid_extraction_preset(self):
        """Extraction preset validation names the supported values."""
        with pytest.raises(ValueError, match="balanced.*gpu_accelerated"):
            normalize_extraction_preset.python_func(preset="turbo")

    def test_compiled_pipeline_wires_s3_maas_and_vector_db_secrets(self):
        """S3 secrets attach to discovery/extraction; MaaS + vector-DB secrets attach to indexing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            compiler.Compiler().compile(
                pipeline_func=documents_indexing_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "AWS_ACCESS_KEY_ID" in content
        assert "MAAS_BASE_URL" in content
        assert "MAAS_API_KEY" in content
        # Union of vector-DB backends is mapped as optional env vars.
        assert "MILVUS_URI" in content
        assert "PGVECTOR_HOST" in content

    def test_compiled_pipeline_declares_component_resource_tiers(self):
        """All indexing pipeline steps declare the workload CPU/memory tier."""
        from kfp_components.utils.pipeline_task_resources import (
            assert_executor_resources,
            compile_executor_resources,
        )

        from .pipeline_resource_expectations import AUTORAG_INDEXING_EXECUTOR_RESOURCES

        assert_executor_resources(
            compile_executor_resources(documents_indexing_pipeline),
            AUTORAG_INDEXING_EXECUTOR_RESOURCES,
            pipeline_name="documents_indexing_pipeline",
        )

    def test_chunk_defaults_match_component(self):
        """Pipeline chunk_size/chunk_overlap defaults stay in sync with the component."""
        pipeline_sig = inspect.signature(documents_indexing_pipeline.pipeline_func)
        component_sig = inspect.signature(documents_indexing.python_func)
        for param in ("chunk_size", "chunk_overlap"):
            assert pipeline_sig.parameters[param].default == component_sig.parameters[param].default, (
                f"pipeline default for {param} ({pipeline_sig.parameters[param].default}) "
                f"diverged from component ({component_sig.parameters[param].default})"
            )
