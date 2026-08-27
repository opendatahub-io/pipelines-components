"""Unit tests for the models_pre_selector component.

The component always runs and branches internally:

* **under caps** — copies the full report through unchanged, without importing
  or calling any MaaS / document / MPS machinery;
* **over caps** — restores the serialized models, runs ``ModelsPreSelector`` and
  writes a reduced report.

ai4rag is mocked via ``sys.modules`` so the pure orchestration is exercised
without the real (heavy) library or any network access.
"""

import inspect
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from ..component import models_pre_selector

MOCKED_ENV_VARIABLES = {
    "MAAS_BASE_URL": "https://maas.example.com/v1",
    "MAAS_API_KEY": "test-api-key",
}


def _make_ai4rag_mocks() -> SimpleNamespace:
    """Build mock ai4rag modules wired into sys.modules for the component body."""
    models_pre_selector_cls = mock.MagicMock(name="ModelsPreSelector")
    models_pre_selector_cls.DEFAULT_N_FOUNDATION_MODELS = 3
    models_pre_selector_cls.DEFAULT_N_EMBEDDING_MODELS = 2

    search_space_report_cls = mock.MagicMock(name="SearchSpaceReport")
    create_maas_client = mock.MagicMock(name="create_maas_client")
    load_docling_documents = mock.MagicMock(name="load_docling_documents")
    benchmark_data_cls = mock.MagicMock(name="BenchmarkData")
    get_foundation_models = mock.MagicMock(name="get_foundation_models")
    get_embedding_models = mock.MagicMock(name="get_embedding_models")
    serialize_model = mock.MagicMock(
        name="serialize_model",
        side_effect=lambda m: {"model_id": m.model_id},
    )
    ensure_sqlite3 = mock.MagicMock(name="ensure_sqlite3")

    mps_module = mock.MagicMock()
    mps_module.ModelsPreSelector = models_pre_selector_cls

    prepare_module = mock.MagicMock()
    prepare_module.SearchSpaceReport = search_space_report_cls
    prepare_module.get_foundation_models = get_foundation_models
    prepare_module.get_embedding_models = get_embedding_models
    prepare_module.serialize_model = serialize_model

    utils_module = mock.MagicMock()
    utils_module.create_maas_client = create_maas_client

    docling_module = mock.MagicMock()
    docling_module.load_docling_documents = load_docling_documents

    benchmark_module = mock.MagicMock()
    benchmark_module.BenchmarkData = benchmark_data_cls

    compat = mock.MagicMock()
    compat.ensure_sqlite3 = ensure_sqlite3

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.core": mock.MagicMock(),
        "ai4rag.core.experiment": mock.MagicMock(),
        "ai4rag.core.experiment.mps": mps_module,
        "ai4rag.core.experiment.benchmark_data": benchmark_module,
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.prepare": prepare_module,
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": utils_module,
        "ai4rag.components.utils.docling_io": docling_module,
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.compat": compat,
        "pandas": mock.MagicMock(name="pandas"),
    }
    return SimpleNamespace(
        modules=modules,
        ModelsPreSelector=models_pre_selector_cls,
        SearchSpaceReport=search_space_report_cls,
        create_maas_client=create_maas_client,
        load_docling_documents=load_docling_documents,
        get_foundation_models=get_foundation_models,
        get_embedding_models=get_embedding_models,
        serialize_model=serialize_model,
        ensure_sqlite3=ensure_sqlite3,
    )


def _write_report(path, n_foundation: int, n_embedding: int) -> dict:
    """Write a search-space report with the given model counts; return the dict."""
    report = {
        "foundation_model": [{"model_id": f"fm-{i}", "type": "generation"} for i in range(n_foundation)],
        "embedding_model": [{"model_id": f"em-{i}", "type": "embedding"} for i in range(n_embedding)],
        "chunk_size": [256, 512],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return report


def _artifacts(tmp_path, *, n_foundation: int, n_embedding: int):
    """Build input/output artifact mocks with a real input report on disk."""
    report_in = mock.MagicMock()
    report_in.path = str(tmp_path / "search_space_report.json")
    _write_report(report_in.path, n_foundation, n_embedding)

    extracted_text = mock.MagicMock()
    extracted_text.path = str(tmp_path / "extracted")
    test_data = mock.MagicMock()
    test_data.path = str(tmp_path / "test_data.json")
    report_out = mock.MagicMock()
    report_out.path = str(tmp_path / "out_report.json")
    return report_in, extracted_text, test_data, report_out


class TestModelsPreSelectorInterface:
    """Static interface checks."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(models_pre_selector)
        assert hasattr(models_pre_selector, "python_func")

    def test_component_has_expected_interface(self):
        """Component has the expected parameters and preset default."""
        sig = inspect.signature(models_pre_selector.python_func)
        params = list(sig.parameters)
        assert "search_space_report" in params
        assert "extracted_text" in params
        assert "test_data" in params
        assert "search_space_mps_report" in params
        assert "preset" in params
        assert sig.parameters["preset"].default == "speed"

    def test_component_status_defaults_to_none(self):
        """component_status defaults to None, enabling direct notebook usage."""
        sig = inspect.signature(models_pre_selector.python_func)
        assert sig.parameters["component_status"].default is None


class TestModelsPreSelectorBehaviour:
    """Branch behaviour: passthrough under caps, reduction over caps."""

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_passthrough_under_caps_skips_mps_and_maas(self, tmp_path):
        """Within caps: no MaaS client, no MPS, report written unchanged — even with no MaaS env."""
        mocks = _make_ai4rag_mocks()
        report_in, extracted_text, test_data, report_out = _artifacts(tmp_path, n_foundation=3, n_embedding=2)

        with mock.patch.dict("sys.modules", mocks.modules):
            models_pre_selector.python_func(
                search_space_report=report_in,
                extracted_text=extracted_text,
                test_data=test_data,
                search_space_mps_report=report_out,
            )

        mocks.ensure_sqlite3.assert_called_once()
        mocks.create_maas_client.assert_not_called()
        mocks.get_foundation_models.assert_not_called()
        mocks.ModelsPreSelector.assert_not_called()

        # The report is written through unchanged (3 foundation, 2 embedding).
        written = mocks.SearchSpaceReport.call_args.kwargs["search_space"]
        assert len(written["foundation_model"]) == 3
        assert len(written["embedding_model"]) == 2
        assert written["chunk_size"] == [256, 512]
        mocks.SearchSpaceReport.return_value.save_json.assert_called_once_with(report_out.path)

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_reduction_over_caps_runs_mps(self, tmp_path):
        """Over caps: restore models, run MPS, and write the reduced report."""
        mocks = _make_ai4rag_mocks()
        report_in, extracted_text, test_data, report_out = _artifacts(tmp_path, n_foundation=4, n_embedding=3)

        mocks.get_foundation_models.return_value = [mock.MagicMock(model_id=f"fm-{i}") for i in range(4)]
        mocks.get_embedding_models.return_value = [mock.MagicMock(model_id=f"em-{i}") for i in range(3)]
        selected_fm = [mock.MagicMock(model_id=f"sel-fm-{i}") for i in range(3)]
        selected_em = [mock.MagicMock(model_id=f"sel-em-{i}") for i in range(2)]
        mocks.ModelsPreSelector.return_value.select_models.return_value = {
            "foundation_models": selected_fm,
            "embedding_models": selected_em,
        }

        with mock.patch.dict("sys.modules", mocks.modules):
            models_pre_selector.python_func(
                search_space_report=report_in,
                extracted_text=extracted_text,
                test_data=test_data,
                search_space_mps_report=report_out,
            )

        mocks.create_maas_client.assert_called_once_with(
            base_url="https://maas.example.com/v1",
            api_key="test-api-key",
        )
        mocks.get_foundation_models.assert_called_once()
        mocks.get_embedding_models.assert_called_once()
        mocks.ModelsPreSelector.return_value.evaluate_patterns.assert_called_once()
        mocks.ModelsPreSelector.return_value.select_models.assert_called_once_with(
            n_foundation_models=3,
            n_embedding_models=2,
        )

        written = mocks.SearchSpaceReport.call_args.kwargs["search_space"]
        assert [m["model_id"] for m in written["foundation_model"]] == ["sel-fm-0", "sel-fm-1", "sel-fm-2"]
        assert [m["model_id"] for m in written["embedding_model"]] == ["sel-em-0", "sel-em-1"]
        # Non-model dimensions are preserved untouched.
        assert written["chunk_size"] == [256, 512]
        mocks.SearchSpaceReport.return_value.save_json.assert_called_once_with(report_out.path)

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_reduction_triggered_when_only_embeddings_exceed_cap(self, tmp_path):
        """Exceeding only the embedding cap still triggers pre-selection."""
        mocks = _make_ai4rag_mocks()
        report_in, extracted_text, test_data, report_out = _artifacts(tmp_path, n_foundation=2, n_embedding=3)
        mocks.get_foundation_models.return_value = [mock.MagicMock(model_id=f"fm-{i}") for i in range(2)]
        mocks.get_embedding_models.return_value = [mock.MagicMock(model_id=f"em-{i}") for i in range(3)]
        mocks.ModelsPreSelector.return_value.select_models.return_value = {
            "foundation_models": [mock.MagicMock(model_id="fm-0"), mock.MagicMock(model_id="fm-1")],
            "embedding_models": [mock.MagicMock(model_id="em-0"), mock.MagicMock(model_id="em-1")],
        }

        with mock.patch.dict("sys.modules", mocks.modules):
            models_pre_selector.python_func(
                search_space_report=report_in,
                extracted_text=extracted_text,
                test_data=test_data,
                search_space_mps_report=report_out,
            )

        mocks.ModelsPreSelector.return_value.evaluate_patterns.assert_called_once()

    def test_preset_validation_rejects_invalid(self, tmp_path):
        """An invalid preset raises before any report work."""
        mocks = _make_ai4rag_mocks()
        report_in, extracted_text, test_data, report_out = _artifacts(tmp_path, n_foundation=2, n_embedding=1)

        with mock.patch.dict("sys.modules", mocks.modules):
            with pytest.raises(ValueError, match="preset must be one of"):
                models_pre_selector.python_func(
                    search_space_report=report_in,
                    extracted_text=extracted_text,
                    test_data=test_data,
                    search_space_mps_report=report_out,
                    preset="invalid",
                )
