"""Unit tests for the sdg_hub component."""

import os
import tempfile
from unittest import mock

import pandas as pd
import pytest

from ..component import sdg


class MockArtifact:
    """Mock KFP artifact with a writable path."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.path = path
        self.metadata = {}

    def log_metric(self, metric: str, value: float):
        """Log a metric value."""
        self.metadata[metric] = value


@pytest.fixture
def tmp_dir():
    """Create temp directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def output_artifact(tmp_dir):
    """Create mock output artifact."""
    return MockArtifact(os.path.join(tmp_dir, "output.jsonl"))


@pytest.fixture
def output_metrics(tmp_dir):
    """Create mock output metrics artifact."""
    return MockArtifact(os.path.join(tmp_dir, "metrics.json"))


@pytest.fixture
def sample_input_file(tmp_dir):
    """Create sample JSONL input file."""
    path = os.path.join(tmp_dir, "input.jsonl")
    df = pd.DataFrame(
        {
            "document": ["Doc one.", "Doc two.", "Doc three."],
            "domain": ["science", "tech", "math"],
        }
    )
    df.to_json(path, orient="records", lines=True)
    return path


def _make_mock_flow(return_df=None):
    """Create a mock Flow that passes through input data."""
    mock_flow = mock.MagicMock()
    mock_flow.metadata.name = "mock-flow"
    mock_flow.metadata.version = "1.0.0"
    mock_flow.blocks = []
    mock_flow.is_model_config_required.return_value = False
    mock_flow.validate_dataset.return_value = []
    if return_df is not None:
        mock_flow.generate.return_value = return_df
    else:
        mock_flow.generate.side_effect = lambda df, **kw: df.copy()
    return mock_flow


def _call_sdg(output_artifact, output_metrics, **kwargs):
    """Helper to call the component's python_func with defaults."""
    defaults = {
        "input_artifact": None,
        "input_pvc_path": "",
        "flow_id": "",
        "flow_yaml_path": "",
        "model": "",
        "max_concurrency": 10,
        "checkpoint_pvc_path": "",
        "save_freq": 100,
        "log_level": "INFO",
        "temperature": -1.0,
        "max_tokens": -1,
        "export_to_pvc": False,
        "export_path": "",
        "runtime_params": None,
    }
    defaults.update(kwargs)
    sdg.python_func(
        output_artifact=output_artifact,
        output_metrics=output_metrics,
        **defaults,
    )


class TestComponentExists:
    """Basic component validation."""

    def test_component_function_exists(self):
        """Test that component function exists and is callable."""
        assert callable(sdg)
        assert hasattr(sdg, "python_func")


class TestInputHandling:
    """Tests for input loading."""

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_load_jsonl_input(self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file):
        """Test that JSONL input file is loaded correctly."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="test-flow")

        result = pd.read_json(output_artifact.path, lines=True)
        assert len(result) == 3
        assert list(result.columns) == ["document", "domain"]

    def test_missing_input_file_raises(self, output_artifact, output_metrics):
        """Test that missing input file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            _call_sdg(output_artifact, output_metrics, input_pvc_path="/nonexistent/data.jsonl", flow_id="test-flow")

    def test_no_input_raises(self, output_artifact, output_metrics):
        """Test that missing input raises ValueError."""
        with pytest.raises(ValueError, match="No input provided"):
            _call_sdg(output_artifact, output_metrics, input_artifact=None, input_pvc_path="", flow_id="test-flow")


class TestInputArtifact:
    """Tests for input_artifact handling."""

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_artifact_reads_jsonl(self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, tmp_dir):
        """Verify input_artifact is correctly loaded from JSONL file."""
        artifact_path = os.path.join(tmp_dir, "artifact_input.jsonl")
        df = pd.DataFrame(
            {
                "document": ["Artifact doc one.", "Artifact doc two."],
                "domain": ["physics", "chemistry"],
            }
        )
        df.to_json(artifact_path, orient="records", lines=True)

        input_artifact = MockArtifact(artifact_path)
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(output_artifact, output_metrics, input_artifact=input_artifact, flow_id="test-flow")

        result = pd.read_json(output_artifact.path, lines=True)
        assert len(result) == 2
        assert list(result.columns) == ["document", "domain"]
        assert result["document"].tolist() == ["Artifact doc one.", "Artifact doc two."]

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_artifact_takes_priority_over_pvc(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, tmp_dir
    ):
        """Verify input_artifact takes precedence when both artifact and PVC path are provided."""
        artifact_path = os.path.join(tmp_dir, "artifact_data.jsonl")
        pvc_path = os.path.join(tmp_dir, "pvc_data.jsonl")

        artifact_df = pd.DataFrame({"source": ["artifact"], "value": [100]})
        pvc_df = pd.DataFrame({"source": ["pvc"], "value": [200]})

        artifact_df.to_json(artifact_path, orient="records", lines=True)
        pvc_df.to_json(pvc_path, orient="records", lines=True)

        input_artifact = MockArtifact(artifact_path)
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_artifact=input_artifact,
            input_pvc_path=pvc_path,
            flow_id="test-flow",
        )

        result = pd.read_json(output_artifact.path, lines=True)
        assert result["source"].tolist() == ["artifact"]
        assert result["value"].tolist() == [100]


class TestFlowSelection:
    """Tests for flow selection logic."""

    def test_no_flow_specified_raises(self, output_artifact, output_metrics, sample_input_file):
        """Test that missing flow specification raises ValueError."""
        with pytest.raises(ValueError, match="Either 'flow_id' or 'flow_yaml_path'"):
            _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="", flow_yaml_path="")

    def test_flow_yaml_path_not_found_raises(self, output_artifact, output_metrics, sample_input_file):
        """Test that missing flow YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Custom flow YAML not found"):
            _call_sdg(
                output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_yaml_path="/nonexistent.yaml"
            )

    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_invalid_flow_id_raises(self, mock_get_path, output_artifact, output_metrics, sample_input_file):
        """Test that invalid flow ID raises ValueError."""
        mock_get_path.side_effect = ValueError("Flow 'bad-id' not found.")
        with pytest.raises(ValueError, match="Flow lookup failed"):
            _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="bad-id")

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_flow_id_resolves_and_loads(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that flow ID resolves to path and loads flow."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="test-flow-id")

        mock_get_path.assert_called_once_with("test-flow-id")
        mock_from_yaml.assert_called_once_with("/resolved/flow.yaml")

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    def test_flow_yaml_path_loads_directly(
        self, mock_from_yaml, output_artifact, output_metrics, sample_input_file, tmp_dir
    ):
        """Test that flow YAML path loads flow directly."""
        yaml_path = os.path.join(tmp_dir, "custom_flow.yaml")
        with open(yaml_path, "w") as f:
            f.write("dummy")

        mock_from_yaml.return_value = _make_mock_flow()
        _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_yaml_path=yaml_path)
        mock_from_yaml.assert_called_once_with(yaml_path)

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    def test_flow_yaml_path_takes_precedence(
        self, mock_from_yaml, output_artifact, output_metrics, sample_input_file, tmp_dir
    ):
        """Test that flow YAML path takes precedence over flow ID."""
        yaml_path = os.path.join(tmp_dir, "custom_flow.yaml")
        with open(yaml_path, "w") as f:
            f.write("dummy")

        mock_from_yaml.return_value = _make_mock_flow()
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="some-id",
            flow_yaml_path=yaml_path,
        )
        mock_from_yaml.assert_called_once_with(yaml_path)


class TestModelConfiguration:
    """Tests for model configuration logic."""

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_llm_flow_without_model_raises(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that LLM flow without model parameter raises ValueError."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = True
        mock_from_yaml.return_value = mock_flow

        with pytest.raises(ValueError, match="requires a 'model' parameter"):
            _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="llm-flow", model="")

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_llm_flow_with_model_configures(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that LLM flow with model parameter configures model correctly."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = True
        mock_from_yaml.return_value = mock_flow

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="llm-flow",
            model="openai/gpt-4o-mini",
            temperature=0.5,
            max_tokens=1024,
        )

        mock_flow.set_model_config.assert_called_once()
        call_kwargs = mock_flow.set_model_config.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 1024

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_non_llm_flow_skips_model_config(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that non-LLM flow skips model configuration."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = False
        mock_from_yaml.return_value = mock_flow

        _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="transform-id")
        mock_flow.set_model_config.assert_not_called()

    @mock.patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_API_BASE": "http://localhost:8080/v1"})
    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_env_credentials_passed(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that environment credentials are passed to model config."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = True
        mock_from_yaml.return_value = mock_flow

        _call_sdg(
            output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="llm-flow", model="openai/gpt-4"
        )

        call_kwargs = mock_flow.set_model_config.call_args
        assert call_kwargs.kwargs["api_key"] == "test-key"
        assert call_kwargs.kwargs["api_base"] == "http://localhost:8080/v1"

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_default_sentinel_excludes_model_kwargs(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that sentinel defaults (-1) exclude temperature/max_tokens from model config."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = True
        mock_from_yaml.return_value = mock_flow

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="llm-flow",
            model="openai/gpt-4o-mini",
            temperature=-1.0,
            max_tokens=-1,
        )

        mock_flow.set_model_config.assert_called_once()
        call_kwargs = mock_flow.set_model_config.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert "temperature" not in call_kwargs
        assert "max_tokens" not in call_kwargs


class TestFlowExecution:
    """Tests for flow execution."""

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_generate_called_with_correct_params(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that flow.generate is called with correct parameters."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="test-flow", max_concurrency=20
        )

        mock_flow = mock_from_yaml.return_value
        mock_flow.generate.assert_called_once()
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert call_kwargs["max_concurrency"] == 20

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_checkpointing_params_passed(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that checkpointing parameters are passed correctly."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
            checkpoint_pvc_path="/mnt/checkpoints/",
            save_freq=50,
        )

        mock_flow = mock_from_yaml.return_value
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert call_kwargs["checkpoint_dir"] == "/mnt/checkpoints/"
        assert call_kwargs["save_freq"] == 50

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_no_checkpoint_when_not_configured(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that checkpoint parameters are not passed when not configured."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
            checkpoint_pvc_path="",
        )

        mock_flow = mock_from_yaml.return_value
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert "checkpoint_dir" not in call_kwargs

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_output_reflects_flow_result(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that output artifact reflects flow generation result."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        enriched_df = pd.DataFrame(
            {"document": ["Doc one."], "domain": ["science"], "generated_qa": ["Q: What? A: Something."]}
        )
        mock_from_yaml.return_value = _make_mock_flow(return_df=enriched_df)

        _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="test-flow")

        result = pd.read_json(output_artifact.path, lines=True)
        assert "generated_qa" in result.columns
        assert len(result) == 1


class TestPVCExport:
    """Tests for PVC export functionality."""

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_export_creates_file(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file, tmp_dir
    ):
        """Verify export_to_pvc creates a timestamped JSONL file in the export directory."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        export_base = os.path.join(tmp_dir, "exports")

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
            export_to_pvc=True,
            export_path=export_base,
        )

        flow_dir = os.path.join(export_base, "test-flow")
        assert os.path.exists(flow_dir), "Flow directory should exist"

        timestamp_dirs = os.listdir(flow_dir)
        assert len(timestamp_dirs) == 1, "Should have exactly one timestamp directory"

        timestamp_dir = timestamp_dirs[0]
        export_file = os.path.join(flow_dir, timestamp_dir, "generated.jsonl")
        assert os.path.exists(export_file), "generated.jsonl should exist"

        exported_df = pd.read_json(export_file, lines=True)
        assert len(exported_df) == 3
        assert list(exported_df.columns) == ["document", "domain"]

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_export_disabled_no_write(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file, tmp_dir
    ):
        """Verify no export files are created when export_to_pvc is False."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        export_base = os.path.join(tmp_dir, "exports")

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
            export_to_pvc=False,
            export_path=export_base,
        )

        assert not os.path.exists(export_base), "Export directory should not be created when export is disabled"

        files_in_tmp = set(os.listdir(tmp_dir))
        expected_files = {"output.jsonl", "input.jsonl"}
        assert files_in_tmp == expected_files, "Only artifact and input files should exist"

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_export_without_path_raises(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Verify ValueError is raised when export_to_pvc is True but export_path is empty."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        with pytest.raises(ValueError, match="export_to_pvc is True but export_path is not provided"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path=sample_input_file,
                flow_id="test-flow",
                export_to_pvc=True,
                export_path="",
            )

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    def test_export_custom_flow_uses_yaml_basename(
        self, mock_from_yaml, output_artifact, output_metrics, sample_input_file, tmp_dir
    ):
        """Verify custom flow YAML path uses the YAML filename as export directory name."""
        yaml_path = os.path.join(tmp_dir, "custom_flow.yaml")
        with open(yaml_path, "w") as f:
            f.write("dummy yaml content")

        mock_from_yaml.return_value = _make_mock_flow()

        export_base = os.path.join(tmp_dir, "exports")

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_yaml_path=yaml_path,
            export_to_pvc=True,
            export_path=export_base,
        )

        custom_dir = os.path.join(export_base, "custom_flow")
        assert os.path.exists(custom_dir), "Export should use YAML basename as flow name"

        timestamp_dirs = os.listdir(custom_dir)
        assert len(timestamp_dirs) == 1, "Should have exactly one timestamp directory"

        timestamp_dir = timestamp_dirs[0]
        export_file = os.path.join(custom_dir, timestamp_dir, "generated.jsonl")
        assert os.path.exists(export_file), "generated.jsonl should exist in custom directory"


class TestRuntimeParams:
    """Tests for runtime_params per-block overrides."""

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_runtime_params_passed_to_generate(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that non-empty runtime_params is passed to flow.generate()."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        params = {"generate_question": {"temperature": 0.9}}
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
            runtime_params=params,
        )

        mock_flow = mock_from_yaml.return_value
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert call_kwargs["runtime_params"] == params

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_none_runtime_params_not_passed(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that None runtime_params is not passed to flow.generate()."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
            runtime_params=None,
        )

        mock_flow = mock_from_yaml.return_value
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert "runtime_params" not in call_kwargs


class TestOutputHandling:
    """Tests for output artifacts and metrics."""

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_metrics_written(self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file):
        """Test that metrics artifact is written with expected metrics."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="test-flow")

        assert set(output_metrics.metadata.keys()) == {"input_rows", "output_rows", "execution_time_seconds"}

    @mock.patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @mock.patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_metrics_row_counts(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics, sample_input_file
    ):
        """Test that metrics contain correct row counts."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(output_artifact, output_metrics, input_pvc_path=sample_input_file, flow_id="test-flow")

        assert output_metrics.metadata["input_rows"] == 3
        assert output_metrics.metadata["output_rows"] == 3
