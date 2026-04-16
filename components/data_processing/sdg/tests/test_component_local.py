"""Local runner tests for the sdg_hub component."""

import os
import tempfile

import pandas as pd
import pytest

from ..component import sdg

# Path to test data relative to repo root
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_INPUT_PATH = os.path.join(TEST_DATA_DIR, "sample_input.jsonl")
TEST_FLOW_PATH = os.path.join(TEST_DATA_DIR, "transform_test_flow.yaml")
LLM_TEST_FLOW_PATH = os.path.join(TEST_DATA_DIR, "llm_test_flow.yaml")


class MockArtifact:
    """Mock KFP artifact with a writable path."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.path = path
        self.metadata = {}

    def log_metric(self, metric: str, value: float):
        """Log a metric value."""
        self.metadata[metric] = value


class TestSdgHubLocalRunner:
    """Test component with real flow execution (no LLM)."""

    def test_local_execution_with_transform_flow(self, setup_and_teardown_subprocess_runner):
        """Test component execution with a transform-only flow.

        This test accepts the ``setup_and_teardown_subprocess_runner`` fixture
        for consistency with the repo-wide local-test pattern (see conftest.py),
        but calls ``python_func()`` directly instead of invoking the component
        through the KFP SubprocessRunner.  The reason is that KFP's
        SubprocessRunner does **not** support ``Input[Dataset]`` artifacts,
        which are present in the sdg component signature, and attempting to
        run the component via the runner raises
        ``ValueError: Input artifacts are not yet supported for local execution``.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_artifact = MockArtifact(os.path.join(tmp_dir, "output.jsonl"))
            output_metrics = MockArtifact(os.path.join(tmp_dir, "metrics.json"))

            sdg.python_func(
                output_artifact=output_artifact,
                output_metrics=output_metrics,
                input_pvc_path=os.path.abspath(TEST_INPUT_PATH),
                flow_yaml_path=os.path.abspath(TEST_FLOW_PATH),
                max_concurrency=1,
                checkpoint_pvc_path="",
                save_freq=100,
                log_level="INFO",
                temperature=0.7,
                max_tokens=2048,
                export_to_pvc=False,
                export_path="",
            )

            # Validate output
            assert os.path.exists(output_artifact.path), "Output artifact not created"
            output_df = pd.read_json(output_artifact.path, lines=True)
            assert len(output_df) == 3, "Expected 3 output rows"
            assert "document" in output_df.columns
            assert "domain" in output_df.columns

            # Validate metrics
            assert set(output_metrics.metadata.keys()) == {"input_rows", "output_rows", "execution_time_seconds"}


@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set - skipping LLM E2E test")
class TestSdgHubLLMFlow:
    """End-to-end tests for LLM flows with real API calls.

    These tests verify the component can execute flows containing LLM blocks
    by making actual API calls. They are skipped if LLM_API_KEY is not set
    to avoid test failures in environments without API access.
    """

    def test_llm_flow_execution(self):
        """Test that the component can run an LLM flow with real API.

        This is an end-to-end test that:
        - Loads real input data
        - Executes an LLM flow with a real model API
        - Validates output contains LLM-generated content
        - Verifies metrics are produced correctly

        Uses max_concurrency=1 to minimize API costs during testing.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create mock KFP artifacts
            output_artifact = MockArtifact(os.path.join(tmp_dir, "output.jsonl"))
            output_metrics = MockArtifact(os.path.join(tmp_dir, "metrics.json"))

            # Execute component with LLM flow
            sdg.python_func(
                output_artifact=output_artifact,
                output_metrics=output_metrics,
                input_pvc_path=os.path.abspath(TEST_INPUT_PATH),
                flow_yaml_path=os.path.abspath(LLM_TEST_FLOW_PATH),
                model="openai/gpt-4o-mini",
                max_concurrency=1,  # Minimize API costs
                temperature=0.7,
                max_tokens=2048,
                checkpoint_pvc_path="",
                save_freq=100,
                log_level="INFO",
            )

            # Validate output artifact exists and has content
            assert os.path.exists(output_artifact.path), "Output artifact file not created"
            output_df = pd.read_json(output_artifact.path, lines=True)
            assert len(output_df) > 0, "Output dataframe is empty"

            # Validate LLM blocks added generated content
            assert "extract_question_content" in output_df.columns, "LLM flow did not produce extracted content"

            # Validate questions were actually generated (not null/empty)
            assert output_df["extract_question_content"].notna().all(), "Some generated content is null"
            assert (output_df["extract_question_content"].str.len() > 0).all(), "Some generated content is empty"

            # Validate original columns are preserved
            assert "document" in output_df.columns, "Original 'document' column missing"
            assert "domain" in output_df.columns, "Original 'domain' column missing"

            # Validate metrics
            expected_metrics = {"input_rows", "output_rows", "execution_time_seconds"}
            assert set(output_metrics.metadata.keys()) == expected_metrics
            assert output_metrics.metadata["input_rows"] == 3, "Expected 3 input rows"
            assert output_metrics.metadata["output_rows"] == 3, "Expected 3 output rows"
            assert output_metrics.metadata["execution_time_seconds"] > 0, "Execution time should be positive"

    def test_llm_flow_with_invalid_model_raises_error(self):
        """Test that using an invalid model identifier raises an appropriate error.

        This validates error handling for misconfigured model parameters.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_artifact = MockArtifact(os.path.join(tmp_dir, "output.jsonl"))
            output_metrics = MockArtifact(os.path.join(tmp_dir, "metrics.json"))

            # Attempt to run with an invalid model identifier
            with pytest.raises((ValueError, RuntimeError)):  # SDG Hub raises for invalid models
                sdg.python_func(
                    output_artifact=output_artifact,
                    output_metrics=output_metrics,
                    input_pvc_path=os.path.abspath(TEST_INPUT_PATH),
                    flow_yaml_path=os.path.abspath(LLM_TEST_FLOW_PATH),
                    model="invalid/nonexistent-model-xyz",
                    max_concurrency=1,
                    temperature=0.7,
                    max_tokens=2048,
                    checkpoint_pvc_path="",
                    save_freq=100,
                    log_level="INFO",
                )

    def test_llm_flow_without_model_parameter_raises_error(self):
        """Test that LLM flows without a model parameter raise ValueError.

        This validates the component properly enforces model requirements
        for flows containing LLM blocks.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_artifact = MockArtifact(os.path.join(tmp_dir, "output.jsonl"))
            output_metrics = MockArtifact(os.path.join(tmp_dir, "metrics.json"))

            # Attempt to run LLM flow without model parameter
            with pytest.raises(ValueError, match="requires a 'model' parameter"):
                sdg.python_func(
                    output_artifact=output_artifact,
                    output_metrics=output_metrics,
                    input_pvc_path=os.path.abspath(TEST_INPUT_PATH),
                    flow_yaml_path=os.path.abspath(LLM_TEST_FLOW_PATH),
                    model="",  # Empty model string
                    max_concurrency=1,
                    temperature=0.7,
                    max_tokens=2048,
                    checkpoint_pvc_path="",
                    save_freq=100,
                    log_level="INFO",
                )
