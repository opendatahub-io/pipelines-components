"""Tests for sft_minimal pipeline."""

from kfp import compiler

from ..pipeline import sft_minimal_pipeline


class TestSftMinimalPipeline:
    """Basic tests for SFT minimal pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(sft_minimal_pipeline)

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=sft_minimal_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
