"""Tests for sft pipeline."""

from kfp import compiler

from ..pipeline import sft_pipeline


class TestSftPipeline:
    """Basic tests for SFT pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(sft_pipeline)

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=sft_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
