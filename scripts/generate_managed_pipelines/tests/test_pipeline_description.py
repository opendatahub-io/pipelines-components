"""Tests for pipeline_description module."""

from pathlib import Path

from ..pipeline_description import extract_pipeline_description_from_file


class TestExtractPipelineDescriptionFromFile:
    """Tests for extract_pipeline_description_from_file."""

    def test_description_keyword_literal(self, tmp_path: Path):
        """Extract description from @dsl.pipeline(description='...')."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text(
            '''from kfp import dsl

@dsl.pipeline(name="p", description="Hello world")
def my_pipeline():
    """Doc ignored when decorator has description."""
    pass
'''
        )
        assert extract_pipeline_description_from_file(pipeline_py, function_name="my_pipeline") == "Hello world"

    def test_description_implicit_string_concat(self, tmp_path: Path):
        """Extract description built from implicit string concatenation."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text(
            '''from kfp import dsl

@dsl.pipeline(
    name="p",
    description=(
        "Part one "
        "part two"
    ),
)
def my_pipeline():
    pass
'''
        )
        assert extract_pipeline_description_from_file(pipeline_py) == "Part one part two"

    def test_bare_pipeline_uses_docstring_first_line(self, tmp_path: Path):
        """@dsl.pipeline without call falls back to first docstring line."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text(
            '''from kfp import dsl

@dsl.pipeline
def my_pipeline():
    """First line summary.

    More text.
    """
    pass
'''
        )
        assert extract_pipeline_description_from_file(pipeline_py) == "First line summary."

    def test_prefers_named_function(self, tmp_path: Path):
        """When function_name is set, use that pipeline function."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text(
            '''from kfp import dsl

@dsl.pipeline(description="first")
def first_pipeline():
    pass

@dsl.pipeline(description="second")
def second_pipeline():
    pass
'''
        )
        assert extract_pipeline_description_from_file(pipeline_py, function_name="second_pipeline") == "second"
