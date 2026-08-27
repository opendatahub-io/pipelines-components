"""Local runner tests for the rag_templates_optimization component."""

import pytest

from ..component import rag_templates_optimization


class TestRagTemplatesOptimizationLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    @pytest.mark.skip(reason="Requires input artifacts and model APIs; run E2E in pipeline")
    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution with LocalRunner."""
        result = rag_templates_optimization(
            extracted_text="/tmp/extracted",
            test_data="/tmp/test_data.json",
            search_space_mps_report="/tmp/report.yml",
            rag_patterns=...,
        )
        assert result is not None
