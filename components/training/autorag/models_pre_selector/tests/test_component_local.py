"""Local runner tests for the models_pre_selector component."""

import pytest

from ..component import models_pre_selector


class TestModelsPreSelectorLocalRunner:
    """Test component with LocalRunner (subprocess execution)."""

    @pytest.mark.skip(reason="Requires input artifacts and model APIs; run E2E in pipeline")
    def test_local_execution(self, setup_and_teardown_subprocess_runner):  # noqa: F811
        """Test component execution with LocalRunner."""
        result = models_pre_selector(
            search_space_report=...,
            extracted_text=...,
            test_data=...,
            search_space_mps_report=...,
        )
        assert result is not None
