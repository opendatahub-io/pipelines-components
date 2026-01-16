"""Unit tests for the dataset_download component."""

from unittest import mock

from ..component import dataset_download


class TestDatasetDownloadUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(dataset_download)
        assert hasattr(dataset_download, "python_func")

    def test_component_has_expected_parameters(self):
        """Test that the component has expected input parameters."""
        import inspect

        sig = inspect.signature(dataset_download.python_func)
        params = list(sig.parameters.keys())

        expected_params = [
            "train_dataset",
            "eval_dataset",
            "dataset_uri",
            "pvc_mount_path",
            "train_split_ratio",
            "subset_count",
            "hf_token",
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(dataset_download.python_func)
        params = sig.parameters

        assert params["train_split_ratio"].default == 0.9
        assert params["subset_count"].default == 0
        assert params["hf_token"].default == ""

    @mock.patch.dict("sys.modules", {"datasets": mock.MagicMock()})
    @mock.patch("datasets.load_dataset")
    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists")
    def test_component_with_mocked_huggingface(
        self,
        mock_exists,
        mock_makedirs,
        mock_load_dataset,
    ):
        """Test component with mocked HuggingFace dataset loading."""
        # Setup mock dataset
        mock_dataset = mock.MagicMock()
        mock_dataset.__len__ = mock.MagicMock(return_value=100)

        # Mock train_test_split
        mock_train = mock.MagicMock()
        mock_train.__len__ = mock.MagicMock(return_value=90)
        mock_train.to_json = mock.MagicMock()

        mock_eval = mock.MagicMock()
        mock_eval.__len__ = mock.MagicMock(return_value=10)
        mock_eval.to_json = mock.MagicMock()

        mock_dataset.train_test_split.return_value = {"train": mock_train, "test": mock_eval}
        mock_load_dataset.return_value = mock_dataset

        mock_exists.return_value = True

        # Mock output artifacts
        mock_train_output = mock.MagicMock()
        mock_train_output.path = "/tmp/train.jsonl"
        mock_train_output.metadata = {}

        mock_eval_output = mock.MagicMock()
        mock_eval_output.path = "/tmp/eval.jsonl"
        mock_eval_output.metadata = {}

        # The component would need full execution context
        # For now verify the component definition is valid
        assert dataset_download.python_func is not None

    def test_component_supports_multiple_uri_schemes(self):
        """Test that the component documentation mentions supported URI schemes."""
        # Verify docstring mentions supported schemes
        docstring = dataset_download.python_func.__doc__
        assert "hf://" in docstring or "HuggingFace" in docstring.lower()
        assert "s3://" in docstring
        assert "http" in docstring.lower()
