"""Tests for the tabular_data_loader component.

boto3, pandas, and sklearn are mocked via sys.modules so the real packages are not required.
Tests use the stdlib csv module for asserting on output CSV content.
"""

import csv
import io
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from ..component import automl_data_loader
from .mocked_pandas import (
    MockedDataFrame,
    _mock_train_test_split,
    make_mocked_pandas_module,
    make_mocked_sklearn_module,
)

TAXI_TRIP_PRICING_SAMPLE = Path(__file__).resolve().parent / "data" / "taxi_trip_pricing_sample.csv"


def _count_rows_with_non_empty_trip_price(csv_path: Path) -> int:
    """Match mocked CSV parsing: empty ``Trip_Price`` cell is a blank last field."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = header.index("Trip_Price")
        return sum(1 for row in reader if len(row) > idx and row[idx].strip() != "")


mocked_env_variables = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "test_url",
}


class _MockSSLError(Exception):
    """Stand-in for botocore.exceptions.SSLError used in unit tests."""

    pass


@contextmanager
def _mock_boto3_module(get_object_return=None, get_object_side_effect=None):
    """Inject a fake boto3 module so the component does not require boto3 to be installed."""
    mock_boto3 = mock.MagicMock()
    mock_s3 = mock.MagicMock()
    if get_object_side_effect is not None:
        mock_s3.get_object.side_effect = get_object_side_effect
    else:
        mock_s3.get_object.return_value = get_object_return or {"Body": io.BytesIO(b"")}
    mock_boto3.client.return_value = mock_s3

    # Inject botocore.exceptions so `from botocore.exceptions import SSLError` works
    mock_botocore = mock.MagicMock()
    mock_botocore_exceptions = mock.MagicMock()
    mock_botocore_exceptions.SSLError = _MockSSLError
    mock_botocore.exceptions = mock_botocore_exceptions

    with mock.patch.dict(
        sys.modules,
        {
            "boto3": mock_boto3,
            "botocore": mock_botocore,
            "botocore.exceptions": mock_botocore_exceptions,
        },
    ):
        yield mock_s3


@contextmanager
def _mock_boto3_and_pandas(get_object_return=None, get_object_side_effect=None):
    """Inject mocked boto3, pandas, and sklearn so the component runs without any dependency."""
    mocked_pandas = make_mocked_pandas_module()
    mock_sklearn, mock_model_selection = make_mocked_sklearn_module()
    with _mock_boto3_module(
        get_object_return=get_object_return, get_object_side_effect=get_object_side_effect
    ) as mock_s3:
        with mock.patch.dict(
            sys.modules,
            {
                "pandas": mocked_pandas,
                "sklearn": mock_sklearn,
                "sklearn.model_selection": mock_model_selection,
            },
        ):
            yield mock_s3


@contextmanager
def _mock_boto3_pandas_custom_train_test_split(
    train_test_split_impl,
    *,
    get_object_return=None,
    get_object_side_effect=None,
):
    """Like ``_mock_boto3_and_pandas`` but with a custom ``train_test_split`` (for split-failure tests)."""
    mocked_pandas = make_mocked_pandas_module()
    import types

    mock_sklearn = types.ModuleType("sklearn")
    mock_model_selection = types.ModuleType("sklearn.model_selection")
    mock_model_selection.train_test_split = train_test_split_impl
    mock_sklearn.model_selection = mock_model_selection
    with _mock_boto3_module(
        get_object_return=get_object_return,
        get_object_side_effect=get_object_side_effect,
    ) as mock_s3:
        with mock.patch.dict(
            sys.modules,
            {
                "pandas": mocked_pandas,
                "sklearn": mock_sklearn,
                "sklearn.model_selection": mock_model_selection,
            },
        ):
            yield mock_s3


def _read_csv_path(path):
    """Read a CSV file with stdlib csv; return (headers, list of rows)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    return header, rows


def _make_test_artifact(tmp_path, name="test_output.csv"):
    """Create a mock artifact with .path and .uri for sampled_test_dataset."""
    art = mock.MagicMock()
    art.path = str(tmp_path / name)
    art.uri = "/artifacts/test"
    return art


class TestAutomlDataLoaderUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(automl_data_loader)
        assert hasattr(automl_data_loader, "python_func")

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_with_default_parameters(self, tmp_path):
        """Test component with default sampling_method=None (resolved from task_type=regression -> random)."""
        csv_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}) as mock_s3:
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                label_column="c",
                sampled_test_dataset=sampled_test,
            )

            assert result is not None
            assert hasattr(result, "sample_config")
            assert result.sample_config["n_samples"] == 3
            mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/file.csv")

        # Verify split outputs exist
        assert (tmp_path / "datasets" / "models_selection_train_dataset.csv").exists()
        assert (tmp_path / "datasets" / "extra_train_dataset.csv").exists()

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_explicit_first_n_rows(self, tmp_path):
        """Test component with explicit sampling_method='first_n_rows'."""
        csv_content = "x,y,z\n10,20,30\n40,50,60\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="s3/path/data.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="z",
                sampled_test_dataset=sampled_test,
                sampling_method="first_n_rows",
            )

            assert hasattr(result, "sample_config")
            assert result.sample_config["n_samples"] == 2

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_stratified_sampling_with_label_column(self, tmp_path):
        """Test component with sampling_method='stratified' and label_column."""
        csv_content = "feature1,feature2,target\n1,2,A\n2,3,A\n3,4,A\n4,5,B\n5,6,B\n6,7,B\n7,8,C\n8,9,C\n9,10,C\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}) as mock_s3:
            result = automl_data_loader.python_func(
                file_key="data/train.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                sampling_method="stratified",
                label_column="target",
                task_type="multiclass",
                sampled_test_dataset=sampled_test,
            )

            assert hasattr(result, "sample_config")
            assert result.sample_config["n_samples"] == 9
            mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/train.csv")
        assert (tmp_path / "datasets" / "models_selection_train_dataset.csv").exists()

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_stratified_requires_label_column(self, tmp_path):
        """Test that sampling_method='stratified' without label_column raises ValueError."""
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas() as mock_s3:
            with pytest.raises(TypeError, match="label_column must be a non-empty string"):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    sampling_method="stratified",
                    label_column=None,
                    task_type="binary",
                    sampled_test_dataset=sampled_test,
                )

            mock_s3.get_object.assert_not_called()

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_stratified_label_column_not_in_dataset(self, tmp_path):
        """Test that stratified sampling with missing target column raises ValueError."""
        csv_content = "a,b,c\n1,2,3\n4,5,6\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            with pytest.raises(ValueError, match=r"Error reading CSV from S3"):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    sampling_method="stratified",
                    label_column="label",
                    task_type="binary",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_stratified_drops_na_in_target(self, tmp_path):
        """Test that stratified sampling drops rows with NA in label_column."""
        csv_content = "f1,f2,target\n1,2,A\n2,3,\n3,4,B\n4,5,B\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                sampling_method="stratified",
                label_column="target",
                task_type="binary",
                sampled_test_dataset=sampled_test,
            )

            assert hasattr(result, "sample_config")
            assert result.sample_config["n_samples"] >= 2

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_regression_drops_missing_label_before_split(self, tmp_path):
        """Regression (random sampling) must drop rows with empty label before splitting."""
        csv_content = "a,b,target\n1,2,10\n3,4,\n5,6,30\n7,8,40\n9,10,50\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        assert result.sample_config["n_samples"] == 4
        _, test_rows = _read_csv_path(sampled_test.path)
        target_idx = 2
        for row in test_rows:
            assert row[target_idx] != "" and row[target_idx] is not None

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_raises_when_all_labels_missing(self, tmp_path):
        """If every row has a missing label, fail with a clear error."""
        csv_content = "a,b,target\n1,2,\n3,4,\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            with pytest.raises(ValueError, match="No rows remain after removing missing values"):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                    task_type="regression",
                )

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_taxi_trip_pricing_sample_regression_row_count(self, tmp_path):
        """Subset of taxi_trip_pricing.csv: rows with empty Trip_Price are excluded (regression path)."""
        assert TAXI_TRIP_PRICING_SAMPLE.is_file()
        body_stream = io.BytesIO(TAXI_TRIP_PRICING_SAMPLE.read_bytes())
        sampled_test = _make_test_artifact(tmp_path)
        expected_rows = _count_rows_with_non_empty_trip_price(TAXI_TRIP_PRICING_SAMPLE)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/taxi_trip_pricing_sample.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="Trip_Price",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        assert result.sample_config["n_samples"] == expected_rows

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_taxi_trip_pricing_sample_output_has_no_empty_trip_price(self, tmp_path):
        """Written train/test CSVs must not contain blank Trip_Price."""
        body_stream = io.BytesIO(TAXI_TRIP_PRICING_SAMPLE.read_bytes())
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            automl_data_loader.python_func(
                file_key="data/taxi_trip_pricing_sample.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="Trip_Price",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        price_idx = None
        for path in (
            sampled_test.path,
            tmp_path / "datasets" / "models_selection_train_dataset.csv",
            tmp_path / "datasets" / "extra_train_dataset.csv",
        ):
            header, rows = _read_csv_path(path)
            if price_idx is None:
                price_idx = header.index("Trip_Price")
            for row in rows:
                assert len(row) > price_idx
                assert row[price_idx].strip() != "", f"empty Trip_Price in {path}"

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_taxi_schema_all_missing_trip_price_raises(self, tmp_path):
        """Same schema as taxi pricing, but every Trip_Price empty → clear ValueError."""
        header = (
            "Trip_Distance_km,Time_of_Day,Day_of_Week,Passenger_Count,Traffic_Conditions,"
            "Weather,Base_Fare,Per_Km_Rate,Per_Minute_Rate,Trip_Duration_Minutes,Trip_Price\n"
        )
        row = "1.0,Morning,Weekday,1.0,Low,Clear,1.0,1.0,0.1,10.0,\n"
        body_stream = io.BytesIO((header + row + row).encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            with pytest.raises(ValueError, match="No rows remain after removing missing values"):
                automl_data_loader.python_func(
                    file_key="data/bad.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="Trip_Price",
                    sampled_test_dataset=sampled_test,
                    task_type="regression",
                )

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_drop_full_row_duplicates_before_split(self, tmp_path):
        """Identical feature+label rows are deduplicated before train/test split."""
        csv_content = "a,b,target\n1,2,10\n1,2,10\n3,4,20\n5,6,30\n7,8,40\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        assert result.sample_config["n_samples"] == 4

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_infinity_in_label_replaced_then_dropped_regression(self, tmp_path):
        """±infinity in the label column becomes NaN and is dropped with other missing labels."""
        csv_content = "a,b,target\n1,2,10\n3,4,20\n5,6,inf\n7,8,40\n9,10,50\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        assert result.sample_config["n_samples"] == 4

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_infinity_in_feature_only_row_retained_regression(self, tmp_path):
        """Infinity in a non-label column is replaced with NaN but the row stays if the label is valid."""
        csv_content = "a,b,target\n1,inf,10\n3,4,20\n5,6,30\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        assert result.sample_config["n_samples"] == 3

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_rows_equal_after_inf_replace_are_deduplicated(self, tmp_path):
        """Rows that differ only by opposite infinities in a feature collapse to one row after replace."""
        csv_content = "a,b,target\n1,inf,10\n1,-inf,10\n3,4,20\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        assert result.sample_config["n_samples"] == 2

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_raises_when_all_labels_are_infinite(self, tmp_path):
        """If every label is ±inf, after replace all labels are missing and the component fails clearly."""
        csv_content = "a,b,target\n1,2,inf\n3,4,-inf\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            with pytest.raises(ValueError, match="No rows remain after removing missing values"):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                    task_type="regression",
                )

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_first_n_rows_sampling_drops_missing_label_before_split(self, tmp_path):
        """``first_n_rows`` must drop missing labels before split (same as random regression path)."""
        csv_content = "a,b,target\n1,2,10\n3,4,\n5,6,30\n7,8,40\n9,10,50\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                sampling_method="first_n_rows",
                task_type="regression",
            )

        assert result.sample_config["n_samples"] == 4

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_secondary_train_test_split_failure_propagates(self, tmp_path):
        """If sklearn rejects the secondary split, the error is not swallowed (AutoAI-style holdout edge)."""
        calls = {"n": 0}

        def train_test_split_impl(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise ValueError(
                    "With n_samples=4, test_size=0.7 and stratify=False, the resulting train set would be empty."
                )
            return _mock_train_test_split(*args, **kwargs)

        csv_content = "a,b,target\n1,2,10\n3,4,20\n5,6,30\n7,8,40\n9,10,50\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_pandas_custom_train_test_split(
            train_test_split_impl,
            get_object_return={"Body": body_stream},
        ):
            with pytest.raises(ValueError, match="resulting train set would be empty"):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                    task_type="regression",
                )

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_stratified_train_test_split_rejection_propagates(self, tmp_path):
        """Classification stratify failures (e.g. too few per class) surface from train_test_split."""

        def train_test_split_impl(*args, **kwargs):
            if kwargs.get("stratify") is not None:
                raise ValueError(
                    "The least populated class in y has only 1 member, which is too few for stratification. "
                    "Minimum 2 members are required for each class."
                )
            return _mock_train_test_split(*args, **kwargs)

        csv_content = "f1,f2,target\n1,2,A\n3,4,B\n5,6,A\n7,8,B\n9,10,A\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_pandas_custom_train_test_split(
            train_test_split_impl,
            get_object_return={"Body": body_stream},
        ):
            with pytest.raises(ValueError, match="least populated class"):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                    task_type="binary",
                )

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_random_sampling_basic(self, tmp_path):
        """Test component with sampling_method='random' writes valid CSV and returns sample_config."""
        csv_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}) as mock_s3:
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                label_column="c",
                sampled_test_dataset=sampled_test,
                sampling_method="random",
            )

            assert result.sample_config["n_samples"] == 4
            mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/file.csv")
        assert (tmp_path / "datasets" / "models_selection_train_dataset.csv").exists()
        assert (tmp_path / "datasets" / "extra_train_dataset.csv").exists()

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_random_sampling_deterministic(self, tmp_path):
        """Test that random sampling with fixed random_state is reproducible.

        Use a large BYTES_PER_ROW so the mock reports >100MB for few rows, triggering
        _sample_random's downsampling. Otherwise no sample() call runs and the test
        would trivially pass without exercising the seed logic.
        """
        csv_content = "x,y\n1,2\n3,4\n5,6\n7,8\n9,10\n"

        def get_object(**kwargs):
            return {"Body": io.BytesIO(csv_content.encode("utf-8"))}

        original_bytes_per_row = MockedDataFrame.BYTES_PER_ROW
        try:
            # 5 rows * 50M bytes/row = 250MB > 100MB limit -> triggers random downsampling
            MockedDataFrame.BYTES_PER_ROW = 50_000_000

            with _mock_boto3_and_pandas(get_object_side_effect=get_object):
                sampled_test1 = _make_test_artifact(tmp_path, "test1.csv")
                result1 = automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path / "ws1"),
                    label_column="y",
                    sampled_test_dataset=sampled_test1,
                    sampling_method="random",
                )
                sampled_test2 = _make_test_artifact(tmp_path, "test2.csv")
                result2 = automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path / "ws2"),
                    label_column="y",
                    sampled_test_dataset=sampled_test2,
                    sampling_method="random",
                )

            n1 = result1.sample_config["n_samples"]
            n2 = result2.sample_config["n_samples"]
            assert n1 == n2, "Same random_state should yield same sample size"
            assert n1 == 2, "Downsampling should have been triggered (5 rows * 50 MB/row > 100 MB)"
        finally:
            MockedDataFrame.BYTES_PER_ROW = original_bytes_per_row

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_component_random_sampling_multiple_chunks(self, tmp_path):
        """Test random sampling with CSV large enough to trigger multiple chunks (>10k rows)."""
        header = "col1,col2\n"
        rows = "\n".join(f"{i},{i * 2}" for i in range(15000))
        csv_content = header + rows
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/large.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="col2",
                sampled_test_dataset=sampled_test,
                sampling_method="random",
            )

            assert result.sample_config["n_samples"] == 15000
        assert (tmp_path / "datasets" / "models_selection_train_dataset.csv").exists()


class TestDataLoaderSplitLogic:
    """Tests for the train/test split logic integrated into the data loader."""

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_split_outputs_have_correct_paths(self, tmp_path):
        """Verify selection-train and extra-train are written to workspace/datasets/."""
        csv_content = "a,b,target\n1,2,X\n3,4,Y\n5,6,X\n7,8,Y\n9,10,X\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
            )

        assert "models_selection_train_dataset.csv" in result.models_selection_train_data_path
        assert "extra_train_dataset.csv" in result.extra_train_data_path
        assert result.models_selection_train_data_path.startswith(str(tmp_path))
        assert result.extra_train_data_path.startswith(str(tmp_path))

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_split_config_defaults(self, tmp_path):
        """Default split_config uses test_size=0.2, random_state=42, stratify=False for regression."""
        csv_content = "a,b,target\n1,2,10\n3,4,20\n5,6,30\n7,8,40\n9,10,50\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="regression",
            )

        assert result.split_config["test_size"] == 0.2
        assert result.split_config["random_state"] == 42
        assert result.split_config["stratify"] is False

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_split_config_custom_values(self, tmp_path):
        """Custom split_config values are used and returned."""
        csv_content = "a,b,target\n1,2,10\n3,4,20\n5,6,30\n7,8,40\n9,10,50\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                split_config={"test_size": 0.3, "random_state": 123},
            )

        assert result.split_config["test_size"] == 0.3
        assert result.split_config["random_state"] == 123

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_classification_stratify_default_true(self, tmp_path):
        """Binary/multiclass tasks default to stratify=True in split_config output."""
        csv_content = "a,b,target\n1,2,A\n3,4,B\n5,6,A\n7,8,B\n9,10,A\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="binary",
            )

        assert result.split_config["stratify"] is True

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_classification_stratify_false_override(self, tmp_path):
        """Setting stratify=False in split_config disables stratification."""
        csv_content = "a,b,target\n1,2,A\n3,4,B\n5,6,A\n7,8,B\n9,10,A\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                task_type="binary",
                split_config={"stratify": False},
            )

        assert result.split_config["stratify"] is False

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_sample_row_is_json_string(self, tmp_path):
        """sample_row output is a JSON string from the test set head(1)."""
        csv_content = "a,b,target\n1,2,X\n3,4,Y\n5,6,X\n7,8,Y\n9,10,X\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
            )

        import json

        parsed = json.loads(result.sample_row)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert isinstance(parsed[0], dict)

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_test_dataset_written_to_artifact(self, tmp_path):
        """Test dataset is written to the sampled_test_dataset artifact path."""
        csv_content = "a,b,target\n1,2,X\n3,4,Y\n5,6,X\n7,8,Y\n9,10,X\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
            )

        assert sampled_test.uri == "/artifacts/test.csv"
        header, rows = _read_csv_path(sampled_test.path)
        assert "target" in header
        assert len(rows) >= 1

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_all_return_fields_present(self, tmp_path):
        """Return value has all expected fields."""
        csv_content = "a,b,target\n1,2,X\n3,4,Y\n5,6,X\n7,8,Y\n9,10,X\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
            )

        assert hasattr(result, "sample_config")
        assert hasattr(result, "split_config")
        assert hasattr(result, "sample_row")
        assert hasattr(result, "models_selection_train_data_path")
        assert hasattr(result, "extra_train_data_path")

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_split_csv_files_have_label_column(self, tmp_path):
        """All split CSV outputs contain the label column."""
        csv_content = "a,b,target\n1,2,X\n3,4,Y\n5,6,X\n7,8,Y\n9,10,X\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
            )

        sel_header, _ = _read_csv_path(result.models_selection_train_data_path)
        extra_header, _ = _read_csv_path(result.extra_train_data_path)
        test_header, _ = _read_csv_path(sampled_test.path)
        assert "target" in sel_header
        assert "target" in extra_header
        assert "target" in test_header

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_invalid_task_type_raises(self, tmp_path):
        """Invalid task_type raises ValueError before any S3 access."""
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas() as mock_s3:
            with pytest.raises(ValueError, match=r"task_type must be one of .*; got 'invalid'."):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                    task_type="invalid",
                )

            mock_s3.get_object.assert_not_called()

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_invalid_sampling_method_raises(self, tmp_path):
        """Invalid sampling_method raises ValueError."""
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match=r"sampling_method must be one of .* or None; got 'invalid'."):
                automl_data_loader.python_func(
                    file_key="data/file.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                    sampling_method="invalid",
                )

    @mock.patch.dict("os.environ", mocked_env_variables)
    def test_ssl_error_retries_with_verify_false(self, tmp_path):
        """SSLError on get_object triggers a retry with verify=False."""
        csv_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"

        call_count = 0

        def get_object_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _MockSSLError("SSL validation failed")
            return {"Body": io.BytesIO(csv_content.encode("utf-8"))}

        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_side_effect=get_object_side_effect):
            # Access the mocked boto3 to inspect client calls
            import boto3 as mocked_boto3

            result = automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                label_column="c",
                sampled_test_dataset=sampled_test,
            )

            assert result is not None
            assert result.sample_config["n_samples"] == 3

            # First call: default (verify not passed or verify=True)
            # Second call after SSL error: verify=False
            client_calls = mocked_boto3.client.call_args_list
            assert len(client_calls) == 2
            second_call_kwargs = client_calls[1][1]
            assert second_call_kwargs["verify"] is False
