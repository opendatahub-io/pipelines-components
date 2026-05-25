"""Unit tests for the timeseries_data_loader component.

boto3 and pandas are mocked via ``sys.modules`` so those packages are not required.
Output CSVs are asserted with the stdlib :mod:`csv` module.
"""

import csv
import datetime
import io
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from ..component import timeseries_data_loader
from .mocked_pandas import MockedDataFrame, make_mocked_pandas_module

mocked_env_variables = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "https://s3.example.local",
    "AWS_DEFAULT_REGION": "us-east-1",
}


class _MockSSLError(Exception):
    """Stand-in for botocore.exceptions.SSLError used in unit tests."""

    pass


@contextmanager
def _mock_boto3_module(get_object_return=None, get_object_side_effect=None):
    """Inject fake boto3/botocore so tests don't require real packages for SSLError handling."""
    mock_boto3 = mock.MagicMock()
    mock_s3 = mock.MagicMock()
    if get_object_side_effect is not None:
        mock_s3.get_object.side_effect = get_object_side_effect
    else:
        mock_s3.get_object.return_value = get_object_return or {"Body": io.BytesIO(b"")}
    mock_boto3.client.return_value = mock_s3

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
    """Inject mocked boto3 and pandas so the component runs without those dependencies."""
    mocked_pandas = make_mocked_pandas_module()
    with _mock_boto3_module(
        get_object_return=get_object_return, get_object_side_effect=get_object_side_effect
    ) as mock_s3:
        with mock.patch.dict(sys.modules, {"pandas": mocked_pandas}):
            yield mock_s3


def _make_test_artifact(tmp_path, name="sampled_test.csv"):
    """Create a simple artifact-like object for sampled_test_dataset."""
    art = mock.MagicMock()
    art.path = str(tmp_path / name)
    return art


def _read_csv_rows(path):
    """Read CSV rows as list[dict] with stdlib csv module."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


MIN_VALID_RECORDS = 100


def _date_from_day_offset(day_offset: int) -> str:
    """Return an ISO date string for day_offset days after 2024-01-01."""
    return (datetime.date(2024, 1, 1) + datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")


def _timeseries_csv(n_rows=MIN_VALID_RECORDS):
    """Build deterministic timeseries CSV content with required columns."""
    lines = ["item_id,timestamp,target,feature"]
    for i in range(n_rows):
        day = (i % 28) + 1
        month = (i // 28) + 1
        lines.append(f"series-1,2024-{month:02d}-{day:02d},{i},{i * 10}")
    return "\n".join(lines) + "\n"


def _shuffled_timeseries_csv(n_rows=MIN_VALID_RECORDS, seed=42):
    """Same rows as ``_timeseries_csv`` but data lines in pseudo-random order (header first)."""
    lines = ["item_id,timestamp,target,feature"]
    data = []
    for i in range(n_rows):
        day = (i % 28) + 1
        month = (i // 28) + 1
        data.append(f"series-1,2024-{month:02d}-{day:02d},{i},{i * 10}")
    rng = random.Random(seed)
    rng.shuffle(data)
    lines.extend(data)
    return "\n".join(lines) + "\n"


def _pad_timeseries_csv(csv_content: str, min_rows: int = MIN_VALID_RECORDS + 1) -> str:
    """Append unique series rows so cleansed data meets the component minimum record count."""
    lines = [ln for ln in csv_content.strip().splitlines() if ln]
    if len(lines) <= 1:
        return csv_content
    header = lines[0]
    data = lines[1:]
    i = len(data)
    while len(data) < min_rows:
        day = (i % 28) + 1
        month = (i // 28) + 1
        data.append(f"series-1,2024-{month:02d}-{day:02d},{i},{i * 10}")
        i += 1
    return header + "\n" + "\n".join(data) + "\n"


def _run_loader(tmp_path, csv_body, selection_train_size=0.3):
    """Execute ``python_func`` with mocked S3/pandas; return paths and ``sample_config``."""
    sampled_test = _make_test_artifact(tmp_path)
    with _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(csv_body.encode("utf-8"))}):
        result = timeseries_data_loader.python_func(
            file_key="ts.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            id_column="item_id",
            timestamp_column="timestamp",
            sampled_test_dataset=sampled_test,
            selection_train_size=selection_train_size,
        )
    return result, sampled_test


def _multiset_observations(selection_path, extra_path, test_path):
    """Stable sort of (id, timestamp, target) across all written splits."""
    rows = []
    for p in (selection_path, extra_path, test_path):
        rows.extend(_read_csv_rows(p))
    return sorted((r["item_id"], str(r["timestamp"]), r["target"]) for r in rows)


TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
MINIMAL_PANEL_CSV = TEST_DATA_DIR / "minimal_panel.csv"


class TestTimeseriesDataLoaderUnitTests:
    """Unit tests for timeseries_data_loader behavior."""

    def test_component_function_exists(self):
        """Component exposes a KFP python_func entrypoint."""
        assert callable(timeseries_data_loader)
        assert hasattr(timeseries_data_loader, "python_func")

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_component_default_split_outputs(self, tmp_path):
        """Default split creates expected files with chronological partitioning."""
        body_stream = io.BytesIO(_timeseries_csv().encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}) as mock_s3:
            result = timeseries_data_loader.python_func(
                file_key="timeseries/train.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                sampled_test_dataset=sampled_test,
            )

            mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="timeseries/train.csv")

        selection_rows = _read_csv_rows(result.models_selection_train_data_path)
        extra_rows = _read_csv_rows(result.extra_train_data_path)
        test_rows = _read_csv_rows(sampled_test.path)

        assert len(selection_rows) == 24
        assert len(extra_rows) == 56
        assert len(test_rows) == 20
        assert selection_rows[0]["target"] == "0"
        assert extra_rows[0]["target"] == "24"
        assert test_rows[0]["target"] == "80"

        assert result.sample_config["sampling_method"] == "first_n_rows"
        assert result.sample_config["total_rows_loaded"] == MIN_VALID_RECORDS
        assert result.split_config["test_size"] == 0.2
        assert result.split_config["selection_train_size"] == 0.3

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_per_series_split_each_id_gets_holdout(self, tmp_path):
        """Panel data: every series with >=2 rows contributes late timestamps to test, not only tail IDs."""
        lines = ["item_id,timestamp,target,feature"]
        for sid, letter in enumerate(["A", "B"]):
            base = sid * 100
            for i in range(50):
                lines.append(f"{letter},{_date_from_day_offset(i)},{base + i},{i}")
        body_stream = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            timeseries_data_loader.python_func(
                file_key="ts.csv",
                bucket_name="b",
                workspace_path=str(tmp_path),
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                sampled_test_dataset=sampled_test,
            )

        test_rows = _read_csv_rows(sampled_test.path)
        by_id = {letter: [r["target"] for r in test_rows if r["item_id"] == letter] for letter in ("A", "B")}
        assert sorted(by_id["A"], key=int) == ["40", "41", "42", "43", "44", "45", "46", "47", "48", "49"]
        assert sorted(by_id["B"], key=int) == ["140", "141", "142", "143", "144", "145", "146", "147", "148", "149"]
        assert len(test_rows) == 20

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_component_custom_selection_train_size(self, tmp_path):
        """Custom selection_train_size is reflected in split sizes and output metadata."""
        body_stream = io.BytesIO(_timeseries_csv().encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = timeseries_data_loader.python_func(
                file_key="timeseries/train.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                sampled_test_dataset=sampled_test,
                selection_train_size=0.5,
            )

        selection_rows = _read_csv_rows(result.models_selection_train_data_path)
        extra_rows = _read_csv_rows(result.extra_train_data_path)

        assert len(selection_rows) == 40
        assert len(extra_rows) == 40
        assert result.split_config["selection_train_size"] == 0.5

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_sample_rows_json_matches_test_tail(self, tmp_path):
        """sample_rows returns JSON records from test split tail (up to 5 rows)."""
        body_stream = io.BytesIO(_timeseries_csv().encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = timeseries_data_loader.python_func(
                file_key="timeseries/train.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                sampled_test_dataset=sampled_test,
            )

        parsed = json.loads(result.sample_rows)
        assert isinstance(parsed, list)
        assert len(parsed) == 5
        assert parsed[-1]["target"] == "99"

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_sampling_truncates_below_minimum_raises(self, tmp_path):
        """Truncation that leaves fewer than 100 rows fails before split."""
        body_stream = io.BytesIO(_timeseries_csv(3).encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        original_bytes_per_row = MockedDataFrame.BYTES_PER_ROW
        try:
            # 3 rows * 60 MB/row = 180 MB, so truncation should keep only 1 row under 100 MB.
            MockedDataFrame.BYTES_PER_ROW = 60_000_000
            with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
                with pytest.raises(ValueError, match="at least 100"):
                    timeseries_data_loader.python_func(
                        file_key="timeseries/train.csv",
                        bucket_name="my-bucket",
                        workspace_path=str(tmp_path),
                        target="target",
                        id_column="item_id",
                        timestamp_column="timestamp",
                        sampled_test_dataset=sampled_test,
                    )
        finally:
            MockedDataFrame.BYTES_PER_ROW = original_bytes_per_row

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_no_data_rows_raises(self, tmp_path):
        """Header-only CSV yields zero rows; fail before split with a clear error."""
        body_stream = io.BytesIO(b"item_id,timestamp,target,feature\n")
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            with pytest.raises(ValueError, match="loaded dataset has no data rows"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_missing_required_columns_raises(self, tmp_path):
        """Missing target/id/timestamp columns causes ValueError."""
        csv_content = "item_id,timestamp,feature\nseries-1,2024-01-01,10\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            with pytest.raises(ValueError, match="Missing required columns in dataset"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key"}, clear=True)
    def test_partial_credentials_raises(self, tmp_path):
        """Setting only one credential variable raises a configuration error."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match="S3 credentials misconfigured"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_raises(self, tmp_path):
        """No AWS credentials configured raises an explicit error."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match="S3 credentials missing"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_invalid_selection_train_size(self, tmp_path):
        """Test that invalid selection_train_size raises ValueError."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match="must be in a range 0 to 1"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                    selection_train_size=1.5,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_invalid_selection_train_size_at_upper_bound(self, tmp_path):
        """selection_train_size == 1 is outside (0, 1)."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match="must be in a range 0 to 1"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                    selection_train_size=1.0,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_invalid_selection_train_size_non_numeric(self, tmp_path):
        """selection_train_size must be int or float."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(TypeError, match=r"not supported between instances of 'str' and 'int'"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                    selection_train_size="0.3",
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_invalid_file_key(self, tmp_path):
        """Test that invalid file_key format raises ValueError."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match="valid S3 object key"):
                timeseries_data_loader.python_func(
                    file_key="/invalid/path/",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_invalid_file_key_double_slash(self, tmp_path):
        """file_key must not contain '//'."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match="valid S3 object key"):
                timeseries_data_loader.python_func(
                    file_key="timeseries//train.csv",
                    bucket_name="my-bucket",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_empty_string_inputs_raise_type_error(self, tmp_path):
        """Required string parameters must be non-empty."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas():
            with pytest.raises(ValueError, match="bucket_name must be a non-empty string"):
                timeseries_data_loader.python_func(
                    file_key="timeseries/train.csv",
                    bucket_name="   ",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_ssl_error_retries_with_verify_false(self, tmp_path):
        """SSLError on get_object triggers a retry with verify=False."""
        csv_content = _timeseries_csv()
        call_count = 0

        def get_object_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _MockSSLError("SSL validation failed")
            return {"Body": io.BytesIO(csv_content.encode("utf-8"))}

        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_side_effect=get_object_side_effect) as mock_s3:
            import boto3 as mocked_boto3

            result = timeseries_data_loader.python_func(
                file_key="timeseries/train.csv",
                bucket_name="my-bucket",
                workspace_path=str(tmp_path),
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                sampled_test_dataset=sampled_test,
            )

        assert result.sample_config["total_rows_loaded"] == MIN_VALID_RECORDS
        assert mock_s3.get_object.call_count == 2

        client_calls = mocked_boto3.client.call_args_list
        assert len(client_calls) == 2
        assert client_calls[0][1].get("verify", True) is True
        assert client_calls[1][1]["verify"] is False

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_duplicate_id_timestamp_keep_last(self, tmp_path):
        """Duplicate (item_id, timestamp) rows: keep last by time order after stable sort."""
        lines = ["item_id,timestamp,target,feature"]
        for i in range(10):
            lines.append(f"series-1,2024-01-{i + 1:02d},{i},{i * 10}")
        # Same day as row with target 4; later in file so keep=last prefers 999
        lines.append("series-1,2024-01-05,999,40")
        body_stream = io.BytesIO(_pad_timeseries_csv("\n".join(lines) + "\n").encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = timeseries_data_loader.python_func(
                file_key="ts.csv",
                bucket_name="b",
                workspace_path=str(tmp_path),
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                sampled_test_dataset=sampled_test,
            )

        assert result.sample_config["total_rows_loaded"] == MIN_VALID_RECORDS
        all_rows = (
            _read_csv_rows(result.models_selection_train_data_path)
            + _read_csv_rows(result.extra_train_data_path)
            + _read_csv_rows(sampled_test.path)
        )
        jan5 = [r for r in all_rows if r["timestamp"].startswith("2024-01-05")]
        assert len(jan5) == 1
        assert jan5[0]["target"] == "999"

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_invalid_timestamp_raises_value_error(self, tmp_path):
        """Unparseable timestamps fail fast (no silent row drops that break regular frequency)."""
        lines = ["item_id,timestamp,target,feature"]
        for i in range(10):
            lines.append(f"series-1,2024-01-{i + 1:02d},{i},{i * 10}")
        lines.append("series-1,not-a-date,99,0")
        body_stream = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            with pytest.raises(ValueError, match="could not be parsed"):
                timeseries_data_loader.python_func(
                    file_key="ts.csv",
                    bucket_name="b",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_non_finite_target_inf_becomes_nan_row_retained(self, tmp_path):
        """±inf in target is replaced with NaN; row is kept so AutoGluon can apply model-specific fill."""
        lines = ["item_id,timestamp,target,feature"]
        for i in range(10):
            lines.append(f"series-1,2024-01-{i + 1:02d},{i},{i * 10}")
        lines.append("series-1,2024-01-11,inf,0")
        body_stream = io.BytesIO(_pad_timeseries_csv("\n".join(lines) + "\n").encode("utf-8"))
        sampled_test = _make_test_artifact(tmp_path)

        with _mock_boto3_and_pandas(get_object_return={"Body": body_stream}):
            result = timeseries_data_loader.python_func(
                file_key="ts.csv",
                bucket_name="b",
                workspace_path=str(tmp_path),
                target="target",
                id_column="item_id",
                timestamp_column="timestamp",
                sampled_test_dataset=sampled_test,
            )

        assert result.sample_config["total_rows_loaded"] >= MIN_VALID_RECORDS
        all_targets = []
        for path in (
            result.models_selection_train_data_path,
            result.extra_train_data_path,
            sampled_test.path,
        ):
            all_targets.extend(r["target"] for r in _read_csv_rows(path))
        assert "inf" not in all_targets
        assert len(all_targets) >= MIN_VALID_RECORDS


class TestTimeseriesDataLoaderScenarioMatrix:
    """Scenario tests inspired by fixture-driven and ordering-invariant patterns."""

    @staticmethod
    def _two_series_sorted_csv():
        lines = ["item_id,timestamp,target,feature"]
        for sid, base in (("X", 0), ("Y", 100)):
            for i in range(50):
                lines.append(f"{sid},{_date_from_day_offset(i)},{base + i},{i}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _two_series_shuffled_csv(seed=7):
        lines = ["item_id,timestamp,target,feature"]
        rows = []
        for idx, (sid, base) in enumerate((("X", 0), ("Y", 100))):
            order = list(range(50))
            rng = random.Random(seed + idx * 31)
            rng.shuffle(order)
            for i in order:
                rows.append(f"{sid},{_date_from_day_offset(i)},{base + i},{i}")
        rng2 = random.Random(seed + 99)
        rng2.shuffle(rows)
        lines.extend(rows)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _fractional_year_csv(n_rows=MIN_VALID_RECORDS):
        """Numeric timestamps (fractional year), monotonic in file order."""
        lines = ["item_id,timestamp,target,feature"]
        for i in range(n_rows):
            t = 1949.0 + (i / 12.0)
            lines.append(f"S,{t},{i * 10},{i}")
        return "\n".join(lines) + "\n"

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_unsorted_rows_same_multiset_as_chronological_file(self, tmp_path_factory):
        """Shuffled row order yields the same observations per split as pre-sorted CSV."""
        d_sorted = tmp_path_factory.mktemp("sorted")
        d_shuf = tmp_path_factory.mktemp("shuffled")
        res_s, st_s = _run_loader(d_sorted, _timeseries_csv())
        res_h, st_h = _run_loader(d_shuf, _shuffled_timeseries_csv(seed=123))
        m_s = _multiset_observations(res_s.models_selection_train_data_path, res_s.extra_train_data_path, st_s.path)
        m_h = _multiset_observations(res_h.models_selection_train_data_path, res_h.extra_train_data_path, st_h.path)
        assert m_s == m_h
        test_s = sorted((r["timestamp"], r["target"]) for r in _read_csv_rows(st_s.path))
        test_h = sorted((r["timestamp"], r["target"]) for r in _read_csv_rows(st_h.path))
        assert test_s == test_h

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_two_series_interleaved_matches_sorted_multiset(self, tmp_path_factory):
        """Panel data with permuted row order matches sorted canonical multiset."""
        d1 = tmp_path_factory.mktemp("canon")
        d2 = tmp_path_factory.mktemp("mixed")
        res_a, st_a = _run_loader(d1, self._two_series_sorted_csv())
        res_b, st_b = _run_loader(d2, self._two_series_shuffled_csv())
        assert _multiset_observations(
            res_a.models_selection_train_data_path, res_a.extra_train_data_path, st_a.path
        ) == _multiset_observations(res_b.models_selection_train_data_path, res_b.extra_train_data_path, st_b.path)

    @pytest.mark.parametrize(
        "n_rows,selection_train_size,expect_selection,expect_extra,expect_test",
        [
            (100, 0.3, 24, 56, 20),
            (100, 0.5, 40, 40, 20),
        ],
        ids=["n100-default", "n100-sel05"],
    )
    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_split_output_sizes_parametrize(
        self,
        tmp_path,
        n_rows,
        selection_train_size,
        expect_selection,
        expect_extra,
        expect_test,
    ):
        """JSON-style matrix: (rows, selection_train_size) → expected split sizes."""
        result, sampled = _run_loader(tmp_path, _timeseries_csv(n_rows), selection_train_size)
        assert len(_read_csv_rows(result.models_selection_train_data_path)) == expect_selection
        assert len(_read_csv_rows(result.extra_train_data_path)) == expect_extra
        assert len(_read_csv_rows(sampled.path)) == expect_test
        assert result.sample_config["total_rows_loaded"] == n_rows

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_numeric_fractional_year_timestamps_split_by_order(self, tmp_path):
        """Non-ISO numeric time axis (fractional year) sorts and splits chronologically."""
        result, sampled = _run_loader(tmp_path, self._fractional_year_csv())
        assert result.sample_config["total_rows_loaded"] == MIN_VALID_RECORDS
        test_rows = _read_csv_rows(sampled.path)
        assert len(test_rows) == 20
        # Latest observations by time should land in test for default 20% per-series split.
        targets = sorted(int(r["target"]) for r in test_rows)
        assert targets == list(range(800, 1000, 10))

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_fixture_minimal_panel_matches_inline_generator(self, tmp_path):
        """Committed ``tests/data/minimal_panel.csv`` drives the same outcome as ``_timeseries_csv()``."""
        assert MINIMAL_PANEL_CSV.is_file()
        body = MINIMAL_PANEL_CSV.read_text(encoding="utf-8")
        res_fix, st_fix = _run_loader(tmp_path / "fx", body)
        res_inl, st_inl = _run_loader(tmp_path / "in", _timeseries_csv())
        assert _multiset_observations(
            res_fix.models_selection_train_data_path, res_fix.extra_train_data_path, st_fix.path
        ) == _multiset_observations(
            res_inl.models_selection_train_data_path, res_inl.extra_train_data_path, st_inl.path
        )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_too_few_valid_records_after_cleansing_raises(self, tmp_path):
        """Fail early when cleansed data has fewer than 100 valid records."""
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(_timeseries_csv(5).encode("utf-8"))}):
            with pytest.raises(ValueError, match="at least 100"):
                timeseries_data_loader.python_func(
                    file_key="ts.csv",
                    bucket_name="b",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_all_rows_invalid_timestamp_raises_after_cleansing(self, tmp_path):
        """If every timestamp is unparseable, fail with a clear error (do not drop all rows)."""
        body = "item_id,timestamp,target,feature\na,not-a-date,0,x\nb,bad-ts,1,y\n"
        sampled_test = _make_test_artifact(tmp_path)
        with _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(body.encode("utf-8"))}):
            with pytest.raises(ValueError, match="could not be parsed"):
                timeseries_data_loader.python_func(
                    file_key="ts.csv",
                    bucket_name="b",
                    workspace_path=str(tmp_path),
                    target="target",
                    id_column="item_id",
                    timestamp_column="timestamp",
                    sampled_test_dataset=sampled_test,
                )
