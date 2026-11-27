"""Parquet-input tests for the tabular_data_loader component.

S3 and sklearn are mocked (scikit-learn is not a test dependency); pandas,
pyarrow, and the Parquet files run for real, so corruption cases are exercised
against actual damaged files. CSV coverage with real pandas comes from the
CSV/Parquet parity test; the mocked-pandas CSV suite lives in
``test_component_unit.py``.
"""

import io
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from ..component import automl_data_loader

mocked_env_variables = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "https://s3.example.local",
    "AWS_DEFAULT_REGION": "us-east-1",
}

MIN_VALID_RECORDS = 100


class _MockSSLError(Exception):
    """Stand-in for botocore.exceptions.SSLError used in unit tests."""

    pass


def _deterministic_train_test_split(X, y, *, test_size=0.25, stratify=None, random_state=None):
    """Positional split: leading rows to train, the rest to test.

    Deterministic and format-agnostic, so a CSV run and a Parquet run over the
    same rows produce identical splits (the parity test relies on that).
    ``stratify`` is accepted and ignored, like the mocked-pandas splitter.
    """
    split_index = max(1, int(len(X) * (1 - test_size)))
    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


@contextmanager
def _mock_boto3_and_sklearn(get_object_side_effect=None, head_object_side_effect=None, download_file_side_effect=None):
    """Inject fake boto3/botocore and sklearn modules; pandas and pyarrow run for real.

    Only the touched sys.modules entries are set (and restored) individually:
    restoring the whole mapping unregisters modules that Arrow's C++ registry
    still references, breaking later Parquet operations in the same session.
    """
    mock_boto3 = mock.MagicMock()
    mock_s3 = mock.MagicMock()
    if get_object_side_effect is not None:
        mock_s3.get_object.side_effect = get_object_side_effect
    if head_object_side_effect is not None:
        mock_s3.head_object.side_effect = head_object_side_effect
    if download_file_side_effect is not None:
        mock_s3.download_file.side_effect = download_file_side_effect
    mock_boto3.client.return_value = mock_s3

    mock_botocore = mock.MagicMock()
    mock_botocore_exceptions = mock.MagicMock()
    mock_botocore_exceptions.SSLError = _MockSSLError
    mock_botocore.exceptions = mock_botocore_exceptions

    mock_sklearn = types.ModuleType("sklearn")
    mock_model_selection = types.ModuleType("sklearn.model_selection")
    mock_model_selection.train_test_split = _deterministic_train_test_split
    mock_sklearn.model_selection = mock_model_selection

    with pytest.MonkeyPatch.context() as patcher:
        for name, module in {
            "boto3": mock_boto3,
            "botocore": mock_botocore,
            "botocore.exceptions": mock_botocore_exceptions,
            "sklearn": mock_sklearn,
            "sklearn.model_selection": mock_model_selection,
        }.items():
            patcher.setitem(sys.modules, name, module)
        yield mock_s3


def _frame(rows: int, *, classification: bool = False) -> pd.DataFrame:
    """Deterministic frame whose dtypes Parquet round-trips exactly (int64, object)."""
    if classification:
        return pd.DataFrame(
            {
                "feature_1": range(rows),
                "f2": [f"s{i}" for i in range(rows)],
                "target": ["A" if i % 2 == 0 else "B" for i in range(rows)],
            }
        )
    return pd.DataFrame(
        {
            "feature_1": range(rows),
            "f2": [f"s{i}" for i in range(rows)],
            "target": [float(i) for i in range(rows)],
        }
    )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe to an in-memory Parquet file."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _make_artifacts(tmp_path):
    """Build sampled_test_dataset and component_status artifact stand-ins."""
    sampled_test = SimpleNamespace(path=str(tmp_path / "sampled_test.parquet"), uri="/artifacts/test.parquet")
    component_status = SimpleNamespace(path=str(tmp_path / "component_status"), metadata={})
    return sampled_test, component_status


def _serving_parquet_payload(payload: bytes):
    """Build S3 mock behaviours serving one Parquet object for staging."""

    def head_object(*, Bucket, Key):
        return {"ContentLength": len(payload)}

    def download_file(bucket, key, filename, **kwargs):
        Path(filename).write_bytes(payload)

    def get_object(*, Bucket, Key, **kwargs):
        # Only reached for magic-byte sniffing (extensionless keys).
        return {"Body": io.BytesIO(payload[:4]), "ContentLength": len(payload)}

    return get_object, head_object, download_file


def _status_json(component_status) -> dict:
    """Read the component_status.json the loader wrote."""
    return json.loads((Path(component_status.path) / "component_status.json").read_text(encoding="utf-8"))


def _stage_metrics(status_payload: dict) -> dict:
    """Return the prepare_data stage metrics."""
    return next(stage for stage in status_payload["stages"] if stage["id"] == "prepare_data")["metrics"]


class TestParquetInput:
    """A valid Parquet training source trains-ready data without any manual conversion."""

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_parquet_input_random_sampling(self, tmp_path):
        """A valid Parquet file is staged, sampled, and split exactly like a CSV source."""
        frame = _frame(120)
        get_object, head_object, download_file = _serving_parquet_payload(_parquet_bytes(frame))
        sampled_test, component_status = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file) as mock_s3:
            result = automl_data_loader.python_func(
                file_key="data/train.parquet",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                component_status=component_status,
            )

        # CSV keys are not sniffed or staged; Parquet keys are staged, never streamed.
        mock_s3.get_object.assert_not_called()
        mock_s3.download_file.assert_called_once()

        assert result.sample_config["n_samples"] == 120
        selection = pd.read_parquet(result.models_selection_train_data_path)
        extra = pd.read_parquet(result.extra_train_data_path)
        test = pd.read_parquet(sampled_test.path)
        assert len(selection) + len(extra) + len(test) == 120
        assert selection["feature_1"].dtype == "int64"

        metrics = _stage_metrics(_status_json(component_status))
        assert metrics["source_format"] == "parquet"
        assert metrics["s3_transfer_mode"] == "parquet_local"
        assert metrics["source_bytes"] == len(_parquet_bytes(frame))

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_parquet_input_stratified_sampling(self, tmp_path):
        """Classification tasks keep stratified splits over Parquet chunks."""
        frame = _frame(120, classification=True)
        get_object, head_object, download_file = _serving_parquet_payload(_parquet_bytes(frame))
        sampled_test, component_status = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            result = automl_data_loader.python_func(
                file_key="data/train.parquet",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                task_type="binary",
                sampled_test_dataset=sampled_test,
                component_status=component_status,
            )

        assert result.split_config["stratify"] is True
        for path in (result.models_selection_train_data_path, result.extra_train_data_path, sampled_test.path):
            labels = pd.read_parquet(path)["target"]
            assert set(labels) == {"A", "B"}

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_extensionless_parquet_detected_by_magic_bytes(self, tmp_path):
        """A Parquet file without a Parquet extension is sniffed and read as Parquet."""
        payload = _parquet_bytes(_frame(120))
        get_object, head_object, download_file = _serving_parquet_payload(payload)
        sampled_test, component_status = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file) as mock_s3:
            result = automl_data_loader.python_func(
                file_key="data/export",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                component_status=component_status,
            )

        # The ranged GET sniffs the leading magic bytes; the full object is staged.
        sniff = mock_s3.get_object.call_args
        assert sniff.kwargs["Range"] == "bytes=0-3"
        assert result.sample_config["n_samples"] == 120
        metrics = _stage_metrics(_status_json(component_status))
        assert metrics["source_format"] == "parquet"


class TestParquetUserDataTestData:
    """A user-provided Parquet test dataset replaces the holdout split."""

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_user_parquet_test_data_happy_path(self, tmp_path):
        """Parquet test data is validated, written to the artifact, and skips the holdout split."""
        train = _frame(120)
        test = pd.DataFrame({"feature_1": [1000, 1001], "f2": ["t0", "t1"], "target": [10.0, 11.0]})
        parquet_payloads = {"data/train.parquet": _parquet_bytes(train), "data/test.parquet": _parquet_bytes(test)}

        def get_object(*, Bucket, Key, **kwargs):
            payload = parquet_payloads[Key]
            return {"Body": io.BytesIO(payload[:4]), "ContentLength": len(payload)}

        def head_object(*, Bucket, Key):
            return {"ContentLength": len(parquet_payloads[Key])}

        def download_file(bucket, key, filename, **kwargs):
            Path(filename).write_bytes(parquet_payloads[key])

        sampled_test, component_status = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            result = automl_data_loader.python_func(
                file_key="data/train.parquet",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                label_column="target",
                sampled_test_dataset=sampled_test,
                component_status=component_status,
                test_data_bucket_name="test-bucket",
                test_data_file_key="data/test.parquet",
            )

        assert result.split_config["test_size"] == 0.0
        written_test = pd.read_parquet(sampled_test.path)
        assert list(written_test["feature_1"]) == [1000, 1001]
        # The holdout split is skipped: all training rows feed selection/extra.
        selection = pd.read_parquet(result.models_selection_train_data_path)
        extra = pd.read_parquet(result.extra_train_data_path)
        assert len(selection) + len(extra) == 120

        split_stage = next(
            stage for stage in _status_json(component_status)["stages"] if stage["id"] == "split_and_export"
        )
        assert split_stage["metrics"]["user_test_source"] == "s3://test-bucket/data/test.parquet"
        assert split_stage["metrics"]["test_rows"] == 2


class TestCorruptParquetRejected:
    """Corrupted or malformed Parquet files are rejected with a descriptive error and no outputs."""

    @staticmethod
    def _assert_no_outputs(tmp_path, sampled_test):
        """No split artifacts may exist when the load fails (no pipeline progress)."""
        datasets_dir = tmp_path / "datasets"
        assert not datasets_dir.exists() or list(datasets_dir.glob("*")) == []
        assert not Path(sampled_test.path).exists()

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_non_parquet_bytes_named_parquet_rejected(self, tmp_path):
        """A CSV payload renamed to .parquet is rejected naming the file and the problem."""
        payload = b"feature_1,f2,target\n1,s0,10.0\n"
        get_object, head_object, download_file = _serving_parquet_payload(payload)
        sampled_test, _ = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            with pytest.raises(ValueError) as exc_info:
                automl_data_loader.python_func(
                    file_key="data/train.parquet",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                )

        message = str(exc_info.value)
        assert "s3://bucket/data/train.parquet" in message
        assert "is not a valid Parquet file" in message
        assert "does not start with the Parquet magic bytes" in message
        self._assert_no_outputs(tmp_path, sampled_test)

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_truncated_parquet_rejected(self, tmp_path):
        """A Parquet file cut short is rejected with a truncation-specific message."""
        payload = _parquet_bytes(_frame(120))[:600]
        get_object, head_object, download_file = _serving_parquet_payload(payload)
        sampled_test, _ = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            with pytest.raises(ValueError) as exc_info:
                automl_data_loader.python_func(
                    file_key="data/train.parquet",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                )

        message = str(exc_info.value)
        assert "s3://bucket/data/train.parquet" in message
        assert "appears truncated" in message
        self._assert_no_outputs(tmp_path, sampled_test)

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_empty_parquet_object_rejected(self, tmp_path):
        """A zero-byte Parquet upload is rejected naming the empty file."""
        get_object, head_object, download_file = _serving_parquet_payload(b"")
        sampled_test, _ = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            with pytest.raises(ValueError, match="is empty \\(0 bytes\\)"):
                automl_data_loader.python_func(
                    file_key="data/train.parquet",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                )

        self._assert_no_outputs(tmp_path, sampled_test)

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_corrupt_parquet_rejected_for_stratified_sampling_too(self, tmp_path):
        """Every sampling strategy rejects damaged Parquet, including the recoverable-partial-read one."""
        payload = b"feature_1,f2,target\n1,s0,A\n"
        get_object, head_object, download_file = _serving_parquet_payload(payload)
        sampled_test, _ = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            with pytest.raises(ValueError, match="is not a valid Parquet file") as exc_info:
                automl_data_loader.python_func(
                    file_key="data/train.parquet",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    task_type="binary",
                    sampled_test_dataset=sampled_test,
                )

        # The descriptive Parquet error is not double-wrapped in a generic read error.
        assert "Error reading Parquet file from S3" not in str(exc_info.value)
        self._assert_no_outputs(tmp_path, sampled_test)

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_corrupt_parquet_test_data_wrapped_as_test_dataset_error(self, tmp_path):
        """Corrupt Parquet test data fails with the distinguishable test-dataset error."""
        train_csv = _frame(120).to_csv(index=False)
        test_payload = b"definitely not parquet"

        parquet_payloads = {"data/test.parquet": test_payload}

        def get_object(*, Bucket, Key, **kwargs):
            return {"Body": io.BytesIO(train_csv.encode("utf-8")), "ContentLength": len(train_csv)}

        def head_object(*, Bucket, Key):
            return {"ContentLength": len(parquet_payloads[Key])}

        def download_file(bucket, key, filename, **kwargs):
            Path(filename).write_bytes(parquet_payloads[key])

        sampled_test, _ = _make_artifacts(tmp_path)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            with pytest.raises(ValueError) as exc_info:
                automl_data_loader.python_func(
                    file_key="data/train.csv",
                    bucket_name="bucket",
                    workspace_path=str(tmp_path),
                    label_column="target",
                    sampled_test_dataset=sampled_test,
                    test_data_bucket_name="test-bucket",
                    test_data_file_key="data/test.parquet",
                )

        message = str(exc_info.value)
        assert "Failed to load user-provided test dataset from s3://test-bucket/data/test.parquet" in message
        assert "is not a valid Parquet file" in message
        self._assert_no_outputs(tmp_path, sampled_test)


class TestCsvParquetParity:
    """The same data loaded as CSV and as Parquet produces identical splits."""

    @mock.patch.dict("os.environ", mocked_env_variables, clear=True)
    def test_csv_and_parquet_inputs_produce_identical_outputs(self, tmp_path):
        """Row-for-row identical selection, extra, and test splits for both formats."""
        frame = _frame(120)
        csv_payload = frame.to_csv(index=False).encode("utf-8")
        parquet_payload = _parquet_bytes(frame)

        csv_run = self._run_csv(tmp_path / "csv", csv_payload)
        parquet_run = self._run_parquet(tmp_path / "parquet", parquet_payload)

        for csv_output, parquet_output in zip(csv_run, parquet_run):
            pd.testing.assert_frame_equal(pd.read_parquet(csv_output), pd.read_parquet(parquet_output))

    @mock.patch.dict(os.environ, mocked_env_variables, clear=True)
    def test_categorical_label_with_unused_category_matches_csv(self, tmp_path):
        """A Parquet categorical label with an unused category splits exactly like the CSV.

        Parquet preserves pandas categorical metadata, so ``value_counts`` reports
        the zero-row category ``C`` alongside the real labels; the stratified
        viability check previously rejected the dataset (counts [60, 60, 0]) while
        the equivalent CSV (object dtype) passed. Unused categories are stripped
        during cleansing, so both formats produce identical splits.
        """
        labels = pd.Categorical(["A" if i % 2 == 0 else "B" for i in range(120)], categories=["A", "B", "C"])
        parquet_frame = pd.DataFrame({"feature_1": range(120), "target": labels})
        csv_payload = (
            pd.DataFrame({"feature_1": range(120), "target": labels.astype(str)}).to_csv(index=False).encode("utf-8")
        )

        csv_run = self._run_csv(tmp_path / "csv", csv_payload, task_type="binary")
        parquet_run = self._run_parquet(tmp_path / "parquet", _parquet_bytes(parquet_frame), task_type="binary")

        for csv_output, parquet_output in zip(csv_run, parquet_run):
            csv_out = pd.read_parquet(csv_output)
            parquet_out = pd.read_parquet(parquet_output)
            assert csv_out["feature_1"].tolist() == parquet_out["feature_1"].tolist()
            assert csv_out["target"].tolist() == parquet_out["target"].tolist()
            assert set(parquet_out["target"]) == {"A", "B"}

    @staticmethod
    def _run_csv(workspace, csv_payload, task_type="regression"):
        """Run the loader over a streamed CSV source; return the three output paths."""
        workspace.mkdir(parents=True)
        sampled_test, component_status = _make_artifacts(workspace)

        def get_object(*, Bucket, Key, **kwargs):
            return {"Body": io.BytesIO(csv_payload), "ContentLength": len(csv_payload)}

        with _mock_boto3_and_sklearn(get_object_side_effect=get_object):
            result = automl_data_loader.python_func(
                file_key="data/train.csv",
                bucket_name="bucket",
                workspace_path=str(workspace),
                label_column="target",
                task_type=task_type,
                sampled_test_dataset=sampled_test,
                component_status=component_status,
            )

        assert _stage_metrics(_status_json(component_status))["source_format"] == "csv"
        return [sampled_test.path, result.models_selection_train_data_path, result.extra_train_data_path]

    @staticmethod
    def _run_parquet(workspace, parquet_payload, task_type="regression"):
        """Run the loader over a staged Parquet source; return the three output paths."""
        workspace.mkdir(parents=True)
        sampled_test, component_status = _make_artifacts(workspace)
        get_object, head_object, download_file = _serving_parquet_payload(parquet_payload)

        with _mock_boto3_and_sklearn(get_object, head_object, download_file):
            result = automl_data_loader.python_func(
                file_key="data/train.parquet",
                bucket_name="bucket",
                workspace_path=str(workspace),
                label_column="target",
                task_type=task_type,
                sampled_test_dataset=sampled_test,
                component_status=component_status,
            )

        assert _stage_metrics(_status_json(component_status))["source_format"] == "parquet"
        return [sampled_test.path, result.models_selection_train_data_path, result.extra_train_data_path]
