"""Regression and real-pandas checks for timestamp-based per-series sampling."""

import io
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from ..component import timeseries_data_loader
from .mocked_pandas import MockedDataFrame
from .test_component_unit import (
    _date_from_day_offset,
    _make_test_artifact,
    _mock_boto3_and_pandas,
    _mock_boto3_module,
    _read_csv_rows,
    mocked_env_variables,
)


def _panel_csv(length=300, order="sorted"):
    rows = [f"{item},{_date_from_day_offset(day)},{day},{day}" for item in ("A", "B") for day in range(length)]
    if order == "reversed":
        rows.reverse()
    elif order == "shuffled":
        import random

        random.Random(17).shuffle(rows)
    return "item_id,timestamp,target,feature\n" + "\n".join(rows) + "\n"


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_large_panel_retains_latest_for_every_series(tmp_path, order):
    """A capped component retains recent A and B rows and profiles the actual splits."""
    test_artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(MockedDataFrame, "BYTES_PER_ROW", 500_000),
        _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(_panel_csv(order=order).encode())}),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=test_artifact,
        )
    splits = {
        "selection_train": _read_csv_rows(result.models_selection_train_data_path),
        "extra_train": _read_csv_rows(result.extra_train_data_path),
        "test": _read_csv_rows(test_artifact.path),
    }
    status = json.loads((tmp_path / "component_status" / "component_status.json").read_text())
    profile = status["metadata"]["sampling_profile"]
    assert result.sample_config["source_rows_seen"] == 600
    assert result.sample_config["sampled_rows"] == 208
    assert "sampling_profile" not in timeseries_data_loader.component_spec.outputs
    for series in profile["datasets"]["retained"]:
        item = series["series_id"]
        retained = [row for rows in splits.values() for row in rows if row["item_id"] == item]
        assert sorted(int(row["target"]) for row in retained) == list(range(196, 300))
        assert series["rows"] == 104
        assert series["timestamp_min"] == _date_from_day_offset(196)
        assert series["timestamp_max"] == _date_from_day_offset(299)
        assert series["source_rows_seen"] == 300
        assert series["source_timestamp_min"] == _date_from_day_offset(0)
        assert series["source_timestamp_max"] == _date_from_day_offset(299)
        for name, rows in splits.items():
            own_rows = [row for row in rows if row["item_id"] == item]
            stats = next(entry for entry in profile["datasets"][name] if entry["series_id"] == item)
            assert stats["rows"] == len(own_rows)
            assert stats["timestamp_min"] == min(row["timestamp"] for row in own_rows)
            assert stats["timestamp_max"] == max(row["timestamp"] for row in own_rows)


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_user_test_sampling_keeps_both_series_tails(tmp_path, caplog):
    """External evaluation data uses its own cap and retains every series' latest dates."""

    def get_object(**kwargs):
        is_test = kwargs["Key"] == "test.csv"
        MockedDataFrame.BYTES_PER_ROW = 4_000_000 if is_test else 100
        return {"Body": io.BytesIO(_panel_csv(30 if is_test else 100).encode())}

    test_artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(MockedDataFrame, "BYTES_PER_ROW", 100),
        _mock_boto3_and_pandas(get_object_side_effect=get_object),
    ):
        timeseries_data_loader.python_func(
            file_key="train.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=test_artifact,
            test_data_bucket_name="b",
            test_data_file_key="test.csv",
        )
    rows = _read_csv_rows(test_artifact.path)
    for item in ("A", "B"):
        assert sorted(int(row["target"]) for row in rows if row["item_id"] == item) == list(range(24, 30))
    status = json.loads((tmp_path / "component_status" / "component_status.json").read_text())
    test_profile = status["metadata"]["sampling_profile"]["datasets"]["test"]
    assert len(test_profile) == 2
    assert all(series["source_rows_seen"] == 30 for series in test_profile)
    assert all(series["source_timestamp_min"] == _date_from_day_offset(0) for series in test_profile)
    split_stage = next(stage for stage in status["stages"] if stage["id"] == "split_and_export")
    assert split_stage["metrics"]["truncated"] is True
    assert "newest timestamps per series" in caplog.text
    assert "leading-row prefix" not in caplog.text


@pytest.fixture
def real_pandas(monkeypatch):
    """Use real pandas, including the shared writer after earlier mocked component tests."""
    pd = pytest.importorskip("pandas")
    from kfp_components.components.training.automl.shared import parquet_utils

    monkeypatch.setattr(parquet_utils, "pd", pd)
    return pd


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
@pytest.mark.parametrize("chunk_size", [1, 17, 10000])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_real_sampler_matches_full_frame_oracle(real_pandas, tmp_path, order, chunk_size):
    """Real pandas/Parquet agree with full group-tail sorting across orders and chunk sizes."""
    pytest.importorskip("pyarrow")
    from pandas.io.parquet import get_engine

    get_engine("pyarrow")
    pd = real_pandas
    read_csv, itertuples = pd.read_csv, pd.DataFrame.itertuples

    class SizedTuple(tuple):
        def __sizeof__(self):
            return 500_000

    def sized_rows(frame, **kwargs):
        return (SizedTuple(row) for row in itertuples(frame, **kwargs))

    def read_chunks(stream, **kwargs):
        kwargs["chunksize"] = chunk_size
        return read_csv(stream, **kwargs)

    # Latest duplicates arrive in a different chunk; keep the last file occurrence.
    body = _panel_csv(order=order) + f"A,{_date_from_day_offset(299)},999,9\n"
    test_artifact = _make_test_artifact(tmp_path)
    with (
        _mock_boto3_module(get_object_return={"Body": io.BytesIO(body.encode())}),
        mock.patch.object(pd, "read_csv", side_effect=read_chunks),
        mock.patch.object(pd.DataFrame, "itertuples", sized_rows),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=test_artifact,
        )
    actual = (
        pd.concat(
            [
                pd.read_parquet(result.models_selection_train_data_path),
                pd.read_parquet(result.extra_train_data_path),
                pd.read_parquet(test_artifact.path),
            ]
        )
        .sort_values(["item_id", "timestamp"])
        .reset_index(drop=True)
    )
    expected = read_csv(io.StringIO(body))
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])
    expected = (
        expected.drop_duplicates(["item_id", "timestamp"], keep="last")
        .sort_values(["item_id", "timestamp"])
        .groupby("item_id")
        .tail(104)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert Path(result.models_selection_train_data_path).read_bytes()[:4] == b"PAR1"


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_invalid_tail_is_not_hidden_by_budget(tmp_path):
    """A malformed timestamp after the old head cutoff fails instead of being skipped."""
    with (
        mock.patch.object(MockedDataFrame, "BYTES_PER_ROW", 500_000),
        _mock_boto3_and_pandas(
            get_object_return={
                "Body": io.BytesIO((_panel_csv() + "A,invalid,1,1\n").encode()),
            }
        ),
        pytest.raises(ValueError, match="could not be parsed"),
    ):
        timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=_make_test_artifact(tmp_path),
        )


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_numeric_ids_keep_file_order_across_chunks(real_pandas, tmp_path):
    """Mixed numeric ID tokens cannot change the identity of integer series IDs."""
    rows = [f"1,{_date_from_day_offset(day)},{day},0" for day in range(150)]
    rows[0] = f"1,{_date_from_day_offset(149)},111,0"
    rows[60] = f"2.5,{_date_from_day_offset(60)},60,0"
    rows.append(f"1,{_date_from_day_offset(149)},999,0")
    actual = _run_real_component(real_pandas, tmp_path, "item_id,timestamp,target,feature\n" + "\n".join(rows), 50)
    own = actual[actual["item_id"] == "1"]
    assert len(own) == 148
    assert own.loc[own["timestamp"] == real_pandas.Timestamp(_date_from_day_offset(149)), "target"].tolist() == [999]
    assert set(actual["item_id"]) == {"1", "2.5"}


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_duplicate_cost_is_independent_of_chunk_boundary(real_pandas, tmp_path):
    """An overwritten large value contributes to the budget in every chunk layout."""
    rows = [f"A,{_date_from_day_offset(0)},111," + "x" * 500_000]
    rows.append(f"A,{_date_from_day_offset(0)},999,small")
    rows.extend(f"A,{_date_from_day_offset(day)},{day},small" for day in range(1, 400))
    body = "item_id,timestamp,target,feature\n" + "\n".join(rows)
    results = []
    for chunk_size in (1, 10000):
        workspace = tmp_path / str(chunk_size)
        workspace.mkdir()
        results.append(_run_real_component(real_pandas, workspace, body, chunk_size))
    real_pandas.testing.assert_frame_equal(*results)
    assert 100 <= len(results[0]) < 400


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_full_profile_preserves_all_series_and_actual_unique_counts(tmp_path):
    """A bounded status summary points to full, post-dedup counts and source ranges."""
    rows = [f"S{item:03d},{_date_from_day_offset(day)},{day},0" for item in range(60) for day in (*range(25), 24)]
    artifact = _make_test_artifact(tmp_path)
    with _mock_boto3_and_pandas(
        get_object_return={"Body": io.BytesIO(("item_id,timestamp,target,feature\n" + "\n".join(rows)).encode())}
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    status_dir = tmp_path / "component_status"
    summary = json.loads((status_dir / "component_status.json").read_text())["metadata"]["sampling_profile"]
    full = json.loads((status_dir / summary["profile_file"]).read_text())
    assert result.sample_config["sampled_rows"] == 1500
    assert full["source_rows_seen"] == 1560
    assert full["input_complete"] is True
    for name, entries in full["datasets"].items():
        assert len(entries) == summary["series_counts"][name] == 60
        assert len(summary["datasets"][name]) == 50
        assert summary["profiles_truncated"][name] is True
    for series in full["datasets"]["retained"]:
        assert series["rows"] == 25
        assert series["source_rows_seen"] == 26
        assert series["source_timestamp_min"] == _date_from_day_offset(0)
        assert series["source_timestamp_max"] == _date_from_day_offset(24)


def _oversized_value_size(value, *args):
    """Simulate an oversized CSV value without allocating a 100 MiB string."""
    if isinstance(value, str) and value == "oversized":
        return 200 * 1024 * 1024
    return _original_getsizeof(value, *args)


_original_getsizeof = sys.getsizeof


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
@pytest.mark.parametrize("outlier_day", [0, 100])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_oversized_history_keeps_contiguous_latest_tail(tmp_path, order, outlier_day):
    """An unretainable old row acts as a boundary, with no gaps in the exported tail."""
    rows = [
        f"A,{_date_from_day_offset(day)},{day},{'oversized' if day == outlier_day else 'small'}" for day in range(301)
    ]
    if order == "reversed":
        rows.reverse()
    elif order == "shuffled":
        import random

        random.Random(17).shuffle(rows)
    artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(sys, "getsizeof", side_effect=_oversized_value_size),
        _mock_boto3_and_pandas(
            get_object_return={"Body": io.BytesIO(("item_id,timestamp,target,feature\n" + "\n".join(rows)).encode())}
        ),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    retained = [
        row
        for path in (result.models_selection_train_data_path, result.extra_train_data_path, artifact.path)
        for row in _read_csv_rows(path)
    ]
    assert sorted(int(row["target"]) for row in retained) == list(range(outlier_day + 1, 301))
    status = json.loads((tmp_path / "component_status" / "component_status.json").read_text())
    series = status["metadata"]["sampling_profile"]["datasets"]["retained"][0]
    assert series["source_rows_seen"] == 301
    assert series["oversized_rows_seen"] == 1
    assert series["source_timestamp_min"] == _date_from_day_offset(0)
    assert series["timestamp_min"] == _date_from_day_offset(outlier_day + 1)


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_oversized_latest_observation_fails_instead_of_returning_old_history(tmp_path):
    """The most recent observation cannot be silently omitted for a series."""
    body = _panel_csv() + f"A,{_date_from_day_offset(300)},300,oversized\n"
    with (
        mock.patch.object(sys, "getsizeof", side_effect=_oversized_value_size),
        _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(body.encode())}),
        pytest.raises(ValueError, match="cannot retain the latest observation"),
    ):
        timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=_make_test_artifact(tmp_path),
        )


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_last_duplicate_can_replace_an_oversized_marker(tmp_path):
    """An early oversized duplicate does not remove valid, later corrections or history."""
    body = "item_id,timestamp,target,feature\nA," + _date_from_day_offset(150) + ",999,oversized\n"
    body += "\n".join(f"A,{_date_from_day_offset(day)},{day},small" for day in range(300))
    artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(sys, "getsizeof", side_effect=_oversized_value_size),
        _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(body.encode())}),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    rows = [
        row
        for path in (result.models_selection_train_data_path, result.extra_train_data_path, artifact.path)
        for row in _read_csv_rows(path)
    ]
    assert sorted(int(row["target"]) for row in rows) == list(range(300))


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_real_duplicate_values_keep_last_file_occurrence(real_pandas, tmp_path):
    """Sorting equal timestamps must not change the values of the last file occurrence."""
    import random

    rows = [f"A,{_date_from_day_offset(i % 150)},{i},0" for i in range(600)]
    random.Random(17).shuffle(rows)
    body = "item_id,timestamp,target,feature\n" + "\n".join(rows)
    actual = _run_real_component(real_pandas, tmp_path, body, 17)
    expected = real_pandas.read_csv(io.StringIO(body), dtype={"item_id": "string"})
    expected["timestamp"] = real_pandas.to_datetime(expected["timestamp"])
    expected = expected.drop_duplicates(["item_id", "timestamp"], keep="last").sort_values("timestamp")
    assert actual["target"].tolist() == expected["target"].tolist()


def _run_real_component(pd, tmp_path, body, chunk_size):
    pytest.importorskip("pyarrow")
    from pandas.io.parquet import get_engine

    get_engine("pyarrow")
    read_csv = pd.read_csv

    def read_chunks(stream, **kwargs):
        kwargs["chunksize"] = chunk_size
        return read_csv(stream, **kwargs)

    artifact = _make_test_artifact(tmp_path)
    with (
        _mock_boto3_module(get_object_return={"Body": io.BytesIO(body.encode())}),
        mock.patch.object(pd, "read_csv", side_effect=read_chunks),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    return (
        pd.concat(
            [
                pd.read_parquet(path)
                for path in (result.models_selection_train_data_path, result.extra_train_data_path, artifact.path)
            ]
        )
        .sort_values(["item_id", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("first_year", [2000, 2000.5])
@pytest.mark.parametrize("chunk_size", [1, 17, 10000])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_numeric_timestamp_report_is_json_serializable(real_pandas, tmp_path, first_year, chunk_size):
    """Numeric years survive production Parquet export and sampling profile serialization."""
    years = [first_year + offset for offset in range(100)]
    body = "item_id,timestamp,target\n" + "\n".join(f"A,{year},1" for year in years)
    actual = _run_real_component(real_pandas, tmp_path, body, chunk_size)
    report = json.loads((tmp_path / "component_status" / "series_sampling_profile.json").read_text())
    series = report["datasets"]["retained"][0]
    assert actual["timestamp"].tolist() == years
    assert series["rows"] == series["source_rows_seen"] == 100
    assert float(series["timestamp_min"]) == float(series["source_timestamp_min"]) == years[0]
    assert float(series["timestamp_max"]) == float(series["source_timestamp_max"]) == years[-1]


@pytest.mark.parametrize("year_token,year_date", [("2000", "2000-01-01"), ("2000.5", "2000-07-02")])
@pytest.mark.parametrize("order", ["years_first", "dates_first", "shuffled"])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_mixed_year_and_iso_timestamps_are_independent_of_chunk_size(
    real_pandas, tmp_path, year_token, year_date, order
):
    """Normalize mixed years before comparisons, including already-capped numeric buffers."""
    import random

    pd = real_pandas
    rows = [("A", year_token, 999)] + [
        ("A", date.date().isoformat(), day)
        for day, date in enumerate(pd.date_range("2000-01-02", periods=300), start=1)
    ]
    if order == "dates_first":
        rows = rows[1:] + rows[:1]
    elif order == "shuffled":
        random.Random(17).shuffle(rows)
    body = "item_id,timestamp,target\n" + "\n".join(",".join(map(str, row)) for row in rows)
    expected = pd.DataFrame(rows, columns=["item_id", "timestamp", "target"])
    expected["timestamp"] = pd.to_datetime(expected["timestamp"].replace({year_token: year_date}))
    expected = expected.drop_duplicates(["item_id", "timestamp"], keep="last").sort_values("timestamp")
    itertuples = pd.DataFrame.itertuples

    class SizedTuple(tuple):
        def __sizeof__(self):
            return 500_000

    def sized_rows(frame, **kwargs):
        return (SizedTuple(row) for row in itertuples(frame, **kwargs))

    results, profiles = [], []
    for chunk_size in (1, 17, 10000):
        workspace = tmp_path / str(chunk_size)
        workspace.mkdir()
        with mock.patch.object(pd.DataFrame, "itertuples", sized_rows):
            actual = _run_real_component(pd, workspace, body, chunk_size)
        assert 100 <= len(actual) < len(expected)
        pd.testing.assert_frame_equal(actual, expected.tail(len(actual)).reset_index(drop=True))
        results.append(actual)
        profiles.append(json.loads((tmp_path / "component_status" / "series_sampling_profile.json").read_text()))
    for result in results[1:]:
        pd.testing.assert_frame_equal(results[0], result)
    assert profiles[0] == profiles[1] == profiles[2]
    series = profiles[0]["datasets"]["retained"][0]
    assert series["source_rows_seen"] == len(rows)
    assert series["source_timestamp_min"] == str(expected["timestamp"].min())
    assert series["source_timestamp_max"] == str(expected["timestamp"].max())


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_sampling_cannot_export_one_observation_per_series_for_training(tmp_path):
    """A large panel must fail in the loader when the budget erases usable history."""
    from .test_component_unit import _run_loader

    body = "item_id,timestamp,target,feature\n" + "\n".join(
        f"U{item:03d},{_date_from_day_offset(day)},{day},0" for item in range(100) for day in range(30)
    )
    with mock.patch.object(MockedDataFrame, "BYTES_PER_ROW", 800_000):
        with pytest.raises(ValueError, match=r"at least 6 observations.*the longest has 1.*Increase preset"):
            _run_loader(tmp_path, body)
        assert not (tmp_path / "datasets" / "models_selection_train_dataset.parquet").exists()
        result, artifact = _run_loader(tmp_path, body, preset="quality")
    selection = _read_csv_rows(result.models_selection_train_data_path)
    assert result.sample_config["sampled_rows"] == 3000
    assert len(selection) == 700
    assert len(_read_csv_rows(artifact.path)) == 600


@pytest.mark.parametrize("preset,series_count,length", [("speed", 500, 70), ("balanced", 4538, 30)])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_wide_numeric_panel_retains_trainable_history(real_pandas, tmp_path, preset, series_count, length):
    """Real 115-column panels keep useful histories without artificial per-cell charges."""
    pytest.importorskip("pyarrow")
    from pandas.io.parquet import get_engine

    get_engine("pyarrow")
    pd = real_pandas
    columns = ["item_id", "timestamp", "target"] + [f"feature_{i}" for i in range(112)]
    features = ",".join(["0.5"] * 112)
    body = (
        ",".join(columns)
        + "\n"
        + "\n".join(
            f"U{item:07d},{_date_from_day_offset(day)},{day},{features}"
            for item in range(series_count)
            for day in range(length)
        )
    )
    artifact = _make_test_artifact(tmp_path)
    with _mock_boto3_module(get_object_return={"Body": io.BytesIO(body.encode())}):
        result = timeseries_data_loader.python_func(
            file_key="wide_panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            preset=preset,
            sampled_test_dataset=artifact,
        )
    selection = pd.read_parquet(result.models_selection_train_data_path)
    assert list(selection.columns) == columns
    assert selection.groupby("item_id").size().min() >= 6
    assert selection["item_id"].nunique() == series_count
    retained = pd.concat([selection, pd.read_parquet(result.extra_train_data_path), pd.read_parquet(artifact.path)])
    assert retained.memory_usage(deep=True).sum() <= {"speed": 100 * 1024**2, "balanced": 1024**3}[preset]
    for _, series in retained.groupby("item_id", sort=False):
        days = sorted(series["target"].tolist())
        assert days == list(range(length - len(days), length))
    assert result.sample_config["source_rows_seen"] == series_count * length
    status = json.loads((tmp_path / "component_status" / "component_status.json").read_text())
    prepare = next(stage for stage in status["stages"] if stage["id"] == "prepare_data")
    assert prepare["metrics"]["sampled_buffer_estimated_bytes"] <= prepare["metrics"]["sample_cap_bytes"]
    if preset == "speed":
        assert result.sample_config["sampled_rows"] < series_count * length
    else:
        assert result.sample_config["sampled_rows"] == series_count * length


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_numeric_dtype_changes_do_not_change_sample_window(real_pandas, tmp_path):
    """Integer/float inference across chunks must not change the retained histories."""
    columns = ["item_id", "timestamp", "target", "feature"] + [f"numeric_{i}" for i in range(111)]
    body = (
        ",".join(columns)
        + "\n"
        + "\n".join(
            f"{item},{_date_from_day_offset(day)},{day},medium," + ",".join(["0.5" if day == 0 else "0"] * 111)
            for item in ("A", "B")
            for day in range(300)
        )
    )
    results = []
    for chunk_size in (17, 10000):
        workspace = tmp_path / str(chunk_size)
        workspace.mkdir()
        with mock.patch.object(sys, "getsizeof", side_effect=_variable_value_size):
            results.append(_run_real_component(real_pandas, workspace, body, chunk_size))
    real_pandas.testing.assert_frame_equal(*results)
    assert 100 <= len(results[0]) < 600
    for _, series in results[0].groupby("item_id"):
        assert series["target"].tolist() == list(range(300 - len(series), 300))


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_selection_history_validation_uses_prediction_length(tmp_path):
    """A nonempty selection split still needs enough history for the forecast horizon."""
    with (
        _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(_panel_csv(100).encode())}),
        pytest.raises(ValueError, match=r"at least 41 observations.*prediction_length=20"),
    ):
        timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            prediction_length=20,
            sampled_test_dataset=_make_test_artifact(tmp_path),
        )


def _variable_value_size(value, *args):
    """Simulate different normal-row costs without allocating large CSV fields."""
    if isinstance(value, str):
        if value == "costly":
            return 60 * 1024 * 1024
        if value == "medium":
            return 200_000
    return _original_getsizeof(value, *args)


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
@pytest.mark.parametrize("chunk_size", [1, 17, 10000])
@pytest.mark.parametrize("costly_series_length", [1, 300])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_heterogeneous_row_costs_preserve_cheaper_series_history(
    real_pandas, tmp_path, order, chunk_size, costly_series_length
):
    """One costly normal row must not collapse every series to a single observation."""
    rows = [f"A,{_date_from_day_offset(0)},0,costly"]
    rows.extend(f"A,{_date_from_day_offset(day)},{day},small" for day in range(1, costly_series_length))
    rows.extend(f"B,{_date_from_day_offset(day)},{day},medium" for day in range(300))
    if order == "reversed":
        rows.reverse()
    elif order == "shuffled":
        import random

        random.Random(17).shuffle(rows)
    body = "item_id,timestamp,target,feature\n" + "\n".join(rows)
    with mock.patch.object(sys, "getsizeof", side_effect=_variable_value_size):
        actual = _run_real_component(real_pandas, tmp_path, body, chunk_size)
    assert set(actual["item_id"]) == {"A", "B"}
    costly = actual[actual["item_id"] == "A"]
    cheaper = actual[actual["item_id"] == "B"]
    assert costly["target"].tolist() == [costly_series_length - 1]
    assert 100 <= len(cheaper) < 300
    assert cheaper["target"].tolist() == list(range(300 - len(cheaper), 300))
    assert 60 * 1024 * 1024 + len(cheaper) * 200_000 <= 100 * 1024 * 1024
    profile = json.loads((tmp_path / "component_status" / "series_sampling_profile.json").read_text())
    assert all(series["oversized_rows_seen"] == 0 for series in profile["datasets"]["retained"])


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_normal_row_costs_are_not_capped_per_series(real_pandas, tmp_path):
    """Fail when even one uncapped normal row per series cannot fit the total budget."""
    body = "item_id,timestamp,target,feature\n"
    body += "\n".join(f"{item},{_date_from_day_offset(0)},0,costly" for item in ("A", "B"))
    with (
        mock.patch.object(sys, "getsizeof", side_effect=_variable_value_size),
        pytest.raises(ValueError, match="cannot retain one row per series"),
    ):
        _run_real_component(real_pandas, tmp_path, body, 17)
