"""Latest-per-series regression tests using real pandas and in-memory S3 objects."""

import io
import json
import logging
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ..component import timeseries_data_loader
from ..sampling import load_latest_timeseries
from .test_component_unit import _mock_boto3_module, mocked_env_variables

pd = pytest.importorskip("pandas")


def _panel(lengths=(2000, 2000)):
    """Build independent series whose most recent observations occur at the file end."""
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "item_id": f"series-{index}",
                    "timestamp": pd.date_range("2020-01-01", periods=length),
                    "target": range(length),
                }
            )
            for index, length in enumerate(lengths)
        ],
        ignore_index=True,
    )


def _sample(frame, budget=24 * 1024, chunk_size=113, version=None, etag='"original"', second_etag=None):
    """Run the sampler with fresh streams and record the S3 snapshot conditions."""
    payload = frame.to_csv(index=False).encode()
    calls = []
    bodies = []

    def open_source(conditions):
        calls.append(conditions)
        body = io.BytesIO(payload)
        bodies.append(body)
        return {"Body": body, "ETag": second_etag if len(calls) == 2 and second_etag else etag, "VersionId": version}

    def prepare(data):
        data = data.replace([float("inf"), float("-inf")], float("nan"))
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_localize(None)
        return data.drop_duplicates(["item_id", "timestamp"], keep="last")

    report = {}
    result = load_latest_timeseries(
        open_source, prepare, "item_id", "timestamp", budget, chunk_size, report, logging.getLogger(__name__)
    )
    assert all(body.closed for body in bodies)
    return result, report, calls


@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled", "interleaved"])
def test_oversized_panel_retains_latest_observations_of_every_series(order):
    """The holdout candidate panel contains the source tail per ID, in any file order."""
    source = _panel()
    if order == "descending":
        source = source.iloc[::-1]
    elif order == "shuffled":
        source = source.sample(frac=1, random_state=17)
    elif order == "interleaved":
        source = source.sort_values(["timestamp", "item_id"])
    result, report, calls = _sample(source)
    assert set(result.item_id) == {"series-0", "series-1"}
    for key, group in result.groupby("item_id"):
        assert sorted(group.target) == list(range(2000 - len(group), 2000))
    assert int(result.memory_usage(deep=True).sum()) <= 24 * 1024
    assert report["source_rows_read"] == 4000
    assert report["cap_reached"] and not report["partial_read"]
    assert calls == [{}, {"IfMatch": '"original"'}]


def test_shuffling_and_chunk_boundaries_do_not_change_sample():
    """Fixed quotas and global timestamp comparison give the same sample after a shuffle."""
    source = _panel()
    ordered, _, _ = _sample(source, chunk_size=97)
    shuffled, _, _ = _sample(source.sample(frac=1, random_state=7), chunk_size=211)
    pd.testing.assert_frame_equal(
        ordered.sort_values(["item_id", "timestamp"]).reset_index(drop=True),
        shuffled.sort_values(["item_id", "timestamp"]).reset_index(drop=True),
    )


def test_short_series_retain_all_rows_and_long_series_use_remaining_budget():
    """A short series does not lose data merely because another series is large."""
    result, _, _ = _sample(_panel((7, 2000, 2000)))
    lengths = result.groupby("item_id").size().to_dict()
    assert lengths["series-0"] == 7
    assert lengths["series-1"] == lengths["series-2"] > 7


def test_cross_chunk_duplicates_keep_last_without_displacing_unique_tail():
    """Later replacements of the newest keys survive; repeated keys consume no quota."""
    source = _panel()
    replacements = source[source.target >= 1995].copy()
    replacements["target"] += 10000
    result, _, _ = _sample(pd.concat([source, replacements, replacements], ignore_index=True), chunk_size=113)
    assert not result.duplicated(["item_id", "timestamp"]).any()
    assert set(result[result.timestamp >= pd.Timestamp("2025-06-18")].target) == set(range(11995, 12000))
    for _, group in result.groupby("item_id"):
        assert group.timestamp.diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_variable_width_rows_and_dtype_promotion_stay_within_budget():
    """Wide strings and numeric-to-object promotion are included in conservative row bounds."""
    source = _panel()
    source["feature"] = [str(i) if i % 9 else "long" * 80 for i in range(len(source))]
    source["feature_2"] = [str(i) if i < 2000 else "value" for i in range(len(source))]
    result, _, _ = _sample(source, budget=100 * 1024)
    assert int(result.memory_usage(deep=True).sum()) <= 100 * 1024
    assert result.groupby("item_id").target.max().to_dict() == {"series-0": 1999, "series-1": 1999}


def test_small_input_uses_one_read_and_preserves_source_rows():
    """The fast path preserves original values and dtypes and does not require rereading."""
    source = _panel((100, 100))
    result, report, calls = _sample(source, budget=1024 * 1024)
    expected = pd.read_csv(io.StringIO(source.to_csv(index=False)))
    expected["timestamp"] = pd.to_datetime(expected.timestamp)
    pd.testing.assert_frame_equal(result, expected)
    assert calls == [{}]
    assert report["read_passes"] == 1 and not report["cap_reached"]


def test_versioned_object_is_reread_by_version_id():
    """A version ID takes precedence over the ETag when S3 versioning is enabled."""
    _, _, calls = _sample(_panel(), version="immutable-version")
    assert calls[-1] == {"VersionId": "immutable-version"}


def test_changed_object_and_missing_snapshot_are_rejected():
    """The sampler cannot silently mix two versions of an unversioned CSV."""
    with pytest.raises(ValueError, match="ETag mismatch"):
        _sample(_panel(), second_etag='"changed"')
    with pytest.raises(ValueError, match="without an ETag or VersionId"):
        _sample(_panel(), etag=None)


def test_integer_and_float_id_chunks_form_one_numeric_series():
    """Chunk-local int/float inference must not split the same item into separate buffers."""
    source = _panel((4000,))
    source["item_id"] = ["1"] * 2000 + ["1.0"] * 2000
    result, _, _ = _sample(source, chunk_size=100)
    assert result.item_id.nunique() == 1
    assert result.item_id.iloc[0] == 1.0
    assert result.target.max() == 3999


@pytest.mark.parametrize("failed_pass", [1, 2])
def test_partial_read_is_explicit_and_reread_errors_are_fatal(monkeypatch, failed_pass):
    """Recover the inventoried prefix only; a failed sampling reread cannot produce a successful sample."""
    read_csv = pd.read_csv
    reads = 0

    def flaky_reader(*args, **kwargs):
        nonlocal reads
        reads += 1
        current_pass = reads
        for index, chunk in enumerate(read_csv(*args, **kwargs)):
            yield chunk
            if current_pass == failed_pass and index == 20:
                raise OSError("connection reset")

    monkeypatch.setattr(pd, "read_csv", flaky_reader)
    if failed_pass == 2:
        with pytest.raises(ValueError, match="Error rereading CSV"):
            _sample(_panel())
    else:
        result, report, _ = _sample(_panel())
        assert report["partial_read"] and report["source_rows_read"] == 21 * 113
        assert result[result.item_id == "series-1"].target.max() == 21 * 113 - 2001


def test_high_cardinality_is_rejected_before_inventory_outgrows_budget():
    """A fixed sample budget cannot support unlimited per-series metadata."""
    with pytest.raises(ValueError, match="Too many time series"):
        _sample(_panel((100,) * 20))


@pytest.fixture
def run_component(tmp_path, monkeypatch):
    """Run the real loader with a small sampling cap while preserving the selected preset."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas.core.arrays.arrow.extension_types")
    artifacts = SimpleNamespace(path=str(tmp_path / "test.parquet"), uri="/test", metadata={})
    status = SimpleNamespace(path=str(tmp_path / "status"), metadata={})
    budgets = []

    def reduced_budget(*args, **kwargs):
        args = list(args)
        budgets.append(args[4])
        args[4] = 24 * 1024
        args[5] = 113
        return load_latest_timeseries(*args, **kwargs)

    sampling_module = import_module("kfp_components.components.data_processing.automl.timeseries_data_loader.sampling")
    monkeypatch.setattr(sampling_module, "load_latest_timeseries", reduced_budget)

    def run(source, preset="speed", id_column="item_id"):
        payload = source.to_csv(index=False).encode()
        with mock.patch.dict("os.environ", mocked_env_variables), _mock_boto3_module() as client:
            client.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(payload), "ETag": '"original"'}
            result = timeseries_data_loader.python_func(
                file_key="train.csv",
                bucket_name="bucket",
                workspace_path=str(tmp_path),
                target="target",
                timestamp_column="timestamp",
                id_column=id_column,
                preset=preset,
                sampled_test_dataset=artifacts,
                component_status=status,
            )
        return result, artifacts, status, budgets

    return run


@pytest.mark.parametrize("preset", ["speed", "balanced", "quality"])
def test_component_exports_recent_splits_and_profile_without_changing_preset(run_component, preset):
    """Exercise the real component, Parquet exports, temporal splits and profile for all tiers."""
    result, artifacts, status, budgets = run_component(_panel(), preset=preset)
    assert budgets == [{"speed": 100 * 1024**2, "balanced": 1024**3, "quality": 10 * 1024**3}[preset]]
    selection = pd.read_parquet(result.models_selection_train_data_path)
    extra = pd.read_parquet(result.extra_train_data_path)
    holdout = pd.read_parquet(artifacts.path)
    for key, group in holdout.groupby("item_id"):
        assert group.target.max() == 1999
        assert selection[selection.item_id == key].timestamp.max() < extra[extra.item_id == key].timestamp.min()
        assert extra[extra.item_id == key].timestamp.max() < group.timestamp.min()
    profile = [
        json.loads(line) for line in (Path(status.path) / result.sample_config["profile_file"]).read_text().splitlines()
    ]
    retained = pd.concat([selection, extra, holdout])
    for series in profile:
        group = retained[retained.item_id == series["item_id"]]
        assert series["retained_rows"] == len(group)
        assert series["min_timestamp"] == group.timestamp.min().isoformat()
        assert series["max_timestamp"] == group.timestamp.max().isoformat()
    metrics = json.loads((Path(status.path) / "component_status.json").read_text())["stages"][0]["metrics"]
    assert metrics["sample_cap_reached"]
    assert result.sample_config["source_rows_read"] == 4000
    assert result.sample_config["source_read_complete"]
    assert result.sample_config["sampling_method"] == "last_values_per_series"


def test_sampling_below_minimum_record_count_still_fails(run_component):
    """Wide rows can leave fewer than 100 unique observations even when every series is represented."""
    source = _panel()
    source["feature"] = "x" * 1000
    with pytest.raises(ValueError, match="at least 100"):
        run_component(source)


@pytest.mark.parametrize("kind", ["synthetic_id", "fractional_year", "timezone", "missing_target"])
def test_large_input_preserves_existing_timeseries_formats_and_missing_value_rules(run_component, kind):
    """The bounded path supports existing IDs, fractional years, UTC normalization and missing targets."""
    source = _panel((4000,))
    id_column = "item_id"
    if kind == "synthetic_id":
        source = source.drop(columns="item_id")
        id_column = ""
    elif kind == "fractional_year":
        source["timestamp"] = 2000 + source.target / 1000
    elif kind == "timezone":
        source["timestamp"] = source.timestamp.dt.tz_localize("Europe/Warsaw").astype(str)
    else:
        source["target"] = source.target.astype(float)
        source.loc[source.target == 3999, "target"] = float("inf")
    result, artifact, _, _ = run_component(source, id_column=id_column)
    test = pd.read_parquet(artifact.path)
    if kind == "synthetic_id":
        assert result.uses_synthetic_id and set(test[result.effective_id_column]) == {"item_0"}
    elif kind == "fractional_year":
        assert test.timestamp.max() == 2003.999
    elif kind == "timezone":
        assert test.timestamp.max() == pd.Timestamp(source.timestamp.iloc[-1]).tz_convert("UTC").tz_localize(None)
    else:
        assert pd.isna(test.loc[test.timestamp == source.timestamp.max(), "target"]).all()


def test_invalid_timestamp_outside_old_head_is_validated(run_component):
    """Bad dates beyond the old head budget must be reported instead of silently skipped."""
    source = _panel()
    source["timestamp"] = source.timestamp.astype(str)
    source.loc[len(source) - 1, "timestamp"] = "invalid"
    with pytest.raises(ValueError, match="could not be parsed"):
        run_component(source)
