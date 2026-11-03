"""Real pandas/Parquet coverage of bounded per-series sampling and disk recovery."""

import json
from unittest import mock

import pandas as pd
import pytest
from kfp_components.components.training.automl.shared import timeseries_sampling as sampling


@pytest.fixture(autouse=True)
def python_string_storage():
    """Keep fixture and Parquet-read storage equal for exact byte/count assertions.

    Parquet retains the logical string dtype but not pandas' storage backend.
    Loader integration tests separately exercise the environment's default storage.
    """
    with pd.option_context("mode.string_storage", "python"):
        yield


def _panel(lengths):
    frames = []
    for series, n in lengths.items():
        frames.append(
            pd.DataFrame(
                {
                    # Preserve the same storage through Parquet, regardless of pandas' inference defaults.
                    "id": pd.Series([series] * n, dtype=pd.StringDtype(storage="python")),
                    "time": pd.date_range("2020-01-01", periods=n),
                    "target": range(n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _run(tmp_path, frame, budget, chunk_size=17, minimum=1):
    chunks = (frame.iloc[i : i + chunk_size].copy() for i in range(0, len(frame), chunk_size))
    return sampling.sample_latest_timeseries(
        chunks,
        id_column="id",
        timestamp_column="time",
        max_size_bytes=budget,
        workspace_path=tmp_path,
        profile_path=tmp_path / "profile.jsonl",
        minimum_rows_per_series=minimum,
    )


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled", "interleaved"])
def test_disk_tails_match_full_sort_for_every_input_order(tmp_path, order):
    """The latest observations of every series survive even when IDs occur late."""
    frame = _panel({"A": 100, "B": 100})
    if order == "reversed":
        frame = frame.iloc[::-1]
    elif order == "shuffled":
        frame = frame.sample(frac=1, random_state=42)
    elif order == "interleaved":
        frame = frame.sort_values(["time", "id"])
    row_bytes = int(frame.memory_usage(index=False, deep=True).sum() // len(frame))
    budget = 132 + 100 * row_bytes
    result, report = _run(tmp_path, frame, budget)
    expected = frame.sort_values(["id", "time"]).groupby("id").tail(50).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)
    assert report["used_disk"] is True
    assert report["scanned_rows"] == 200
    assert report["full_source_read"] is True
    assert report["sampled_in_memory_bytes"] <= budget
    assert report["sampled_rows"] == 100
    assert not list(tmp_path.glob("timeseries-sampling-*"))
    profile = [json.loads(line) for line in (tmp_path / "profile.jsonl").read_text().splitlines()]
    assert len(profile) == 2
    for series in profile:
        assert series["source_rows"] == 100
        assert series["retained_rows"] == 50
        assert series["retained_min_timestamp"].startswith("2020-02-20")
        assert series["retained_max_timestamp"] == series["source_max_timestamp"]


def test_unused_short_series_budget_is_redistributed(tmp_path):
    """Short histories do not reserve bytes that longer histories could use."""
    frame = _panel({"A": 100, "B": 10, "C": 500})
    row_bytes = int(frame.memory_usage(index=False, deep=True).sum() // len(frame))
    result, _ = _run(tmp_path, frame, 132 + 110 * row_bytes)
    assert result.groupby("id").size().to_dict() == {"A": 50, "B": 10, "C": 50}
    assert result.groupby("id")["target"].max().to_dict() == {"A": 99, "B": 9, "C": 499}


@pytest.mark.parametrize("disk", [False, True])
def test_cross_chunk_duplicates_keep_last_file_occurrence(tmp_path, disk):
    """Duplicate updates replace earlier values without consuming tail slots."""
    frame = _panel({"A": 100, "B": 100})
    updates = frame[frame["target"] >= 95].copy()
    updates["target"] = 999
    source = pd.concat([frame, updates], ignore_index=True)
    budget = int(frame.memory_usage(index=False, deep=True).sum() // 2) + 132 if disk else 100_000
    result, report = _run(tmp_path, source, budget, chunk_size=23)
    assert report["duplicate_rows"] == 10
    assert report["source_unique_rows"] == 200
    assert result.groupby("id").tail(5)["target"].eq(999).all()
    assert not result.duplicated(["id", "time"]).any()


def test_small_input_never_creates_spool(tmp_path):
    """The RAM path keeps the complete source and uses real typed Parquet downstream."""
    frame = _panel({"A": 50, "B": 50}).sample(frac=1, random_state=8)
    with mock.patch.object(sampling, "_Spool", side_effect=AssertionError("unnecessary spill")):
        result, report = _run(tmp_path, frame, 100_000)
    pd.testing.assert_frame_equal(result, frame.sort_values(["id", "time"]).reset_index(drop=True))
    assert report["used_disk"] is False
    assert report["cap_reached"] is False


def test_nanosecond_timestamp_keys_are_not_rounded(tmp_path):
    """SQLite keys preserve nanosecond distinctions at contemporary epoch magnitudes."""
    frame = _panel({"A": 100})
    frame["time"] = pd.date_range("2026-01-01", periods=100, freq="ns")
    result, report = _run(tmp_path, frame.sample(frac=1, random_state=1), 2000)
    assert report["source_unique_rows"] == 100
    pd.testing.assert_frame_equal(result, frame.tail(len(result)).reset_index(drop=True))


def test_fractional_years_keep_numeric_order(tmp_path):
    """Legacy fractional years are indexed as numeric values, not lexicographic strings."""
    frame = _panel({"A": 100})
    frame["time"] = [2000 + i / 12 for i in range(100)]
    result, _ = _run(tmp_path, frame.sample(frac=1, random_state=2), 2000)
    pd.testing.assert_frame_equal(result, frame.tail(len(result)).reset_index(drop=True))


def test_dtype_promotion_after_spill_still_obeys_pandas_budget(tmp_path):
    """Chunk-local numeric/string inference cannot inflate the final output past its cap."""
    first = _panel({"A": 100})
    first["covariate"] = range(100)
    second = _panel({"B": 100})
    second["covariate"] = "category"
    result, _ = sampling.sample_latest_timeseries(
        iter([first, second]),
        id_column="id",
        timestamp_column="time",
        max_size_bytes=5000,
        workspace_path=tmp_path,
        profile_path=tmp_path / "profile.jsonl",
    )
    assert int(result.memory_usage(deep=True).sum()) <= 5000
    assert result.groupby("id")["target"].max().to_dict() == {"A": 99, "B": 99}


def test_budget_cannot_shorten_history_below_minimum(tmp_path):
    """The sampler fails instead of silently dropping IDs or unusably truncating them."""
    frame = _panel({"A": 100, "B": 100})
    budget = int(frame.head(20).reset_index(drop=True).memory_usage(deep=True).sum())
    with pytest.raises(ValueError, match="sufficient history"):
        _run(tmp_path, frame, budget, minimum=30)
    assert not list(tmp_path.glob("timeseries-sampling-*"))


def test_wide_rows_do_not_break_byte_budget_or_tail_continuity(tmp_path):
    """Uneven string widths use actual row costs rather than a fixed row allowance."""
    frame = _panel({"A": 100, "B": 100})
    frame["text"] = ["x" * (i % 13 * 100) for i in range(len(frame))]
    result, report = _run(tmp_path, frame.sample(frac=1, random_state=4), 30_000)
    assert int(result.memory_usage(deep=True).sum()) <= 30_000
    for series, retained in result.groupby("id"):
        expected = frame[frame["id"] == series].tail(len(retained))
        pd.testing.assert_frame_equal(retained.reset_index(drop=True), expected.reset_index(drop=True))
    assert report["cap_reached"] is True


def test_workspace_exhaustion_is_explicit_and_cleans_spool(tmp_path, monkeypatch):
    """Insufficient scratch space never degrades into head sampling."""
    monkeypatch.setattr(sampling, "MAX_STAGING_BYTES", 1)
    with pytest.raises(ValueError, match="Insufficient workspace"):
        _run(tmp_path, _panel({"A": 100}), 2000)
    assert not list(tmp_path.glob("timeseries-sampling-*"))


def test_interrupted_source_fails_and_cleans_spool(tmp_path):
    """Partial reads cannot be advertised as the latest observations."""

    def chunks():
        yield _panel({"A": 100})
        raise OSError("connection reset")

    with pytest.raises(OSError, match="connection reset"):
        sampling.sample_latest_timeseries(
            chunks(),
            id_column="id",
            timestamp_column="time",
            max_size_bytes=2000,
            workspace_path=tmp_path,
            profile_path=tmp_path / "profile.jsonl",
        )
    assert not list(tmp_path.glob("timeseries-sampling-*"))


@pytest.mark.parametrize("horizon", [1, 7, 28, 100])
@pytest.mark.parametrize("external", [False, True])
def test_minimum_covers_selection_validation_and_test_context(horizon, external):
    """The computed floor accounts for both temporal splits and evaluation context."""
    n = sampling.minimum_retained_rows(horizon, 0.3, external)
    train = n if external else int(n * 0.8)
    assert int(train * 0.3) >= horizon + max(horizon + 1, 5)
    assert external or n - train > horizon
