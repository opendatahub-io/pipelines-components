"""Exercise the fallback on real pandas timestamps without mocking frequency logic."""

import pandas as pd
import pytest

from ..timeseries_frequency import infer_frequency_from_gaps, validate_frequency_grid


def panel(*indexes):
    """Build AutoGluon's item/timestamp index without depending on AutoGluon."""
    return pd.concat(
        [
            pd.DataFrame(
                {"target": range(len(index))},
                index=pd.MultiIndex.from_product([[str(item)], index], names=["item_id", "timestamp"]),
            )
            for item, index in enumerate(indexes)
        ]
    )


@pytest.mark.parametrize("freq", ["D", "h", "15min", "100ns", "7D", "MS", "ME", "3MS", "3ME", "12MS"])
def test_infers_fixed_and_calendar_grids_with_gaps(freq):
    """Gaps preserve the data's cadence, values, and timestamps when resampled."""
    grid = pd.date_range("2020-01-01", periods=18, freq=freq)
    data = panel(grid.delete([2, 5]), grid.delete([3, 7])).sample(frac=1, random_state=42)
    original = data.copy(deep=True)
    inferred = infer_frequency_from_gaps(data)
    assert inferred == pd.tseries.frequencies.to_offset(freq).freqstr
    for _, series in data.groupby(level="item_id"):
        values = series.droplevel("item_id").sort_index()
        resampled = values.resample(inferred).mean()
        pd.testing.assert_frame_equal(resampled.loc[values.index], values.astype(float), check_freq=False)
        pd.testing.assert_index_equal(resampled.index, grid.rename("timestamp"))
    pd.testing.assert_frame_equal(data, original)


def test_kaggle_daily_mapping():
    """Three series sharing the supplied 2433 -> 2436 gap resolve to daily data."""
    dates = pd.to_datetime(["1976-08-30", "1976-09-02", "1976-09-03", "1976-09-04", "1976-09-05"])
    assert infer_frequency_from_gaps(panel(dates, dates, dates)) == "D"


def test_short_series_are_validated_against_supported_grid():
    """A short history cannot vote but can share a grid supported by other items."""
    grid = pd.date_range("2020-01-01", periods=12, freq="h")
    assert infer_frequency_from_gaps(panel(grid.delete(2), grid[[0, 3]], grid[[0]])) == "h"


@pytest.mark.parametrize(
    "indexes",
    [
        [pd.date_range("2020-01-01", periods=12, freq="h"), pd.date_range("2020-01-01", periods=12, freq="D")],
        [pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])],  # no dominant step
        [pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05", "2020-01-08"])],  # non-multiple gap
        [pd.date_range("2020-01-01", periods=4, freq="D"), pd.to_datetime(["2020-01-02 12:00"])],
        [pd.date_range("2020-01-01 00:15", periods=6, freq="h").delete(2)],  # resampling would move rows
        [pd.date_range("2020-01-01", periods=12, freq="B").delete(3)],  # daily vs business days
        [pd.date_range("2020-01-01", periods=2, freq="h")],  # insufficient evidence
        [pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])],
        [pd.to_datetime(["2020-01-01", "2020-01-02", None])],
    ],
)
def test_ambiguous_or_incompatible_data_requires_explicit_frequency(indexes):
    """Reject mixed cadences, ambiguity and any conversion that would move data."""
    with pytest.raises(ValueError, match="Set freq explicitly"):
        infer_frequency_from_gaps(panel(*indexes))


def test_large_expansion_is_rejected_before_constructing_a_grid(monkeypatch):
    """A large gap cannot turn a tiny input into an unbounded resampling job."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2026-01-01"])

    def unexpected_grid(*args, **kwargs):
        raise AssertionError("Expansion must be checked before constructing a grid")

    monkeypatch.setattr(pd, "date_range", unexpected_grid)
    with pytest.raises(ValueError, match="expand a series by more than 10x"):
        infer_frequency_from_gaps(panel(dates))


@pytest.mark.parametrize("freq", ["h", "D", "7D", "W-MON", "B", "MS", "ME", "3MS", "3ME", "YS"])
def test_validate_grid_allows_missing_periods(freq):
    """Grid validation permits gaps without requiring the holdout to infer frequency."""
    dates = pd.date_range("2020-01-01", periods=18, freq=freq).delete([3, 7])
    data = panel(dates)
    original = data.copy(deep=True)
    validate_frequency_grid(data, freq, name="test data")
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(("start", "freq"), [("2020-01-01 12:00", "D"), ("2020-01-01 00:15", "h")])
def test_regular_non_midnight_panels_keep_existing_behavior(start, freq):
    """AutoGluon skips resampling when all series strictly match the frequency."""
    validate_frequency_grid(panel(pd.date_range(start, periods=12, freq=freq)), freq)


@pytest.mark.parametrize(("start", "freq"), [("2020-01-01 12:00", "D"), ("2020-01-01 00:15", "h")])
@pytest.mark.parametrize("length", [1, 2])
def test_short_regular_series_use_shared_grid_with_index_metadata(start, freq, length):
    """Short items can avoid resampling when AutoGluon can use index metadata."""
    dates = pd.date_range(start, periods=12, freq=freq)
    data = panel(dates, dates[:length])
    assert data.index.levels[1].freq == pd.tseries.frequencies.to_offset(freq)
    original = data.copy(deep=True)
    validate_frequency_grid(data, freq)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("positions", [[0], [0, 1], [0, 2]])
@pytest.mark.parametrize("freq", ["D", "h", "MS"])
def test_short_aligned_series_allow_safe_resampling(positions, freq):
    """Short items on the resampling grid remain valid even without metadata."""
    dates = pd.date_range("2020-01-01", periods=12, freq=freq)
    data = panel(dates, dates[positions])
    data.index = data.index.set_levels(pd.DatetimeIndex(data.index.levels[1].values), level="timestamp")
    validate_frequency_grid(data, freq)


@pytest.mark.parametrize("positions", [[0], [0, 1], [0, 2]])
def test_short_non_midnight_series_without_metadata_require_resampling(positions):
    """Without metadata, AutoGluon resamples even short items on a daily noon grid."""
    dates = pd.date_range("2020-01-01 12:00", periods=12, freq="D")
    data = panel(dates, dates[positions])
    data.index = data.index.set_levels(pd.DatetimeIndex(data.index.levels[1].values), level="timestamp")
    assert data.index.levels[1].freq is None
    with pytest.raises(ValueError, match="incompatible"):
        validate_frequency_grid(data, "D")


@pytest.mark.parametrize("short_dates", [["2020-01-01 12:00", "2020-01-03 12:00"], ["2020-01-01 13:00"]])
def test_short_incompatible_series_do_not_bypass_alignment(short_dates):
    """Neither gaps nor shifted timestamps in short items can bypass validation."""
    dates = pd.date_range("2020-01-01 12:00", periods=12, freq="D")
    with pytest.raises(ValueError, match="incompatible"):
        validate_frequency_grid(panel(dates, pd.to_datetime(short_dates)), "D")


def test_one_irregular_item_causes_all_items_to_be_validated_for_resampling():
    """AutoGluon resamples the whole panel when even one item is irregular."""
    regular_noon = pd.date_range("2020-01-01 12:00", periods=12, freq="D")
    gappy_midnight = pd.date_range("2020-01-01", periods=12, freq="D").delete(3)
    with pytest.raises(ValueError, match="incompatible"):
        validate_frequency_grid(panel(regular_noon, gappy_midnight), "D", name="test data")


def test_test_data_cannot_be_silently_aggregated_to_training_frequency():
    """An hourly holdout must not be averaged into daily evaluation targets."""
    data = panel(pd.date_range("2020-01-01", periods=48, freq="h"))
    with pytest.raises(ValueError, match="test data.*incompatible"):
        validate_frequency_grid(data, "D", name="test data")


def test_full_training_data_reveals_incompatible_extra_timestamps():
    """A regular selection prefix alone must not determine the refit frequency."""
    selection = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    assert pd.infer_freq(selection) == "D"
    full_train = selection.append(pd.to_datetime(["2020-01-03 12:00"]))
    with pytest.raises(ValueError, match="Set freq explicitly"):
        infer_frequency_from_gaps(panel(full_train))


@pytest.mark.parametrize("freq", ["h", "MS", "ME", "B"])
def test_split_expansion_is_bounded_before_allocating_grid(freq, monkeypatch):
    """Even a two-row holdout cannot expand without a bound, including calendar offsets."""
    dates = pd.date_range("2020-01-01", periods=128, freq=freq)[[0, -1]]
    data = panel(dates)

    def unexpected_grid(*args, **kwargs):
        raise AssertionError("Expansion must be checked before allocating the grid")

    monkeypatch.setattr(pd, "date_range", unexpected_grid)
    with pytest.raises(ValueError, match="more than 10x in test data"):
        validate_frequency_grid(data, freq, name="test data")
