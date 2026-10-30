"""Conservative frequency inference for panels with missing observations."""

from __future__ import annotations

import pandas as pd


def infer_frequency_from_gaps(data: pd.DataFrame, max_expansion: int = 10) -> str:
    """Infer a shared observation grid without moving or aggregating timestamps.

    Each series with at least three observations must support the same step with
    a strict majority of its gaps. Remaining gaps must be whole multiples of it.
    Month-start/end data is measured in calendar months, other data in elapsed
    time. Shorter series are validated against the candidate without voting.

    Ambiguous panels, off-grid timestamps and grids exceeding ``max_expansion``
    times the observed rows in any series require an explicit user frequency.
    This heuristic identifies a grid, not the business meaning of the time axis.
    """
    error = "Cannot infer a common frequency from timestamp gaps. Set freq explicitly to the expected pandas frequency."

    def dominant_step(gaps):
        counts = pd.Series(gaps).value_counts()
        if counts.empty or counts.iloc[0] * 2 <= len(gaps):
            raise ValueError(error)
        step = int(counts.index[0])
        if step <= 0 or (gaps % step != 0).any():
            raise ValueError(error)
        return step

    indexes = []
    candidates = set()
    for _, series in data.groupby(level="item_id", sort=False):
        timestamps = pd.DatetimeIndex(series.index.get_level_values("timestamp")).as_unit("ns").sort_values()
        if timestamps.hasnans or timestamps.has_duplicates:
            raise ValueError(error)
        indexes.append(timestamps)
        if len(timestamps) < 3:
            continue
        midnight = timestamps.equals(timestamps.normalize())
        if midnight and (timestamps.is_month_start.all() or timestamps.is_month_end.all()):
            months = timestamps.year.to_numpy(dtype="int64") * 12 + timestamps.month.to_numpy(dtype="int64")
            step = dominant_step(months[1:] - months[:-1])
            suffix = "MS" if timestamps.is_month_start.all() else "ME"
            candidate = pd.tseries.frequencies.to_offset(f"{step}{suffix}")
        else:
            deltas = timestamps.asi8[1:] - timestamps.asi8[:-1]
            step = dominant_step(deltas)
            candidate = pd.tseries.frequencies.to_offset(pd.Timedelta(step, unit="ns"))
        candidates.add(candidate.freqstr)

    if len(candidates) != 1:
        raise ValueError(error)
    freq = candidates.pop()
    offset = pd.tseries.frequencies.to_offset(freq)
    for timestamps in indexes:
        first, last = timestamps[0], timestamps[-1]
        if isinstance(offset, pd.offsets.Tick):
            expected_rows = (int(last.value) - int(first.value)) // offset.nanos + 1
            # AutoGluon's resampling anchors fixed offsets at the start of the day.
            if (first.value - first.normalize().value) % offset.nanos:
                raise ValueError(error)
            # Missing weekends cannot distinguish a daily from a business-day grid.
            if offset == pd.offsets.Day() and (timestamps.dayofweek < 5).all() and last - first >= pd.Timedelta(days=7):
                raise ValueError(error)
        else:
            expected_rows = ((last.year - first.year) * 12 + last.month - first.month) // offset.n + 1
        if expected_rows > max_expansion * len(timestamps):
            raise ValueError(
                f"Automatic frequency {freq!r} would expand a series by more than {max_expansion}x. {error}"
            )
        grid = pd.date_range(first, last, freq=offset)
        if not timestamps.isin(grid).all():
            raise ValueError(error)
    return freq


def validate_frequency_grid(data: pd.DataFrame, freq: str, name: str = "data", max_expansion: int = 10) -> None:
    """Validate automatic resampling; explicit user frequencies bypass this check."""
    error = f"{name} has timestamps incompatible with automatic frequency {freq!r}. Set freq explicitly."
    indexes = [
        pd.DatetimeIndex(series.index.get_level_values("timestamp")).as_unit("ns").sort_values()
        for _, series in data.groupby(level="item_id", sort=False)
    ]
    offset = pd.tseries.frequencies.to_offset(freq)
    # AutoGluon skips resampling when the whole panel already matches, even at noon.
    # Short series also need compatible index metadata for AutoGluon to skip resampling.
    candidate_freq = data.index.levels[data.index.names.index("timestamp")].freq
    if all(
        pd.infer_freq(timestamps) == offset.freqstr
        if len(timestamps) >= 3
        else (
            candidate_freq == offset
            and not timestamps.hasnans
            and timestamps.equals(pd.date_range(timestamps[0], periods=len(timestamps), freq=offset))
        )
        for timestamps in indexes
    ):
        return
    for timestamps in indexes:
        if timestamps.hasnans or timestamps.has_duplicates:
            raise ValueError(error)
        first, last = timestamps[0], timestamps[-1]
        limit = max_expansion * len(timestamps)
        if isinstance(offset, pd.offsets.Tick):
            if (first.value - first.normalize().value) % offset.nanos:
                raise ValueError(error)
            too_large = (int(last.value) - int(first.value)) // offset.nanos + 1 > limit
        else:
            if offset.rollforward(first.normalize()) != first:
                raise ValueError(error)
            try:
                too_large = last > first + offset * (limit - 1)
            except (OverflowError, pd.errors.OutOfBoundsDatetime):
                # The upper bound lies beyond representable dates.
                too_large = False
        if too_large:
            raise ValueError(
                f"Automatic frequency {freq!r} would expand a series by more than {max_expansion}x in {name}."
            )
        if not timestamps.isin(pd.date_range(first, last, freq=offset)).all():
            raise ValueError(error)
