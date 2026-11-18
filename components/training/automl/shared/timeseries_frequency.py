"""Conservative frequency inference for panels with missing observations."""

from __future__ import annotations

import pandas as pd


def _frequency_grid_size(first, last, offset):
    """Estimate resampling rows without allocating a grid, conservatively for calendar offsets."""
    if isinstance(offset, pd.offsets.Tick):
        origin = int(first.normalize().value)
        return (int(last.value) - origin) // offset.nanos - (int(first.value) - origin) // offset.nanos + 1

    def reaches_last(steps):
        try:
            return first + offset * steps >= last
        except (OverflowError, pd.errors.OutOfBoundsDatetime):
            return True

    lower, upper = 0, 1
    while not reaches_last(upper):
        lower, upper = upper, upper * 2
    while lower < upper:
        middle = (lower + upper) // 2
        if reaches_last(middle):
            upper = middle
        else:
            lower = middle + 1
    return lower + 1


def validate_frequency_expansion(data: pd.DataFrame, freq: str, name: str = "data", max_expansion: int = 1000) -> None:
    """Bound explicit-frequency resampling without requiring grid or midnight alignment."""
    offset = pd.tseries.frequencies.to_offset(freq)
    for item_id, series in data.groupby(level="item_id", sort=False):
        timestamps = pd.DatetimeIndex(series.index.get_level_values("timestamp")).as_unit("ns")
        if timestamps.hasnans:
            raise ValueError(f"Series {item_id!r} in {name} has missing timestamps.")
        expected_rows = _frequency_grid_size(timestamps.min(), timestamps.max(), offset)
        expansion = expected_rows / len(timestamps)
        if expansion > max_expansion:
            raise ValueError(
                f"Frequency {freq!r} would expand series {item_id!r} in {name} by approximately {expansion:.6g}x "
                f"({expected_rows} rows from {len(timestamps)}), exceeding the {max_expansion}x limit."
            )


def infer_frequency_from_gaps(data: pd.DataFrame, max_expansion: int = 10) -> str:
    """Infer a shared observation grid without moving or aggregating timestamps.

    Each series with at least three observations must support the same step with
    a strict majority of its gaps. Remaining gaps must be whole multiples of it.
    Month-start/end data is measured in calendar months, other data in elapsed
    time. Shorter series are validated against the candidate without voting.
    Timestamps must be timezone-naive, as required by AutoGluon.

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
        if timestamps.tz is not None:
            raise ValueError("AutoGluon requires timezone-naive timestamps.")
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
    """Validate automatic resampling of timezone-naive data; explicit frequencies bypass this check."""
    error = f"{name} has timestamps incompatible with automatic frequency {freq!r}. Set freq explicitly."
    indexes = [
        pd.DatetimeIndex(series.index.get_level_values("timestamp")).as_unit("ns").sort_values()
        for _, series in data.groupby(level="item_id", sort=False)
    ]
    if any(timestamps.tz is not None for timestamps in indexes):
        raise ValueError(f"{name}: AutoGluon requires timezone-naive timestamps.")
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
        else:
            if offset.rollforward(first.normalize()) != first:
                raise ValueError(error)
        if _frequency_grid_size(first, last, offset) > limit:
            raise ValueError(
                f"Automatic frequency {freq!r} would expand a series by more than {max_expansion}x in {name}."
            )
        if not timestamps.isin(pd.date_range(first, last, freq=offset)).all():
            raise ValueError(error)
