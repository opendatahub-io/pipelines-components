"""Budgeted, order-independent per-series tails with a disposable Parquet/SQLite spool.

Only normalized chunks enter this module. SQLite indexes keys and row locations, not
the payload. Its primary key also implements last-occurrence-wins deduplication.
The source is consumed once, including when the sample budget is exhausted.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .parquet_utils import stringify_mixed_object_columns

logger = logging.getLogger(__name__)

PRESET_MAX_SIZE_BYTES = {"speed": 100 * 1024**2, "balanced": 1024**3, "quality": 10 * 1024**3}
PRESET_TEST_DATA_MAX_SIZE_BYTES = {"speed": 50 * 1024**2, "balanced": 100 * 1024**2, "quality": 1024**3}
PANDAS_CHUNK_SIZE = 10_000
MAX_STAGING_BYTES = 16 * 1024**3
DISK_RESERVE_BYTES = 256 * 1024**2
_INDEX_BYTES = pd.RangeIndex(0).memory_usage()
_LOCATION_DTYPE = np.dtype([("part", "i8"), ("position", "i8"), ("cost", "i8")])


def minimum_retained_rows(prediction_length: int, selection_fraction: float, external_test: bool) -> int:
    """Protect selection training plus one internal validation window and evaluation context.

    AutoGluon can reduce additional validation windows when history is short. Existing
    source series shorter than this minimum are preserved in full, never shortened.
    """
    selection_minimum = prediction_length + max(prediction_length + 1, 5)

    def sufficient(n):
        train = n if external_test else max(1, min(int(n * 0.8), n - 1))
        selection = max(1, min(int(train * selection_fraction), train - 1))
        return selection >= selection_minimum and (external_test or n - train > prediction_length)

    high = 2
    while not sufficient(high):
        high *= 2
    low = 1
    while low < high:
        mid = (low + high) // 2
        if sufficient(mid):
            high = mid
        else:
            low = mid + 1
    return low


def _row_costs(frame: pd.DataFrame) -> np.ndarray:
    """Attribute pandas deep bytes to rows, excluding the single output RangeIndex."""
    costs = np.zeros(len(frame), dtype=np.int64)
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object:
            costs += np.fromiter(map(sys.getsizeof, series), dtype=np.int64, count=len(frame)) + 8
        elif isinstance(series.dtype, np.dtype):
            costs += series.dtype.itemsize
        else:
            # Nullable/extension arrays can own shared buffers. Round up rather than
            # mistaking their Arrow/serialized size for the downstream pandas footprint.
            costs += int(np.ceil(series.memory_usage(index=False, deep=True) / max(1, len(frame))))
    return costs


def _time_keys(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.to_numpy(dtype="datetime64[ns]").view("i8")
    return series.to_numpy(dtype="float64")


def _timestamp(value, numeric: bool):
    return float(value) if numeric else pd.Timestamp(int(value))


def add_split_profile(profile_path: Path, splits: dict, id_column: str, timestamp_column: str) -> None:
    """Extend a JSONL source profile with split ranges without building a giant JSON object."""
    summaries = {}
    for name, frame in splits.items():
        summary = frame.groupby(id_column)[timestamp_column].agg(["min", "max", "size"])
        summary.index = summary.index.map(str)
        summaries[name] = summary.to_dict(orient="index")
    temporary = profile_path.with_suffix(".tmp")
    with profile_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
        for line in source:
            record = json.loads(line)
            record["splits"] = {}
            for name, summary in summaries.items():
                row = summary.get(record["series_id"])
                if row is not None:
                    record["splits"][name] = {
                        "rows": int(row["size"]),
                        "min_timestamp": str(row["min"]),
                        "max_timestamp": str(row["max"]),
                    }
                else:
                    record["splits"][name] = {"rows": 0, "min_timestamp": None, "max_timestamp": None}
            output.write(json.dumps(record) + "\n")
    temporary.replace(profile_path)


class _Spool:
    """A bounded-cache index into individually typed Parquet chunks."""

    def __init__(self, directory: Path, max_size_bytes: int):
        self.directory = directory
        self.max_size_bytes = max_size_bytes
        self.parquet_bytes = 0
        self.peak_bytes = 0
        self.parts = 0
        self.connection = sqlite3.connect(directory / "index.sqlite")
        # This database is disposable scratch data, never a user database. Disabling
        # the journal avoids duplicating the index on every chunk transaction.
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE observations (
                series TEXT NOT NULL, timestamp NUMERIC NOT NULL,
                part INTEGER NOT NULL, position INTEGER NOT NULL, cost INTEGER NOT NULL,
                PRIMARY KEY (series, timestamp DESC)
            ) WITHOUT ROWID;
            CREATE TABLE sources (
                series TEXT PRIMARY KEY, source_rows INTEGER NOT NULL
            ) WITHOUT ROWID;
            """
        )

    def close(self):
        """Close the index before its temporary directory is removed."""
        self.connection.close()

    def check_disk(self, next_bytes: int = 0):
        """Reserve workspace for outputs and fail explicitly before filling the PVC."""
        used = self.parquet_bytes + (self.directory / "index.sqlite").stat().st_size
        self.peak_bytes = max(self.peak_bytes, used)
        reserve = self.max_size_bytes + DISK_RESERVE_BYTES
        if used + next_bytes > MAX_STAGING_BYTES or shutil.disk_usage(self.directory).free < reserve + next_bytes:
            raise ValueError(
                "Insufficient workspace for latest-per-series sampling: "
                f"scratch uses {used} bytes (limit {MAX_STAGING_BYTES}); "
                f"{reserve} bytes must remain available for outputs. "
                "Reduce or partition the source dataset, or provide more workspace."
            )

    def append(self, frame: pd.DataFrame, id_column: str, timestamp_column: str):
        """Write a chunk and atomically replace locations for duplicate keys."""
        self.check_disk(int(frame.memory_usage(deep=True).sum()))
        path = self.directory / f"{self.parts}.parquet"
        frame.to_parquet(path, index=False, compression="snappy")
        self.parquet_bytes += path.stat().st_size
        costs = _row_costs(frame)
        self.connection.executemany(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(series, timestamp) DO UPDATE SET "
            "part=excluded.part, position=excluded.position, cost=excluded.cost",
            zip(
                frame[id_column].astype(str),
                _time_keys(frame[timestamp_column]).tolist(),
                [self.parts] * len(frame),
                range(len(frame)),
                costs.tolist(),
            ),
        )
        counts = frame.groupby(id_column, sort=False).size()
        self.connection.executemany(
            "INSERT INTO sources VALUES (?, ?) ON CONFLICT(series) DO UPDATE SET "
            "source_rows=source_rows + excluded.source_rows",
            ((str(series), int(count)) for series, count in counts.items()),
        )
        self.connection.commit()
        self.parts += 1
        self.check_disk()

    def plan(self, minimum_rows: int):
        """Find exact byte costs of equal-depth tails without loading their payload.

        Per-series minimum row costs give an upper bound on the feasible depth.
        Only that bounded prefix of each timestamp index is inspected. Candidate
        arrays hold integer locations/costs, never complete source rows.
        """
        self.connection.executescript(
            "CREATE TABLE counts (series TEXT PRIMARY KEY, n INTEGER, min_cost INTEGER, "
            "first_timestamp NUMERIC, last_timestamp NUMERIC) WITHOUT ROWID; "
            "INSERT INTO counts SELECT series, COUNT(*), MIN(cost), MIN(timestamp), MAX(timestamp) "
            "FROM observations GROUP BY series;"
        )
        self.check_disk()
        count, max_depth = self.connection.execute("SELECT COUNT(*), MAX(n) FROM counts").fetchone()
        budget = self.max_size_bytes - _INDEX_BYTES
        low, high = 0, max_depth
        while low < high:
            middle = (low + high + 1) // 2
            lower_cost = self.connection.execute("SELECT SUM(MIN(n, ?) * min_cost) FROM counts", (middle,)).fetchone()[
                0
            ]
            if lower_cost <= budget:
                low = middle
            else:
                high = middle - 1
        upper_depth = low
        if upper_depth == 0:
            raise ValueError("Sampling budget cannot preserve even one latest observation for every series.")

        layers = np.zeros(upper_depth, dtype=np.int64)
        candidates = []
        for series, n in self.connection.execute("SELECT series, n FROM counts ORDER BY series"):
            size = min(n, upper_depth)
            cursor = self.connection.execute(
                "SELECT part, position, cost FROM observations WHERE series=? ORDER BY timestamp DESC LIMIT ?",
                (series, size),
            )
            locations = np.fromiter(cursor, dtype=_LOCATION_DTYPE, count=size)
            layers[:size] += locations["cost"]
            candidates.append((series, n, locations))
        cumulative = layers.cumsum()
        depth = int(np.searchsorted(cumulative, budget, side="right"))
        required_depth = min(minimum_rows, max_depth)
        if depth < required_depth:
            required_bytes = int(cumulative[required_depth - 1]) if required_depth <= upper_depth else None
            raise ValueError(
                "Sampling budget cannot preserve sufficient history for every series: "
                f"{count} series need up to {minimum_rows} latest observations each "
                f"for the forecast horizon and temporal splits; budget={self.max_size_bytes} bytes, "
                f"required_bytes={required_bytes or 'more than the budget'}. "
                "Choose a larger preset or explicitly reduce the number of series."
            )
        remaining = budget - (int(cumulative[depth - 1]) if depth else 0)
        selected = []
        for _, _, locations in candidates:
            take = min(depth, len(locations))
            if take < len(locations) and locations["cost"][take] <= remaining:
                remaining -= int(locations["cost"][take])
                take += 1
            selected.append(locations[:take].copy())
        return np.concatenate(selected)

    def read(self, locations: np.ndarray) -> pd.DataFrame:
        """Decode only selected Parquet parts; each part is read at most once."""
        locations.sort(order=["part", "position"])
        boundaries = np.r_[0, np.flatnonzero(np.diff(locations["part"])) + 1, len(locations)]
        parts = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            selection = locations[start:end]
            frame = pd.read_parquet(self.directory / f"{selection['part'][0]}.parquet")
            parts.append(frame.iloc[selection["position"]].copy())
        return pd.concat(parts, ignore_index=True)


def _trim_to_budget(frame, id_column, timestamp_column, budget, minimum_rows):
    """Reconcile dtype promotion after concat without sacrificing any series' latest row."""
    frame = frame.sort_values([id_column, timestamp_column]).reset_index(drop=True)
    while int(frame.memory_usage(deep=True).sum()) > budget:
        sizes = frame.groupby(id_column, sort=False).size()
        depth = int(sizes.max())
        if depth <= minimum_rows:
            raise ValueError("Sampling budget cannot preserve sufficient history after pandas dtype normalization.")
        # Remove oldest rows only from the longest tails. Replan the depth in one
        # step when dtype promotion changes the footprint substantially.
        ratio = budget / int(frame.memory_usage(deep=True).sum())
        next_depth = max(minimum_rows, min(depth - 1, int(depth * ratio)))
        frame = frame.groupby(id_column, sort=False).tail(next_depth).reset_index(drop=True)
    return frame


def sample_latest_timeseries(
    chunks,
    *,
    id_column: str,
    timestamp_column: str,
    max_size_bytes: int,
    workspace_path: Path,
    profile_path: Path,
    minimum_rows_per_series: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Consume all normalized CSV chunks, retaining balanced, byte-bounded per-ID tails.

    Small sources stay in RAM. Once accumulated pandas bytes exceed the preset
    budget, previously buffered and subsequent chunks spill into a disposable
    Parquet/SQLite spool. Input ordering never affects timestamp selection.
    """
    buffered = []
    buffered_bytes = 0
    scanned_rows = 0
    spool = None
    timestamp_kind = None
    started_at = time.perf_counter()
    with ExitStack() as stack:
        for chunk in chunks:
            if not len(chunk):
                continue
            numeric = not pd.api.types.is_datetime64_any_dtype(chunk[timestamp_column])
            if timestamp_kind is not None and timestamp_kind != numeric:
                raise ValueError("Timestamp representation changes between CSV chunks; use one consistent format.")
            timestamp_kind = numeric
            scanned_rows += len(chunk)
            chunk_bytes = int(chunk.memory_usage(deep=True).sum())
            if spool is None and buffered_bytes + chunk_bytes <= max_size_bytes:
                buffered.append(chunk)
                buffered_bytes += chunk_bytes
                continue
            if spool is None:
                workspace_path.mkdir(parents=True, exist_ok=True)
                directory = Path(
                    stack.enter_context(TemporaryDirectory(prefix="timeseries-sampling-", dir=workspace_path))
                )
                spool = _Spool(directory, max_size_bytes)
                stack.callback(spool.close)
                for previous in buffered:
                    spool.append(previous, id_column, timestamp_column)
                buffered.clear()
                logger.info(
                    "Timeseries source exceeds sample budget; continuing the full scan using workspace storage."
                )
            spool.append(chunk, id_column, timestamp_column)

        scan_seconds = time.perf_counter() - started_at
        if scanned_rows == 0:
            return pd.DataFrame(), {"truncated": False, "cap_reached": False, "scanned_rows": 0}
        if spool is None:
            source = pd.concat(buffered, ignore_index=True)
            source_counts = source.groupby(id_column, sort=False).size().to_dict()
            frame = source.drop_duplicates([id_column, timestamp_column], keep="last")
            unique_rows = len(frame)
            source_ranges = (
                frame.groupby(id_column)[timestamp_column].agg(["min", "max", "size"]).to_dict(orient="index")
            )
            source_stats = (
                (str(series), int(source_counts[series]), int(row["size"]), row["min"], row["max"])
                for series, row in source_ranges.items()
            )
        else:
            locations = spool.plan(minimum_rows_per_series)
            frame = spool.read(locations)
            del locations
            unique_rows = spool.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            source_stats = (
                (series, rows, n, _timestamp(first, timestamp_kind), _timestamp(last, timestamp_kind))
                for series, rows, n, first, last in spool.connection.execute(
                    "SELECT c.series, s.source_rows, c.n, c.first_timestamp, c.last_timestamp "
                    "FROM counts c JOIN sources s ON s.series=c.series ORDER BY c.series"
                )
            )

        stringify_mixed_object_columns(frame)
        frame = _trim_to_budget(frame, id_column, timestamp_column, max_size_bytes, minimum_rows_per_series)
        retained = frame.groupby(id_column)[timestamp_column].agg(["min", "max", "size"])
        retained.index = retained.index.map(str)
        retained = retained.to_dict(orient="index")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", encoding="utf-8") as profile:
            for series, source_rows, n_unique, first, last in source_stats:
                row = retained[series]
                profile.write(
                    json.dumps(
                        {
                            "series_id": series,
                            "source_rows": source_rows,
                            "source_unique_rows": n_unique,
                            "source_min_timestamp": str(first),
                            "source_max_timestamp": str(last),
                            "retained_rows": int(row["size"]),
                            "retained_min_timestamp": str(row["min"]),
                            "retained_max_timestamp": str(row["max"]),
                        }
                    )
                    + "\n"
                )
        truncated = len(frame) < unique_rows
        report = {
            "sampling_method": "latest_per_series",
            "scanned_rows": scanned_rows,
            "source_unique_rows": unique_rows,
            "duplicate_rows": scanned_rows - unique_rows,
            "total_rows_loaded": len(frame),
            "sampled_rows": len(frame),
            "series_count": len(retained),
            "sampled_in_memory_bytes": int(frame.memory_usage(deep=True).sum()),
            "sample_cap_bytes": max_size_bytes,
            "cap_reached": truncated,
            "truncated": truncated,
            "full_source_read": True,
            "used_disk": spool is not None,
            "peak_staging_bytes": spool.peak_bytes if spool else 0,
            "minimum_rows_per_series": minimum_rows_per_series,
            "series_profile": profile_path.name,
            "scan_seconds": scan_seconds,
            "sampling_seconds": time.perf_counter() - started_at - scan_seconds,
        }
        return frame, report
