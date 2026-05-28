"""Minimal mocked pandas for timeseries_data_loader unit tests (no real pandas).

Mirrors the tabular_data_loader tests approach: patch ``sys.modules['pandas']`` with
``read_csv`` (chunked iterator), ``concat``, and a small DataFrame-like type that supports
the operations used in ``component.load_timeseries_data_truncate`` and the split logic.
"""

import csv
import io
import json
import math
import re
import types

_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")


class MockSeries:
    """Column vector with ``map``, ``isna``, boolean ``~``, and ``sum`` (count of True)."""

    def __init__(self, values):
        """Store cell values as a list."""
        self._values = list(values)

    def map(self, fn):
        """Apply ``fn`` to each value; return a new ``MockSeries``."""
        return MockSeries([fn(v) for v in self._values])

    def isna(self):
        """Return boolean ``MockSeries`` aligned with ``pandas.Series.isna()``."""
        return MockSeries([_cell_is_na(v) for v in self._values])

    def notna(self):
        """Return boolean ``MockSeries`` aligned with ``pandas.Series.notna()``."""
        return ~self.isna()

    def __invert__(self):
        """Boolean negation element-wise (for ``~mask``)."""
        return MockSeries([not bool(x) for x in self._values])

    def sum(self):
        """Count values that are truthy (used for boolean masks)."""
        return sum(1 for x in self._values if x)

    def any(self):
        """Return whether any value is truthy."""
        return any(self._values)

    def all(self):
        """Return whether all values are truthy."""
        return all(self._values)

    def __getitem__(self, key):
        """Boolean masking or positional indexing."""
        if isinstance(key, MockSeries):
            # Boolean mask
            return MockSeries([v for v, include in zip(self._values, key._values) if include])
        else:
            # Positional indexing
            return self._values[key]

    def __len__(self):
        """Return the number of values."""
        return len(self._values)

    def min(self):
        """Return minimum value."""
        non_null = [v for v in self._values if not _cell_is_na(v)]
        return min(non_null) if non_null else None

    def max(self):
        """Return maximum value."""
        non_null = [v for v in self._values if not _cell_is_na(v)]
        return max(non_null) if non_null else None

    def astype(self, dtype):
        """Type conversion."""
        if dtype is int:
            return MockSeries([int(v) if not _cell_is_na(v) else v for v in self._values])
        elif dtype is str:
            return MockSeries([str(v) if not _cell_is_na(v) else v for v in self._values])
        return self

    def round(self):
        """Round numeric values."""
        return MockSeries([round(v) if isinstance(v, (int, float)) and not _cell_is_na(v) else v for v in self._values])


def _parse_csv_cell(cell):
    """Parse special float tokens from CSV text (real pandas infers these)."""
    if not isinstance(cell, str):
        return cell
    sl = cell.strip().lower()
    if sl in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if sl in ("-inf", "-infinity"):
        return float("-inf")
    return cell


def _to_datetime_scalar(val, errors="coerce"):
    """Return a sortable value: ISO date strings, or numeric timestamps (e.g. fractional year)."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return val
    s = val.strip()
    if _DATE_PREFIX.match(s):
        return s[:10] if len(s) >= 10 else s
    # Numeric timestamps (e.g. 1949.083333) — sort chronologically as floats, like many CSV panels.
    try:
        return float(s)
    except ValueError:
        pass
    if errors == "raise":
        raise ValueError(f"cannot parse datetime from {val!r}")
    return None


def _cell_is_na(val):
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


class MockedDataFrame:
    """Columns + rows; supports the subset of pandas API used by timeseries_data_loader."""

    BYTES_PER_ROW = 100  # Same convention as tabular mock for memory_usage().sum()

    def __init__(self, columns, rows):
        """Store column names and row values (list of lists)."""
        self._columns = list(columns)
        self._rows = list(rows)

    @property
    def columns(self):
        """Column names."""
        return self._columns

    def __len__(self):
        """Row count."""
        return len(self._rows)

    def memory_usage(self, deep=True):
        """Return an object whose ``sum()`` estimates bytes (for size truncation loop)."""

        class MemUsage:
            def __init__(self, df):
                self._df = df

            def sum(self):
                return len(self._df._rows) * MockedDataFrame.BYTES_PER_ROW

        return MemUsage(self)

    def head(self, n):
        """First n rows."""
        return MockedDataFrame(self._columns, self._rows[:n])

    def tail(self, n):
        """Last n rows."""
        return MockedDataFrame(self._columns, self._rows[-n:])

    def copy(self, deep=True):
        """Shallow copy of rows."""
        return MockedDataFrame(self._columns, [list(r) for r in self._rows])

    def sort_values(self, by, ascending=True):
        """Sort rows lexicographically by the given column name(s)."""
        _ = ascending
        cols = list(by) if isinstance(by, (list, tuple)) else [by]
        col_indices = [self._columns.index(c) for c in cols]

        def sort_key(row):
            return tuple(row[i] for i in col_indices)

        sorted_rows = sorted(self._rows, key=sort_key)
        return MockedDataFrame(self._columns, sorted_rows)

    def reset_index(self, drop=True):
        """No index column in mock; return self."""
        return self

    @property
    def iloc(self):
        """Integer-location slice (``df.iloc[a:b]`` only)."""
        return _IlocIndexer(self)

    def to_csv(self, path, index=False):
        """Write CSV (index ignored; mock has no row index column)."""
        _ = index
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._columns)
            writer.writerows(self._rows)

    def to_json(self, orient="records"):
        """JSON records like pandas."""
        if orient == "records":
            records = []
            for row in self._rows:
                records.append(dict(zip(self._columns, row)))
            return json.dumps(records)
        raise NotImplementedError(f"to_json orient={orient!r} not supported in mock")

    def groupby(self, by, sort=True):
        """Group rows by one column (``by``); yields ``(key, MockedDataFrame)`` like pandas."""
        return MockedGroupBy(self, by, sort=sort)

    def __getitem__(self, key):
        """Column access (``str``) or boolean row mask (``MockSeries`` of bool)."""
        if isinstance(key, MockSeries):
            mask = key._values
            if len(mask) != len(self._rows):
                raise ValueError("boolean mask length mismatch")
            new_rows = [row for row, m in zip(self._rows, mask) if m]
            return MockedDataFrame(self._columns, new_rows)
        idx = self._columns.index(key)
        return MockSeries([row[idx] for row in self._rows])

    def __setitem__(self, key, value):
        """Assign a column from a list or ``MockSeries`` (same length as frame)."""
        idx = self._columns.index(key)
        if isinstance(value, MockSeries):
            vals = value._values
        else:
            vals = value
        if not isinstance(vals, list):
            vals = [vals] * len(self._rows)
        if len(vals) != len(self._rows):
            raise ValueError("column assign length mismatch")
        for i, row in enumerate(self._rows):
            row[idx] = vals[i]

    def replace(self, to_replace, value):
        """Replace ``±inf`` (or listed values) with ``value`` in every cell."""
        reps = set(to_replace) if isinstance(to_replace, (list, tuple)) else {to_replace}
        new_rows = []
        for row in self._rows:
            new_row = []
            for c in row:
                if c in reps:
                    new_row.append(value)
                else:
                    new_row.append(c)
            new_rows.append(new_row)
        return MockedDataFrame(self._columns, new_rows)

    def dropna(self, subset=None, inplace=False):
        """Drop rows with NA in any of ``subset`` columns."""
        _ = inplace
        cols = list(subset) if subset is not None else list(self._columns)
        idxs = [self._columns.index(c) for c in cols]
        new_rows = []
        for row in self._rows:
            if any(_cell_is_na(row[i]) for i in idxs):
                continue
            new_rows.append(row)
        return MockedDataFrame(self._columns, new_rows)

    def drop_duplicates(self, subset=None, keep="last"):
        """Drop duplicate keys; ``keep='last'`` keeps the last row per key in row order."""
        if subset is None:
            subset = list(self._columns)
        if keep != "last":
            raise NotImplementedError("mock only implements keep='last'")
        idxs = [self._columns.index(c) for c in subset]
        last_idx = {}
        for i, row in enumerate(self._rows):
            key = tuple(row[j] for j in idxs)
            last_idx[key] = i
        keep_order = sorted(last_idx.values())
        new_rows = [self._rows[i] for i in keep_order]
        return MockedDataFrame(self._columns, new_rows)


class MockedGroupBy:
    """Minimal ``DataFrame.groupby`` for a single column."""

    def __init__(self, df, by, sort=True):
        """Store parent frame, column name, and whether to sort group keys."""
        self._df = df
        self._by = by
        self._sort = sort

    def __iter__(self):
        """Yield ``(group_key, group_df)`` in first-seen key order if ``sort=False``."""
        col_idx = self._df._columns.index(self._by)
        groups: dict = {}
        key_order: list = []
        for row in self._df._rows:
            key = row[col_idx]
            if key not in groups:
                key_order.append(key)
                groups[key] = []
            groups[key].append(row)
        keys = sorted(key_order) if self._sort else key_order
        for k in keys:
            yield k, MockedDataFrame(self._df._columns, groups[k])


class _IlocIndexer:
    """Supports only slice indexing along rows (as used by the component)."""

    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockedDataFrame(self._df._columns, self._df._rows[key])
        raise TypeError(f"mock iloc does not support {type(key)!r}")


def _read_csv_chunks(text_stream, chunksize):
    """Parse CSV and yield MockedDataFrame chunks."""
    if hasattr(text_stream, "read"):
        content = text_stream.read()
    else:
        content = text_stream
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    reader = csv.reader(io.StringIO(content))
    header = next(reader, None)
    if not header:
        return
    rows = list(reader)
    if not rows:
        yield MockedDataFrame(header, [])
        return
    for start in range(0, len(rows), chunksize):
        chunk_rows = [[_parse_csv_cell(c) for c in row] for row in rows[start : start + chunksize]]
        yield MockedDataFrame(header, chunk_rows)


def _concat(dfs, ignore_index=True, axis=0):
    """Concatenate MockedDataFrames along rows (axis=0)."""
    _ = ignore_index
    if not dfs:
        return MockedDataFrame([], [])
    columns = dfs[0]._columns
    rows = []
    for df in dfs:
        rows.extend(df._rows)
    return MockedDataFrame(columns, rows)


def make_mocked_pandas_module():
    """Build a module suitable for ``sys.modules['pandas']``."""
    mod = types.ModuleType("pandas")

    def to_datetime(arg, errors="coerce", utc=False):
        """Parse timestamp column like ``pandas.to_datetime`` (subset)."""
        _ = utc
        if isinstance(arg, MockSeries):
            return MockSeries([_to_datetime_scalar(v, errors) for v in arg._values])
        # Handle string datetime (used in fractional year conversion)
        if isinstance(arg, str):
            return _to_datetime_scalar(arg, errors)
        raise TypeError(f"mock to_datetime not implemented for {type(arg)!r}")

    def to_numeric(arg, errors="coerce"):
        """Convert to numeric like ``pandas.to_numeric``."""
        if isinstance(arg, MockSeries):
            result = []
            for v in arg._values:
                if _cell_is_na(v):
                    result.append(None if errors == "coerce" else v)
                elif isinstance(v, (int, float)):
                    result.append(float(v))
                else:
                    try:
                        result.append(float(v))
                    except (ValueError, TypeError):
                        if errors == "coerce":
                            result.append(None)
                        else:
                            raise
            return MockSeries(result)
        raise TypeError(f"mock to_numeric not implemented for {type(arg)!r}")

    def to_timedelta(arg, unit="D"):
        """Convert to timedelta - simplified for day offsets."""
        if isinstance(arg, (int, float)):
            # Just return the numeric value - we'll add it to dates
            return arg
        if isinstance(arg, MockSeries):
            return MockSeries([v if not _cell_is_na(v) else None for v in arg._values])
        raise TypeError(f"mock to_timedelta not implemented for {type(arg)!r}")

    def _read_csv(stream, chunksize=None):
        if chunksize is not None:
            return _read_csv_chunks(stream, chunksize)
        chunks = list(_read_csv_chunks(stream, 10000))
        return _concat(chunks) if chunks else MockedDataFrame([], [])

    def _dataframe(*args, **kwargs):
        """Support ``DataFrame(columns=...)`` for empty frames (concat with no parts)."""
        cols = kwargs.get("columns")
        if cols is not None:
            return MockedDataFrame(list(cols), [])
        return MockedDataFrame([], [])

    mod.read_csv = _read_csv
    mod.concat = _concat
    mod.DataFrame = _dataframe
    mod.to_datetime = to_datetime
    mod.to_numeric = to_numeric
    mod.to_timedelta = to_timedelta
    return mod
