"""Minimal mocked pandas implementation for unit tests without the real pandas dependency.

Provides MockedDataFrame and a mocked pandas module so the tabular_data_loader component
can run with sys.modules['pandas'] patched. Used together with _mock_boto3_and_pandas() in tests.
"""

import csv
import io
import json
import math
import random
from collections import Counter


class MockedDataFrame:
    """Minimal DataFrame-like object: columns (list) and rows (list of lists, one per row)."""

    BYTES_PER_ROW = 100  # Used for memory_usage(deep=True).sum()

    def __init__(self, columns, rows):
        """Store column names and row data."""
        self._columns = list(columns)
        self._rows = list(rows)

    @property
    def columns(self):
        """Return the list of column names."""
        return self._columns

    @property
    def empty(self):
        """Return True if there are no rows."""
        return len(self._rows) == 0

    def __len__(self):
        """Return the number of rows."""
        return len(self._rows)

    def memory_usage(self, deep=True):
        """Return a mock object whose sum() is BYTES_PER_ROW times row count."""

        class MemUsage:
            """Mock memory usage object with sum() returning byte estimate."""

            def __init__(self, df):
                """Store reference to the dataframe."""
                self._df = df

            def sum(self):
                """Return BYTES_PER_ROW times number of rows."""
                return len(self._df._rows) * MockedDataFrame.BYTES_PER_ROW

        return MemUsage(self)

    def head(self, n):
        """Return a new MockedDataFrame with the first n rows."""
        return MockedDataFrame(self._columns, self._rows[:n])

    def drop(self, columns=None, inplace=False):
        """Drop columns from the dataframe."""
        if columns is None:
            return None if inplace else self
        col_indices = [i for i, c in enumerate(self._columns) if c not in columns]
        new_columns = [self._columns[i] for i in col_indices]
        new_rows = [[row[i] for i in col_indices] for row in self._rows]
        if inplace:
            self._columns = new_columns
            self._rows = new_rows
            return None
        return MockedDataFrame(new_columns, new_rows)

    def dropna(self, subset=None):
        """Drop rows with missing values in the given columns."""
        if not subset:
            return self
        col_indices = [self._columns.index(c) for c in subset]

        def _cell_missing(val) -> bool:
            if val is None or val == "":
                return True
            try:
                return math.isnan(float(val))
            except (TypeError, ValueError):
                return False

        new_rows = [row for row in self._rows if all(not _cell_missing(row[i]) for i in col_indices)]
        return MockedDataFrame(self._columns, new_rows)

    def replace(self, to_replace, value, inplace=False):
        """Minimal ``DataFrame.replace``: map ±infinity to NaN (float), matching production pandas."""
        inf_like = False
        if isinstance(to_replace, (list, tuple)):
            for x in to_replace:
                if isinstance(x, float) and math.isinf(x):
                    inf_like = True
                    break
        if not inf_like:
            out = MockedDataFrame(self._columns, [list(r) for r in self._rows])
            return None if inplace else out

        def _map_cell(v):
            try:
                fv = float(v)
                if math.isinf(fv):
                    return float("nan")
            except (TypeError, ValueError):
                pass
            return v

        new_rows = [[_map_cell(c) for c in row] for row in self._rows]
        out = MockedDataFrame(self._columns, new_rows)
        if inplace:
            self._columns = out._columns
            self._rows = out._rows
            return None
        return out

    def drop_duplicates(self, inplace=False):
        """Drop full-row duplicates (first occurrence kept).

        NaN in any cell is treated like pandas duplicate detection (two NaNs in the
        same column positions count as equal), not Python ``tuple`` equality.
        """

        def _dedup_key_part(cell):
            if isinstance(cell, float):
                if math.isnan(cell):
                    return "__PANDAS_NAN__"
                return ("float", cell)
            try:
                fv = float(cell)
                if math.isnan(fv):
                    return "__PANDAS_NAN__"
            except (TypeError, ValueError):
                pass
            return cell

        seen: set[tuple] = set()
        new_rows = []
        for row in self._rows:
            key = tuple(_dedup_key_part(c) for c in row)
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(list(row))
        out = MockedDataFrame(self._columns, new_rows)
        if inplace:
            self._columns = out._columns
            self._rows = out._rows
            return None
        return out

    def _col_index(self, col):
        """Return the index of the given column name."""
        return self._columns.index(col)

    def _value_counts_for_column(self, col):
        """Return MockedValueCounts for the given column."""
        idx = self._col_index(col)
        counts = Counter(row[idx] for row in self._rows)
        return MockedValueCounts(counts)

    def __getitem__(self, key):
        """Return MockedSeries (str), column subset (list of names), filtered rows (mask), or self."""
        if isinstance(key, list):
            # pandas df[['col1', 'col2']] returns only those columns; match it so tests catch subsetting bugs
            if all(isinstance(k, str) for k in key):
                col_indices = [self._columns.index(k) for k in key]
                new_columns = [self._columns[i] for i in col_indices]
                new_rows = [[row[i] for i in col_indices] for row in self._rows]
                return MockedDataFrame(new_columns, new_rows)
            return self
        if isinstance(key, tuple):
            return self
        # Boolean "mask" style: df[df[col] != val]
        if hasattr(key, "_column") and hasattr(key, "_value"):
            col_idx = self._col_index(key._column)
            val = key._value
            return MockedDataFrame(
                self._columns,
                [row for row in self._rows if row[col_idx] != val],
            )
        return self

    def __ne__(self, other):
        """Return a mask object for filtering rows by column != value."""

        class Mask:
            _column = None
            _value = other

        Mask._column = getattr(self, "_last_column", None)
        return Mask

    def value_counts(self):
        """Return value counts for the column (used via MockedSeries)."""
        col = getattr(self, "_value_counts_column", None)
        if col is not None:
            return self._value_counts_for_column(col)
        return MockedValueCounts({})

    def groupby(self, by, group_keys=False):
        """Return a MockedGroupBy for stratified sampling."""
        return MockedGroupBy(self, by)

    def sample(self, frac=1.0, random_state=None):
        """Return a new MockedDataFrame with a random sample of rows."""
        rng = random.Random(random_state)
        n = max(1, int(len(self._rows) * frac)) if frac < 1.0 else len(self._rows)
        n = min(n, len(self._rows))
        indices = rng.sample(range(len(self._rows)), n)
        new_rows = [self._rows[i] for i in indices]
        return MockedDataFrame(self._columns, new_rows)

    def reset_index(self, drop=True):
        """Return self (no-op for mock)."""
        return self

    def to_csv(self, path, index=False):
        """Write the data to a CSV file at the given path."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._columns)
            writer.writerows(self._rows)

    def to_json(self, orient="records"):
        """Serialize the dataframe to a JSON string."""
        if orient == "records":
            records = []
            for row in self._rows:
                record = {}
                for col, val in zip(self._columns, row):
                    record[col] = val
                records.append(record)
            return json.dumps(records)
        raise NotImplementedError(f"to_json orient={orient!r} not supported in mock")


class MockedColumn(MockedDataFrame):
    """Single-column dataframe that also supports value_counts and != comparisons (like pandas Series)."""

    def __init__(self, parent_df, column_name):
        """Extract a single column from parent_df as a one-column MockedDataFrame."""
        col_idx = parent_df._columns.index(column_name)
        super().__init__([column_name], [[row[col_idx]] for row in parent_df._rows])
        self._parent = parent_df
        self._column_name = column_name

    def value_counts(self):
        """Return value counts for this column."""
        return self._parent._value_counts_for_column(self._column_name)

    def __ne__(self, other):
        """Return a mask for filtering rows where this column != other."""

        class Mask:
            _column = self._column_name
            _value = other
            _mask_series = True

        return Mask


class MockedValueCounts:
    """Minimal value_counts() result: supports .index.values and comparison for singleton detection."""

    def __init__(self, count_dict):
        """Store a mapping of value -> count."""
        self._counts = dict(count_dict)

    @property
    def index(self):
        """Return an object with .values listing the distinct values."""

        class Index:
            def __init__(self, keys):
                """Store index keys"""
                self._keys = keys

            @property
            def values(self):
                return self._keys

        return Index(list(self._counts.keys()))

    def __eq__(self, other):
        """Return an object whose .index.values are keys with count equal to other."""
        matching_keys = [k for k, v in self._counts.items() if v == other]

        class FilteredResult:
            @property
            def index(self):
                class Idx:
                    values = matching_keys

                return Idx()

        return FilteredResult()


class MockedGroupBy:
    """Minimal groupby().apply(fn).reset_index(drop=True) for stratified sampling."""

    def __init__(self, df, by):
        """Store the dataframe and the column to group by."""
        self._df = df
        self._by = by

    def apply(self, fn, **kwargs):
        """Group rows by column, apply fn to each group, and concatenate results."""
        col_idx = self._df._col_index(self._by)
        groups = {}
        for row in self._df._rows:
            key = row[col_idx]
            groups.setdefault(key, []).append(row)
        result_rows = []
        for key in sorted(groups.keys()):
            group_df = MockedDataFrame(self._df._columns, groups[key])
            sampled = fn(group_df)
            if hasattr(sampled, "_rows"):
                result_rows.extend(sampled._rows)
            else:
                result_rows.extend(sampled)
        return MockedDataFrame(self._df._columns, result_rows)

    def reset_index(self, drop=True):
        """Return stored result or self (no-op for mock)."""
        return self._result if hasattr(self, "_result") else self


def _read_csv_chunks(text_stream, chunksize):
    """Parse CSV from text_stream and yield MockedDataFrame chunks."""
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
    for start in range(0, len(rows), chunksize):
        chunk_rows = rows[start : start + chunksize]
        yield MockedDataFrame(header, chunk_rows)


def _concat(dfs, ignore_index=True, axis=0):
    """Concatenate a list of MockedDataFrames."""
    if not dfs:
        return MockedDataFrame([], [])
    if axis == 1:
        # Horizontal concat: merge columns side by side
        all_columns = []
        for df in dfs:
            all_columns.extend(df._columns)
        n_rows = max(len(df._rows) for df in dfs) if dfs else 0
        all_rows = []
        for i in range(n_rows):
            row = []
            for df in dfs:
                if i < len(df._rows):
                    row.extend(df._rows[i])
                else:
                    row.extend([""] * len(df._columns))
            all_rows.append(row)
        return MockedDataFrame(all_columns, all_rows)
    # axis=0: vertical concat (existing logic)
    columns = dfs[0]._columns
    rows = []
    for df in dfs:
        rows.extend(df._rows)
    return MockedDataFrame(columns, rows)


def _mock_train_test_split(*args, test_size=0.25, stratify=None, random_state=None):
    """Simple deterministic split for testing: first (1-test_size) rows for train, rest for test."""
    X, y = args[0], args[1]
    n = len(X)
    split_idx = max(1, int(n * (1 - test_size)))

    X_train = MockedDataFrame(X._columns, X._rows[:split_idx])
    X_test = MockedDataFrame(X._columns, X._rows[split_idx:])
    y_train = MockedDataFrame(y._columns, y._rows[:split_idx])
    y_test = MockedDataFrame(y._columns, y._rows[split_idx:])

    return X_train, X_test, y_train, y_test


def make_mocked_sklearn_module():
    """Build mock sklearn and sklearn.model_selection modules for sys.modules patching."""
    import types

    mock_sklearn = types.ModuleType("sklearn")
    mock_model_selection = types.ModuleType("sklearn.model_selection")
    mock_model_selection.train_test_split = _mock_train_test_split
    mock_sklearn.model_selection = mock_model_selection

    return mock_sklearn, mock_model_selection


def make_mocked_pandas_module():
    """Build a module-like object that can be used as sys.modules['pandas']."""
    import types

    mod = types.ModuleType("pandas")

    def _read_csv(stream, chunksize=None):
        if chunksize is not None:
            return _read_csv_chunks(stream, chunksize)
        chunks = list(_read_csv_chunks(stream, 10000))
        return _concat(chunks) if chunks else MockedDataFrame([], [])

    mod.read_csv = _read_csv
    mod.concat = _concat
    mod.DataFrame = lambda: MockedDataFrame([], [])

    _original_getitem = MockedDataFrame.__getitem__

    def getitem(self, key):
        """Return MockedColumn for column name, or filter by mask."""
        if isinstance(key, str) and key in self._columns:
            return MockedColumn(self, key)
        return _original_getitem(self, key)

    MockedDataFrame.__getitem__ = getitem

    return mod
