"""Shared Parquet-write helpers for the AutoML data loaders.

Both AutoML data loaders (tabular and timeseries) stream their source CSV via
``pd.read_csv(..., chunksize=...)`` and concatenate the chunks with ``pd.concat``.
When two chunks disagree on a column's inferred dtype (e.g. one chunk all-numeric,
another with a stray non-numeric value), the concatenated column stays
``object``-dtype with genuinely mixed Python types (e.g. both ``int`` and ``str``).
``to_csv`` stringifies everything silently, but pyarrow's ``to_parquet`` raises
``ArrowInvalid``/``ArrowTypeError`` on a mixed-type object column, so both loaders
normalize before every Parquet write. Lives here instead of being duplicated (see
``user_test_data.py`` for the same rationale).
"""

from __future__ import annotations

import pandas as pd

# infer_dtype() outcomes that indicate genuinely mixed Python types within an
# object column, as opposed to a column that is merely object-dtype because it
# holds strings (optionally with missing values), which pyarrow writes fine.
_MIXED_DTYPE_KINDS = frozenset({"mixed", "mixed-integer", "mixed-integer-float"})


def stringify_mixed_object_columns(df: pd.DataFrame) -> None:
    """Cast genuinely mixed-type ``object`` columns to pandas' nullable string dtype, in place.

    ``df[column].astype("string")`` inspects and converts every element into a new
    array, which is costly on a large dataframe. ``pd.api.types.infer_dtype(...,
    skipna=True)`` classifies a column without building a replacement array, so it
    is used first to skip that cost for the common case (homogeneous object columns,
    e.g. plain strings with missing values); only columns it reports as mixed pay
    for the full conversion. ``astype("string")`` preserves missing values as
    ``pd.NA`` instead of the literal string ``"nan"``.
    """
    for column in df.select_dtypes(include="object").columns:
        if pd.api.types.infer_dtype(df[column], skipna=True) in _MIXED_DTYPE_KINDS:
            df[column] = df[column].astype("string")
