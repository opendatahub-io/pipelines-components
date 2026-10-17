"""Bounded, timestamp-based panel sampling without loading the source into memory."""

import io
import sys


def load_latest_timeseries(
    open_source, prepare_chunk, id_column, timestamp_column, max_size_bytes, chunk_size, report, log, synthetic_id=False
):
    """Keep a fixed number of latest distinct timestamps per series, under a shared byte budget.

    Small sources need one read. Oversized sources are inventoried, then reread using
    an immutable S3 version or an ETag precondition. Conservative per-column bounds
    set fixed row quotas before the second read, so file order cannot affect the sample.
    Short series retain all rows; longer series share a common row quota. Pending
    batches are compacted at twice the quota rather than sorting the sample on every
    chunk. Memory is O(sample budget + one CSV chunk), not O(source size).
    """
    import pandas as pd
    from kfp_components.components.training.automl.shared.parquet_utils import stringify_mixed_object_columns

    frames = []
    raw_bytes = 0
    stats = {}
    object_columns = set()
    oversized = False
    timestamp_kind = None
    metadata_bytes = 0
    floating_ids = False
    report.update(source_rows_read=0, partial_read=False, cap_reached=False, read_passes=1)
    response = open_source({})
    snapshot = {}
    if response.get("VersionId") and response["VersionId"] != "null":
        snapshot["VersionId"] = response["VersionId"]
    elif response.get("ETag"):
        snapshot["IfMatch"] = response["ETag"]

    def inventory(raw, prepared=False):
        nonlocal timestamp_kind, metadata_bytes, floating_ids
        data = raw if prepared else prepare_chunk(raw)
        floating_ids |= pd.api.types.is_float_dtype(data[id_column])
        numeric_time = pd.api.types.is_numeric_dtype(data[timestamp_column])
        if timestamp_kind is not None and timestamp_kind != numeric_time:
            raise ValueError("Timestamp column mixes fractional years and datetimes. Use one timestamp format.")
        timestamp_kind = numeric_time
        for column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]) and column != timestamp_column:
                object_columns.add(column)
        for item_id, group in data.groupby(id_column, sort=False):
            entry = stats.get(item_id)
            if entry is None:
                entry = {"rows": 0, "widths": {}, "objects": {}, "ordered": True, "last": None}
                stats[item_id] = entry
                # Account conservatively for the Python inventory as well as buffers.
                # An unbounded number of IDs cannot be represented within a fixed budget.
                metadata_bytes += 1024 + 256 * len(data.columns) + sys.getsizeof(item_id)
                if metadata_bytes > max_size_bytes // 4:
                    raise ValueError("Too many time series for the sampling memory budget. Use a larger preset.")
            entry["rows"] += len(group)
            times = group[timestamp_column]
            entry["ordered"] &= bool(times.is_monotonic_increasing) and (
                entry["last"] is None or times.iloc[0] >= entry["last"]
            )
            entry["last"] = times.iloc[-1]
            for column in group.columns:
                values = group[column]
                if column == timestamp_column or pd.api.types.is_numeric_dtype(values):
                    width = int(values.dtype.itemsize)
                    entry["widths"][column] = max(entry["widths"].get(column, 0), width)
                    # If another chunk promotes this column to object/string, these
                    # numeric values may also need boxing or stringification. Reserve
                    # 32 ASCII characters for any CSV-inferred 64-bit numeric value;
                    # endpoint lengths alone do not bound intermediate float strings.
                    object_width = sys.getsizeof("0" * 32) + 8
                else:
                    object_width = int(
                        values.map(lambda value: max(sys.getsizeof(value), sys.getsizeof(str(value))) + 8).max()
                    )
                entry["objects"][column] = max(entry["objects"].get(column, 0), object_width, sys.getsizeof(pd.NA) + 8)

    with io.TextIOWrapper(response["Body"], encoding="utf-8") as stream:
        reader = iter(pd.read_csv(stream, chunksize=chunk_size))
        while True:
            try:
                chunk = next(reader)
            except StopIteration:
                break
            except Exception as exc:
                if not report["source_rows_read"]:
                    raise ValueError(f"Error reading CSV from S3: {exc}") from exc
                log.warning("Partial training CSV read after %s rows: %s", report["source_rows_read"], exc)
                report["partial_read"] = True
                break
            report["source_rows_read"] += len(chunk)
            raw_bytes += int(chunk.memory_usage(deep=True).sum())
            if synthetic_id:
                raw_bytes += len(chunk) * (sys.getsizeof("item_0") + 8)
            if not oversized and raw_bytes <= max_size_bytes:
                frames.append(chunk)
                continue
            if not oversized:
                oversized = True
                for frame in frames:
                    inventory(frame)
                frames.clear()
                # Release the last loop reference to a potentially large frame.
                if "frame" in locals():
                    del frame
            inventory(chunk)

    if not report["source_rows_read"]:
        raise ValueError("No data was loaded from S3. The file may be empty or inaccessible.")
    if not oversized:
        result = prepare_chunk(pd.concat(frames, ignore_index=True))
        frames.clear()
        stringify_mixed_object_columns(result)
        # Concatenation can promote previously numeric columns to object dtype.
        if int(result.memory_usage(deep=True).sum()) <= max_size_bytes:
            report["cleansed"] = True
            return result
        inventory(result, prepared=True)
        del result
    del chunk
    if not snapshot:
        raise ValueError("Cannot safely reread a large S3 CSV without an ETag or VersionId.")

    def canonical_id(value):
        if id_column in object_columns:
            return str(value)
        return float(value) if floating_ids else value

    # Match the file-wide ID dtype, rather than chunk-local CSV inference. Numeric
    # IDs remain numeric; mixed object IDs use the existing Parquet string policy.
    combined_stats = {}
    for key, entry in stats.items():
        key = canonical_id(key)
        if key not in combined_stats:
            combined_stats[key] = entry
        else:
            existing = combined_stats[key]
            existing["rows"] += entry["rows"]
            existing["ordered"] = False
            for name in ("widths", "objects"):
                for column, width in entry[name].items():
                    existing[name][column] = max(existing[name].get(column, 0), width)
    stats = combined_stats

    # Reserve inventory space and per-buffer pandas indexes. Actual row bounds
    # include object payloads, dtype promotion, and the per-row integer index.
    metadata_bytes = sum(1024 + 256 * len(entry["objects"]) + sys.getsizeof(key) for key, entry in stats.items())
    available = max_size_bytes - metadata_bytes - 132 * len(stats)
    for entry in stats.values():
        entry["row_bytes"] = 8 + sum(
            entry["objects"][column] if column in object_columns else entry["widths"][column]
            for column in entry["objects"]
        )
    if available < sum(entry["row_bytes"] for entry in stats.values()):
        raise ValueError("Sampling budget cannot retain even one observation per time series. Use a larger preset.")
    low, high = 1, max(entry["rows"] for entry in stats.values())
    while low < high:
        middle = (low + high + 1) // 2
        required = sum(min(entry["rows"], middle) * entry["row_bytes"] for entry in stats.values())
        if required <= available:
            low = middle
        else:
            high = middle - 1
    quotas = {key: min(entry["rows"], low) for key, entry in stats.items()}
    report.update(
        cap_reached=True, read_passes=2, cleansed=True, unsorted_input=any(not e["ordered"] for e in stats.values())
    )

    def compact(parts, key):
        merged = pd.concat(parts, ignore_index=True)
        # Drop old buffers before sorting/deduplicating the merged frame. This
        # matters for quality's 10 GiB sample under the loader's 64 GiB pod limit.
        parts.clear()
        if not stats[key]["ordered"]:
            merged = merged.sort_values(timestamp_column, kind="stable")
        retained = merged.drop_duplicates(subset=[timestamp_column], keep="last").tail(quotas[key]).copy()
        retained.index = pd.RangeIndex(len(retained))
        return retained

    buffers = {}
    pending_rows = {}
    rows_read = 0
    try:
        response = open_source(snapshot)
    except Exception as exc:
        raise ValueError(f"Could not reread the same S3 CSV snapshot: {exc}") from exc
    with io.TextIOWrapper(response["Body"], encoding="utf-8") as stream:
        if ("VersionId" in snapshot and response.get("VersionId") != snapshot["VersionId"]) or (
            "IfMatch" in snapshot and response.get("ETag") != snapshot["IfMatch"]
        ):
            raise ValueError("S3 CSV changed between sampling reads (version or ETag mismatch).")
        try:
            for raw in pd.read_csv(stream, chunksize=chunk_size):
                if report["partial_read"]:
                    raw = raw.head(report["source_rows_read"] - rows_read)
                rows_read += len(raw)
                data = prepare_chunk(raw)
                data[id_column] = data[id_column].map(canonical_id)
                for key, group in data.groupby(id_column, sort=False):
                    # Bound a wide chunk before adding it to retained buffers.
                    if len(group) > quotas[key]:
                        group = compact([group], key)
                    else:
                        # GroupBy slices can share arrays with the whole CSV chunk.
                        # Detach both data and index before retaining a small group.
                        group = group.copy()
                        group.index = pd.RangeIndex(len(group))
                    buffers.setdefault(key, []).append(group)
                    pending_rows[key] = pending_rows.get(key, 0) + len(group)
                    if pending_rows[key] >= 2 * quotas[key]:
                        buffers[key] = [compact(buffers[key], key)]
                        pending_rows[key] = len(buffers[key][0])
                if report["partial_read"] and rows_read == report["source_rows_read"]:
                    break
        except Exception as exc:
            raise ValueError(f"Error rereading CSV from S3 for latest-per-series sampling: {exc}") from exc
    if rows_read != report["source_rows_read"]:
        raise ValueError("S3 CSV changed between sampling reads (row count mismatch).")
    # Consume buffers as they are finalized, avoiding an extra retained panel copy.
    selected = []
    for key in stats:
        selected.append(compact(buffers.pop(key), key))
    result = pd.concat(selected, ignore_index=True)
    selected.clear()
    stringify_mixed_object_columns(result)
    if int(result.memory_usage(deep=True).sum()) > max_size_bytes:
        raise ValueError("Retained time series exceed the sampling memory budget after dtype normalization.")
    return result
