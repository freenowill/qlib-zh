"""CachedAlpha158 — cache Alpha158 _data across walk-forward folds.

Usage:
    from cached_handler import CachedAlpha158, precompute_handler_cache

    # Before fold loop:
    precompute_handler_cache("path/to/cache.parquet", template_cfg, ...)

    # In each fold's YAML, replace handler with:
    #   class: CachedAlpha158
    #   module_path: scripts.small.cached_handler
    #   kwargs:
    #       cache_path: "path/to/cache.parquet"
    #       ... (original Alpha158 kwargs)
"""
from __future__ import annotations

import ast
import os
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.handler_extra import AlphaExtra
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import TimeInspector

_PARQUET_OK: bool = False
try:
    import pyarrow  # noqa: F401

    _PARQUET_OK = True
except ImportError:
    pass


def precompute_handler_cache(
    cache_path: str,
    template_cfg: dict,
    start_time: str,
    end_time: str,
) -> None:
    """Precompute Alpha158 features for the full date range and save to disk.

    Must be called after qlib.init().
    """
    import copy
    from qlib.utils import init_instance_by_config

    handler_cfg = _resolve_handler_config(template_cfg)
    if handler_cfg is None:
        raise RuntimeError("Cannot find handler config in template")
    handler_cfg = copy.deepcopy(handler_cfg)

    handler_cfg["kwargs"]["start_time"] = start_time
    handler_cfg["kwargs"]["end_time"] = end_time
    handler_cfg["kwargs"]["fit_start_time"] = start_time
    handler_cfg["kwargs"]["fit_end_time"] = end_time

    handler = init_instance_by_config(handler_cfg)
    handler.setup_data(DataHandlerLP.IT_FIT_SEQ)

    if handler._data is None or handler._data.empty:
        raise RuntimeError("Handler produced empty _data — cannot cache")

    _dump_data(handler._data, cache_path)
    n_rows = len(handler._data)
    n_dates = handler._data.index.get_level_values("datetime").nunique()
    n_inst = handler._data.index.get_level_values("instrument").nunique()
    print(f"  [Cache] Saved {n_rows} rows ({n_dates} dates × {n_inst} inst) → {cache_path}")
    print(f"  [Cache] Columns: {list(handler._data.columns[:5])}...")


class CachedAlpha158(Alpha158):
    """Alpha158 variant that loads precomputed _data from cache in setup_data().

    On cache hit:  loads _data from disk → runs processor pipeline (fit+transform)
    On cache miss: calls normal setup_data() → saves _data to disk for future folds
    """

    def __init__(self, cache_path: str | None = None, **kwargs):
        self._cache_path = cache_path
        # Remove any forwarded cache keys before passing to parent
        kwargs.pop("cache_path", None)
        super().__init__(**kwargs)

    def setup_data(self, init_type: str = DataHandlerLP.IT_FIT_SEQ, **kwargs):
        cache_file = Path(self._cache_path) if self._cache_path else None

        # ── Cache hit path ──────────────────────────────────
        _cache_was_corrupt = False
        if cache_file is not None and cache_file.exists():
            _data = _load_data(self._cache_path, self.start_time, self.end_time)
            if _data is not None and not _data.empty:
                print(
                    f"  [Cache] Loaded {len(_data)} rows from {cache_file.name} "
                    f"({self.start_time} ~ {self.end_time})"
                )
                self._data = _data
                # Run the processor pipeline on the cached raw data
                with TimeInspector.logt("fit & process data (cached)"):
                    if init_type == DataHandlerLP.IT_FIT_SEQ:
                        self.fit_process_data()
                    elif init_type == DataHandlerLP.IT_FIT_IND:
                        self.fit()
                        self.process_data()
                    elif init_type == DataHandlerLP.IT_LS:
                        self.process_data()
                    else:
                        raise NotImplementedError(f"Unknown init_type: {init_type}")
                return

            # Cache exists but is unreadable — remove it so fold-loop
            # fallback doesn't overwrite the full-range cache with
            # fold-specific data (which corrupted it previously).
            cache_file.unlink(missing_ok=True)
            _cache_was_corrupt = True
            print(f"  [Cache] Removing corrupt cache ({cache_file.name})")

        # ── Cache miss path ─────────────────────────────────
        super().setup_data(init_type=init_type, **kwargs)

        # Save _data to cache ONLY if this is a true first-creation
        # (file didn't exist before).  Never overwrite from fold-loop
        # fallback — that would truncate the full-range cache.
        if (
            cache_file is not None
            and not _cache_was_corrupt
            and self._data is not None
            and not self._data.empty
        ):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            _dump_data(self._data, str(cache_file))
            print(f"  [Cache] Saved {len(self._data)} rows → {cache_file}")


class CachedAlphaExtra(AlphaExtra):
    """AlphaExtra variant that loads precomputed _data from cache in setup_data().

    Same caching logic as CachedAlpha158, but works with AlphaExtra's factor_config
    and precomputed (direct:true) mode.  Each fold loads the full-range parquet cache,
    filters to its date window, then runs the processor pipeline.

    Usage:
        # In each fold's YAML:
        #   class: CachedAlphaExtra
        #   module_path: scripts.small.cached_handler
        #   kwargs:
        #       cache_path: "path/to/cache.parquet"
        #       ... (original AlphaExtra kwargs)
    """

    def __init__(self, cache_path: str | None = None, **kwargs):
        self._cache_path = cache_path
        kwargs.pop("cache_path", None)
        super().__init__(**kwargs)

    def setup_data(self, init_type: str = DataHandlerLP.IT_FIT_SEQ, **kwargs):
        cache_file = Path(self._cache_path) if self._cache_path else None

        _cache_was_corrupt = False
        if cache_file is not None and cache_file.exists():
            _data = _load_data(self._cache_path, self.start_time, self.end_time)
            if _data is not None and not _data.empty:
                print(
                    f"  [Cache] Loaded {len(_data)} rows from {cache_file.name} "
                    f"({self.start_time} ~ {self.end_time})"
                )
                self._data = _data
                with TimeInspector.logt("fit & process data (cached)"):
                    if init_type == DataHandlerLP.IT_FIT_SEQ:
                        self.fit_process_data()
                    elif init_type == DataHandlerLP.IT_FIT_IND:
                        self.fit()
                        self.process_data()
                    elif init_type == DataHandlerLP.IT_LS:
                        self.process_data()
                    else:
                        raise NotImplementedError(f"Unknown init_type: {init_type}")
                return

            cache_file.unlink(missing_ok=True)
            _cache_was_corrupt = True
            print(f"  [Cache] Removing corrupt cache ({cache_file.name})")

        super().setup_data(init_type=init_type, **kwargs)

        if (
            cache_file is not None
            and not _cache_was_corrupt
            and self._data is not None
            and self._data is not None
            and not self._data.empty
        ):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            _dump_data(self._data, str(cache_file))
            print(f"  [Cache] Saved {len(self._data)} rows → {cache_file}")


# ── helpers ──────────────────────────────────────────────────────────


def _resolve_handler_config(template_cfg: dict) -> dict | None:
    """Extract the handler config dict from a workflow YAML template."""
    try:
        return template_cfg["task"]["dataset"]["kwargs"]["handler"]
    except KeyError:
        return None


def _dump_data(df: pd.DataFrame, path: str) -> None:
    """Save a MultiIndex (datetime, instrument) DataFrame to disk (pickle).

    Pickle is used for reliability with wide DataFrames (200+ features).
    """
    path = str(path)
    df_to_save = df.reset_index()
    if isinstance(df_to_save.columns, pd.MultiIndex):
        df_to_save.columns = [str(col) for col in df_to_save.columns]
    df_to_save.to_pickle(path)


def _load_data(
    path: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame | None:
    """Load cached DataFrame, optionally filtering by date range."""
    path = str(path)
    if not Path(path).exists():
        return None
    try:
        # Try pickle first (primary format; also catch renamed .parquet files)
        _ext = Path(path).suffix.lower()
        if _ext in (".pkl", ".pickle", ".parquet"):
            if _PARQUET_OK and _ext == ".parquet":
                import pyarrow.parquet as pq
                with redirect_stdout(open(os.devnull, "w")):
                    table = pq.read_table(path)
                    df = table.to_pandas()
            else:
                df = pd.read_pickle(path)
        else:
            df = pd.read_pickle(path)

        # Normalize MultiIndex columns to plain strings.
        # pyarrow's table.to_pandas() may or may not reconstruct the MultiIndex
        # column structure from the parquet metadata depending on the version.
        # Flattening here ensures consistent column name format for _find_col.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col) for col in df.columns]

        if start_time and end_time and not df.empty:
            date_col = _find_col(df, "datetime")
            if date_col is None:
                raise KeyError("datetime column not found in cached data")
            dates = pd.to_datetime(df[date_col])
            mask = (dates >= pd.Timestamp(start_time)) & (dates <= pd.Timestamp(end_time))
            df = df[mask]

        date_col = _find_col(df, "datetime")
        inst_col = _find_col(df, "instrument")
        if date_col and inst_col:
            df = df.set_index([date_col, inst_col])
            df.index = df.index.set_names(["datetime", "instrument"])

        # Rebuild tuple columns from string reprs if needed
        # e.g. "('feature', 'KMID')" → ("feature", "KMID")
        _rebuild_tuple_columns(df)

        # Convert tuple columns to proper MultiIndex for downstream compatibility.
        # After _rebuild_tuple_columns, columns are a flat Index of tuples like
        # ('feature', 'KMD').  qlib processors (e.g. CSZScoreNorm) call
        # df.columns.get_loc('feature') which requires a proper MultiIndex.
        if len(df.columns) > 0 and isinstance(df.columns[0], tuple):
            df.columns = pd.MultiIndex.from_tuples(df.columns)

        return df
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [Cache] Read error ({e}), falling back")
        return None


def _rebuild_tuple_columns(df: pd.DataFrame) -> None:
    """Convert string-repr columns like ``"('feature', 'KMID')"`` back to tuples.

    Mutates ``df.columns`` in-place.  Uses ``ast.literal_eval`` on any column
    whose string representation looks like ``('x', 'y')`` — i.e. starts with
    ``('`` and ends with ``')``.
    """
    new_cols = []
    changed = False
    for col in df.columns:
        if (
            isinstance(col, str)
            and col.startswith("('")
            and col.endswith("')")
        ):
            try:
                parsed = ast.literal_eval(col)
                if isinstance(parsed, tuple) and len(parsed) == 2:
                    new_cols.append(parsed)
                    changed = True
                    continue
            except (ValueError, SyntaxError):
                pass
        new_cols.append(col)
    if changed:
        df.columns = pd.Index(new_cols)


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    """Find a column by simple name or tuple name (qlib MultiIndex convention).

    Handles three cases:
    1. Bare string column name: ``key`` (e.g. ``"datetime"``)
    2. Tuple column name: ``(key, "")`` (e.g. ``("datetime", "")``)
    3. PyArrow string repr of a tuple: ``"('datetime', '')"``
    4. (fallback) DateTime dtype column when looking for ``"datetime"``
    """
    if key in df.columns:
        return key
    tup = (key, "")
    if tup in df.columns:
        return tup
    # Parquet written by pandas with MultiIndex columns stores tuple names
    # as string reprs like "('datetime', '')"
    tuple_str = f"('{key}', '')"
    for col in df.columns:
        if str(col) == tuple_str:
            return col
    # Fallback: when the key is "datetime", find the first datetime-dtype column.
    # This handles cases where the column name format is unexpected.
    if key == "datetime":
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return col
            except Exception:
                pass
    return None
