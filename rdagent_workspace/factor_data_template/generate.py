"""Generate daily_pv_all.h5 and daily_pv_debug.h5 from extra_data CSV files.

Reads ALL 58 features directly from tushare/extra_data/{SYMBOL}/*.csv,
bypassing qlib entirely. fin_factor uses these HDF5 files to understand
available data columns for factor mining.

Usage:
    cd rdagent_workspace/factor_data_template && python generate.py
"""
import gc
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
EXTRA_DATA = ROOT / "tushare" / "extra_data"
OUT_DIR = Path(__file__).resolve().parent

# ── Output column order (58 fields, with $ prefix) ───────────────────
MARKET_FIELDS = [
    "$adjclose", "$amount", "$change", "$close", "$factor",
    "$high", "$low", "$open", "$volume", "$vwap",
]
VALUATION_FIELDS = [
    "$pe", "$pe_ttm", "$pb", "$ps", "$ps_ttm",
    "$dv_ratio", "$dv_ttm", "$turnover", "$turnover_f",
    "$vol_ratio", "$total_mv", "$circ_mv", "$total_sh", "$float_sh", "$free_sh",
]
FUNDAMENTAL_FIELDS = [
    "$eps", "$dt_eps", "$bps", "$ocfps", "$cfps", "$revenue_ps",
    "$undist_ps", "$roe", "$roe_yearly", "$roa_yearly", "$npta",
    "$netprofit_margin", "$debt_to_assets", "$assets_to_eqt",
    "$eps_yoy", "$netprofit_yoy", "$roe_yoy", "$bps_yoy",
    "$assets_yoy", "$revenue_yoy",
]
FINANCIAL_FIELDS = [
    "$revenue", "$n_income", "$operate_profit",
    "$total_assets", "$total_liab", "$total_equity",
    "$ocf", "$icf", "$fcf",
    "$liab_to_eqty", "$op_to_revenue", "$ocf_to_profit", "$ocf_to_assets",
]
ALL_OUTPUT_FIELDS = MARKET_FIELDS + VALUATION_FIELDS + FUNDAMENTAL_FIELDS + FINANCIAL_FIELDS


# ── Helpers ──────────────────────────────────────────────────────────
def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, np.nan)


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, low_memory=False)
    id_cols = {"ts_code", "trade_date", "ann_date", "f_ann_date", "end_date",
               "report_type", "comp_type", "end_type"}
    for c in df.columns:
        if c not in id_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _forward_fill(df: pd.DataFrame, col: str, calendar_compact: list[str]) -> np.ndarray:
    """Forward-fill a quarterly column (ann_date) to the daily calendar."""
    sub = df[["ann_date", col]].dropna(subset=[col, "ann_date"])
    if sub.empty:
        return np.full(len(calendar_compact), np.nan, dtype=np.float32)
    sub = sub.sort_values("ann_date").drop_duplicates(subset=["ann_date"], keep="last")
    ann = sub["ann_date"].astype(str).values
    vals = sub[col].values.astype(np.float32)
    result = np.full(len(calendar_compact), np.nan, dtype=np.float32)
    cur, ai = np.nan, 0
    for i, cd in enumerate(calendar_compact):
        cc = cd.replace("-", "")
        while ai < len(ann) and ann[ai] <= cc:
            cur = vals[ai]; ai += 1
        result[i] = cur
    return result


def _align_daily(df: pd.DataFrame, col: str, calendar_compact: list[str]) -> np.ndarray:
    """Exact match of trade_date to calendar."""
    sub = df[["trade_date", col]].dropna(subset=[col])
    if sub.empty:
        return np.full(len(calendar_compact), np.nan, dtype=np.float32)
    lut = {}
    for _, row in sub.iterrows():
        v = row[col]
        if pd.notna(v):
            lut[str(row["trade_date"])] = float(v)
    result = np.full(len(calendar_compact), np.nan, dtype=np.float32)
    for i, d in enumerate(calendar_compact):
        dc = d.replace("-", "")
        if dc in lut:
            result[i] = np.float32(lut[dc])
    return result


def _build_calendar(dirs: list[Path]) -> list[str]:
    dates: set[str] = set()
    for d in dirs:
        p = d / "daily.csv"
        if not p.exists():
            continue
        try:
            for dt in pd.read_csv(p, dtype=str, usecols=["trade_date"])["trade_date"].dropna():
                s = str(dt).strip()
                if len(s) == 8:
                    dates.add(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
        except Exception:
            continue
    return sorted(dates)


# ── Per-stock extraction ─────────────────────────────────────────────
def extract_stock(stock_dir: Path, calendar_compact: list[str]) -> dict[str, np.ndarray]:
    """Extract all 58 fields for one stock, aligned to the global calendar."""
    sym = stock_dir.name
    inst = sym.upper()
    n = len(calendar_compact)
    nan = lambda: np.full(n, np.nan, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {}

    daily = _load_csv(stock_dir / "daily.csv")
    dbasic = _load_csv(stock_dir / "daily_basic.csv")
    adj_csv = _load_csv(stock_dir / "adj_factor.csv")
    fina = _load_csv(stock_dir / "fina_indicator.csv")
    income = _load_csv(stock_dir / "income.csv")
    balance = _load_csv(stock_dir / "balancesheet.csv")
    cashflow = _load_csv(stock_dir / "cashflow.csv")

    # ── 前复权因子: close_adj[t] = close[t] * adj_factor[t] / adj_factor[latest] ──
    _adj_ratio = None
    if adj_csv is not None and "adj_factor" in adj_csv.columns:
        af = _align_daily(adj_csv, "adj_factor", calendar_compact)
        if af is not None and np.any(~np.isnan(af)):
            last_val = np.float32(1.0)
            for i in range(n):
                if np.isnan(af[i]):
                    af[i] = last_val
                else:
                    last_val = af[i]
            latest_af = af[-1]
            if latest_af > 0:
                _adj_ratio = (af / latest_af).astype(np.float32)

    # ── Market (10) from daily.csv ──
    if daily is not None:
        arrays["$open"]   = _align_daily(daily, "open",   calendar_compact)
        arrays["$high"]   = _align_daily(daily, "high",   calendar_compact)
        arrays["$low"]    = _align_daily(daily, "low",    calendar_compact)
        arrays["$close"]  = _align_daily(daily, "close",  calendar_compact)
        arrays["$volume"] = _align_daily(daily, "vol",    calendar_compact)
        arrays["$amount"] = _align_daily(daily, "amount", calendar_compact)

        vol = arrays["$volume"]; amt = arrays["$amount"]
        arrays["$vwap"] = _safe_div(amt, vol).astype(np.float32)

        if _adj_ratio is not None:
            # 前复权调整
            arrays["$close"]  = (arrays["$close"]  * _adj_ratio).astype(np.float32)
            arrays["$open"]   = (arrays["$open"]   * _adj_ratio).astype(np.float32)
            arrays["$high"]   = (arrays["$high"]   * _adj_ratio).astype(np.float32)
            arrays["$low"]    = (arrays["$low"]    * _adj_ratio).astype(np.float32)
            arrays["$vwap"]   = (arrays["$vwap"]   * _adj_ratio).astype(np.float32)
            arrays["$adjclose"] = arrays["$close"].copy()
            arrays["$factor"] = _adj_ratio.copy()
            # 基于复权价的真实日收益率
            chg = np.full(n, np.nan, dtype=np.float32)
            chg[1:] = arrays["$close"][1:] / arrays["$close"][:-1] - 1.0
            arrays["$change"] = chg
        else:
            arrays["$adjclose"] = arrays["$close"].copy()
            arrays["$change"] = _align_daily(daily, "pct_chg", calendar_compact) / 100.0 if "pct_chg" in daily.columns else nan()
            if "pre_close" in daily.columns:
                pc = _align_daily(daily, "pre_close", calendar_compact)
                arrays["$factor"] = _safe_div(arrays["$close"], pc).astype(np.float32)
            else:
                arrays["$factor"] = nan()
    else:
        for f in MARKET_FIELDS:
            arrays[f] = nan()

    # ── Valuation (15) from daily_basic.csv ──
    _vb = [
        ("pe", "$pe"), ("pe_ttm", "$pe_ttm"), ("pb", "$pb"),
        ("ps", "$ps"), ("ps_ttm", "$ps_ttm"),
        ("dv_ratio", "$dv_ratio"), ("dv_ttm", "$dv_ttm"),
        ("turnover_rate", "$turnover"), ("turnover_rate_f", "$turnover_f"),
        ("volume_ratio", "$vol_ratio"),
        ("total_mv", "$total_mv"), ("circ_mv", "$circ_mv"),
        ("total_share", "$total_sh"), ("float_share", "$float_sh"),
        ("free_share", "$free_sh"),
    ]
    for csv_col, out_name in _vb:
        arrays[out_name] = _align_daily(dbasic, csv_col, calendar_compact) if dbasic is not None else nan()

    # ── Fundamental (20) from fina_indicator.csv ──
    _fm = [
        ("eps", "$eps"), ("dt_eps", "$dt_eps"), ("bps", "$bps"),
        ("ocfps", "$ocfps"), ("cfps", "$cfps"), ("revenue_ps", "$revenue_ps"),
        ("undist_profit_ps", "$undist_ps"),
        ("roe", "$roe"), ("roe_yearly", "$roe_yearly"), ("roa_yearly", "$roa_yearly"),
        ("npta", "$npta"), ("netprofit_margin", "$netprofit_margin"),
        ("debt_to_assets", "$debt_to_assets"), ("assets_to_eqt", "$assets_to_eqt"),
        ("basic_eps_yoy", "$eps_yoy"), ("netprofit_yoy", "$netprofit_yoy"),
        ("roe_yoy", "$roe_yoy"), ("bps_yoy", "$bps_yoy"),
        ("assets_yoy", "$assets_yoy"), ("or_yoy", "$revenue_yoy"),
    ]
    for csv_col, out_name in _fm:
        arrays[out_name] = _forward_fill(fina, csv_col, calendar_compact) if fina is not None else nan()

    # ── Financial statements (13) — merge income+balancesheet+cashflow on ann_date ──
    _fs_sources = {
        "income":        ["total_revenue", "n_income", "operate_profit"],
        "balancesheet":  ["total_assets", "total_liab", "total_hldr_eqy_exc_min_int"],
        "cashflow":      ["n_cashflow_act", "n_cashflow_inv_act", "free_cashflow"],
    }
    _df_map = {"income": income, "balancesheet": balance, "cashflow": cashflow}
    merged = None
    for src, cols in _fs_sources.items():
        src_df = _df_map[src]
        if src_df is None:
            continue
        avail = ["ann_date"] + [c for c in cols if c in src_df.columns]
        sub = src_df[avail].copy()
        merged = sub if merged is None else merged.merge(sub, on="ann_date", how="outer")

    if merged is not None:
        _fo = [
            ("total_revenue", "$revenue"), ("n_income", "$n_income"),
            ("operate_profit", "$operate_profit"),
            ("total_assets", "$total_assets"), ("total_liab", "$total_liab"),
            ("total_hldr_eqy_exc_min_int", "$total_equity"),
            ("n_cashflow_act", "$ocf"), ("n_cashflow_inv_act", "$icf"),
            ("free_cashflow", "$fcf"),
        ]
        for csv_col, out_name in _fo:
            arrays[out_name] = _forward_fill(merged, csv_col, calendar_compact) if csv_col in merged.columns else nan()

        ta = arrays.get("$total_assets"); tl = arrays.get("$total_liab")
        op = arrays.get("$operate_profit"); tr_arr = arrays.get("$revenue")
        oc = arrays.get("$ocf"); ni = arrays.get("$n_income")

        arrays["$liab_to_eqty"]   = _safe_div(tl, (ta - tl)).astype(np.float32) if ta is not None and tl is not None else nan()
        arrays["$op_to_revenue"]  = (_safe_div(op, tr_arr) * 100).astype(np.float32) if op is not None and tr_arr is not None else nan()
        arrays["$ocf_to_profit"]  = _safe_div(oc, ni).astype(np.float32) if oc is not None and ni is not None else nan()
        arrays["$ocf_to_assets"]  = (_safe_div(oc, ta) * 100).astype(np.float32) if oc is not None and ta is not None else nan()
    else:
        for f in FINANCIAL_FIELDS:
            arrays[f] = nan()

    return arrays


# ── Parallel chunk worker ────────────────────────────────────────────

def _process_chunk(args):
    """Worker: process a chunk of stocks, write to temporary parquet.

    Args:
        args: (stock_dirs, calendar, calendar_compact, tmp_path, chunk_id)

    Returns:
        (chunk_id, tmp_path, n_rows) or (chunk_id, None, 0) if empty
    """
    stock_dirs, calendar, cal_compact, tmp_path, chunk_id = args
    frames = {}
    for sd in stock_dirs:
        arrays = extract_stock(sd, cal_compact)
        df = pd.DataFrame(arrays, index=pd.to_datetime(calendar))
        df.index.name = "datetime"
        df["instrument"] = sd.name.upper()
        df = df.set_index("instrument", append=True)
        frames[sd.name.upper()] = df

    if not frames:
        return (chunk_id, None, 0)

    chunk_df = pd.concat(frames.values())[ALL_OUTPUT_FIELDS]
    n_rows = len(chunk_df)
    chunk_df.to_hdf(tmp_path, key="chunk", mode="w", complevel=1, complib="zlib")
    del frames, chunk_df
    gc.collect()
    return (chunk_id, tmp_path, n_rows)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    if not EXTRA_DATA.exists():
        print(f"ERROR: extra_data not found at {EXTRA_DATA}")
        sys.exit(1)

    _env_workers = os.environ.get("GENERATE_NUM_WORKERS", "")
    NUM_WORKERS = int(_env_workers) if _env_workers else min(cpu_count(), 16)
    # Explicit env override takes priority; default uses cpu_count capped at 16
    print(f"Workers: {NUM_WORKERS}")

    stock_dirs = sorted(d for d in EXTRA_DATA.iterdir() if d.is_dir() and (d / "daily.csv").exists())
    print(f"Stock dirs with daily.csv: {len(stock_dirs)}")

    # ── Phase 1: Build global calendar (sequential, fast) ──
    print("Building global calendar ...", end=" ", flush=True)
    calendar = _build_calendar(stock_dirs)
    cal_compact = [d.replace("-", "") for d in calendar]
    print(f"{len(calendar)} trading days ({calendar[0]} ~ {calendar[-1]})")

    # ── Phase 2: Process stocks in parallel (16 workers) ──
    # Split stock_dirs into chunks, one per worker
    chunk_size = max(1, len(stock_dirs) // NUM_WORKERS)
    chunks = [stock_dirs[i:i + chunk_size] for i in range(0, len(stock_dirs), chunk_size)]
    print(f"Chunks: {len(chunks)} (avg {chunk_size} stocks/chunk)")

    # Temporary directory for intermediate files (use OUT_DIR so Docker mount has fast I/O)
    tmp_dir = os.environ.get("GENERATE_TMP_DIR",
        str(OUT_DIR / f".h5gen_tmp_{os.getpid()}"))
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Temp dir: {tmp_dir}")

    # Build args for each worker
    worker_args = [
        (chunk, calendar, cal_compact, os.path.join(tmp_dir, f"chunk_{i:03d}.h5"), i)
        for i, chunk in enumerate(chunks)
    ]

    print(f"Processing {len(stock_dirs)} stocks with {NUM_WORKERS} workers ...")
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(_process_chunk, worker_args)

    # ── Phase 3: Collect results from parquet files ──
    parquet_paths = []
    total_rows = 0
    for chunk_id, path, n_rows in sorted(results, key=lambda x: x[0]):
        if path is not None and n_rows > 0:
            parquet_paths.append(path)
            total_rows += n_rows
    print(f"Collected {len(parquet_paths)} parquet files ({total_rows} rows)")

    # Batch concat: merge 4-5 chunks at a time to control peak memory
    print("Concatenating ...", end=" ", flush=True)
    out_all = OUT_DIR / "daily_pv_all.h5"
    BATCH_SIZE = max(1, min(4, len(parquet_paths) // 4))
    merged_paths = []

    for bi in range(0, len(parquet_paths), BATCH_SIZE):
        batch = parquet_paths[bi:bi + BATCH_SIZE]
        frames = [pd.read_hdf(p, key="chunk") for p in batch]
        batch_df = pd.concat(frames)
        del frames
        gc.collect()
        merged_path = os.path.join(tmp_dir, f"merged_{bi:03d}.h5")
        batch_df.to_hdf(merged_path, key="chunk", mode="w", complevel=1, complib="zlib")
        merged_paths.append(merged_path)
        print(f"[{bi + len(batch)}/{len(parquet_paths)}]", end=" ", flush=True)
        del batch_df
        gc.collect()

    # Final merge of merged batches
    print("final ...", end=" ", flush=True)
    frames = []
    for mp in merged_paths:
        frames.append(pd.read_hdf(mp, key="chunk"))
    data = pd.concat(frames).sort_index()
    del frames
    gc.collect()

    # ── Write HDF5 ──
    data.to_hdf(str(out_all), key="data", mode="w")
    print(f"full shape={data.shape}")

    # Debug: first 300 instruments, 2020-2024
    debug_inst = sorted(data.index.get_level_values("instrument").unique())[:300]
    ds = os.environ.get("GENERATE_DEBUG_START", "2020-01-01")
    de = os.environ.get("GENERATE_DEBUG_END", "2024-12-31")
    mask = (data.index.get_level_values("instrument").isin(debug_inst)
            & (data.index.get_level_values("datetime") >= ds)
            & (data.index.get_level_values("datetime") <= de))
    debug = data.loc[mask].copy()
    print(f"Debug shape={debug.shape}  instruments={len(debug_inst)}  range=[{ds}, {de}]")
    out_debug = OUT_DIR / "daily_pv_debug.h5"
    debug.to_hdf(str(out_debug), key="data", mode="w")
    print(f"{out_debug} written")
    del data, debug
    gc.collect()

    # ── Cleanup ──
    if os.environ.get("GENERATE_KEEP_TMP") != "1":
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up {tmp_dir}")
    else:
        print(f"Kept temp dir: {tmp_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
