#!/usr/bin/env python3
"""
generate_h5.py — 按指数生成 HDF5 数据文件

从 tushare/extra_data 中提取指定指数的成分股数据（58 个原始字段），
生成 H5 文件，供 fin_factor 和 build_features_from_h5.py 使用。

用法:
  python3 scripts/generate_h5.py csi300          # 默认
  python3 scripts/generate_h5.py csi1000
  python3 scripts/generate_h5.py csi300 --output /tmp/mydata.h5
  python3 scripts/generate_h5.py --help
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──
ROOT = Path(__file__).resolve().parents[1]
EXTRA_DATA = ROOT / "tushare" / "extra_data"
INSTRUMENTS = ROOT / "tushare" / "cn_data" / "instruments"
DEFAULT_OUTPUT_DIR = ROOT / "rdagent_workspace" / "factor_data_template"

# ── 58 fields ──
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
ALL_FIELDS = MARKET_FIELDS + VALUATION_FIELDS + FUNDAMENTAL_FIELDS + FINANCIAL_FIELDS


# ── Helpers ──
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
            cur = vals[ai]
            ai += 1
        result[i] = cur
    return result


def _align_daily(df: pd.DataFrame, col: str, calendar_compact: list[str]) -> np.ndarray:
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


# ── Per-stock extraction ──
def extract_stock(stock_dir: Path, calendar_compact: list[str]) -> dict[str, np.ndarray]:
    sym = stock_dir.name
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

    # Market (10)
    if daily is not None:
        arrays["$open"]   = _align_daily(daily, "open",   calendar_compact)
        arrays["$high"]   = _align_daily(daily, "high",   calendar_compact)
        arrays["$low"]    = _align_daily(daily, "low",    calendar_compact)
        arrays["$close"]  = _align_daily(daily, "close",  calendar_compact)
        arrays["$volume"] = _align_daily(daily, "vol",    calendar_compact)
        arrays["$amount"] = _align_daily(daily, "amount", calendar_compact)
        vol = arrays["$volume"]
        amt = arrays["$amount"]
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

    # Valuation (15)
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

    # Fundamental (20)
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

    # Financial statements (13)
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

        ta = arrays.get("$total_assets")
        tl = arrays.get("$total_liab")
        op = arrays.get("$operate_profit")
        tr_arr = arrays.get("$revenue")
        oc = arrays.get("$ocf")
        ni = arrays.get("$n_income")
        arrays["$liab_to_eqty"]  = _safe_div(tl, (ta - tl)).astype(np.float32) if ta is not None and tl is not None else nan()
        arrays["$op_to_revenue"] = (_safe_div(op, tr_arr) * 100).astype(np.float32) if op is not None and tr_arr is not None else nan()
        arrays["$ocf_to_profit"] = _safe_div(oc, ni).astype(np.float32) if oc is not None and ni is not None else nan()
        arrays["$ocf_to_assets"] = (_safe_div(oc, ta) * 100).astype(np.float32) if oc is not None and ta is not None else nan()
    else:
        for f in FINANCIAL_FIELDS:
            arrays[f] = nan()

    return arrays


# ── Index filter ──
def load_current_index_stocks(index_name: str) -> set[str]:
    """Load current constituent symbols from index file."""
    path = INSTRUMENTS / f"{index_name}.txt"
    if not path.exists():
        print(f"ERROR: 指数文件不存在: {path}")
        sys.exit(1)
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "join", "leave"])
    latest_leave = df["leave"].max()
    current = set(df[df["leave"] == latest_leave]["code"].unique())
    print(f"  {index_name}: {len(df)} 条记录, {len(current)} 当前成分股")
    return current


# ── Parallel worker ──
def _process_chunk(args):
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

    chunk_df = pd.concat(frames.values())[ALL_FIELDS]
    n_rows = len(chunk_df)
    chunk_df.to_hdf(tmp_path, key="chunk", mode="w", complevel=1, complib="zlib")
    del frames, chunk_df
    gc.collect()
    return (chunk_id, tmp_path, n_rows)


# ── Main ──
def main():
    parser = ArgumentParser(description="按指数生成 HDF5 数据文件")
    parser.add_argument("index", nargs="?", default="csi300",
                        choices=["csi300", "csi1000", "csi500", "csi800", "all"],
                        help="指数名称 (默认: csi300)")
    parser.add_argument("--output", "-o", default=None,
                        help="输出 H5 文件路径 (默认: rdagent_workspace/factor_data_template/daily_pv_{index}.h5)")
    parser.add_argument("--workers", "-w", type=int, default=0,
                        help="并行 worker 数 (默认: min(cpu_count, 16))")
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else min(cpu_count(), 16)

    # ── 确定输出路径 ──
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = DEFAULT_OUTPUT_DIR / f"daily_pv_{args.index}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"指数: {args.index}")
    print(f"输出: {out_path}")
    print(f"Workers: {num_workers}")

    # ── 加载成分股列表 ──
    current_codes = load_current_index_stocks(args.index)

    # ── 匹配 extra_data 目录 ──
    stock_dirs = [
        EXTRA_DATA / code
        for code in current_codes
        if (EXTRA_DATA / code).is_dir() and (EXTRA_DATA / code / "daily.csv").exists()
    ]
    missing = current_codes - {d.name for d in stock_dirs}
    if missing:
        print(f"  extra_data 中缺失: {len(missing)} 只 ({', '.join(sorted(missing)[:10])}{'...' if len(missing) > 10 else ''})")
    print(f"  extra_data 有数据: {len(stock_dirs)} 只")

    if not stock_dirs:
        print("ERROR: 没有可用的股票数据，终止")
        sys.exit(1)

    # ── 构建日历 ──
    print("构建日历...", end=" ", flush=True)
    calendar = _build_calendar(stock_dirs)
    cal_compact = [d.replace("-", "") for d in calendar]
    print(f"{len(calendar)} 个交易日 ({calendar[0]} ~ {calendar[-1]})")

    # ── 分片 ──
    chunk_size = max(1, len(stock_dirs) // num_workers)
    chunks = [stock_dirs[i:i + chunk_size] for i in range(0, len(stock_dirs), chunk_size)]
    print(f"分片: {len(chunks)} (平均 {chunk_size} 只/片)")

    tmp_dir = Path(out_path.parent) / f".h5gen_{args.index}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"临时目录: {tmp_dir}")

    worker_args = [
        (chunk, calendar, cal_compact, str(tmp_dir / f"chunk_{i:03d}.h5"), i)
        for i, chunk in enumerate(chunks)
    ]

    print(f"处理 {len(stock_dirs)} 只股票...")
    with Pool(num_workers) as pool:
        results = pool.map(_process_chunk, worker_args)

    # ── 收集合并 ──
    h5_paths = []
    total_rows = 0
    for chunk_id, path, n_rows in sorted(results, key=lambda x: x[0]):
        if path is not None and n_rows > 0:
            h5_paths.append(path)
            total_rows += n_rows
    print(f"收集 {len(h5_paths)} 个分片 ({total_rows} 行)")

    print("合并中...", end=" ", flush=True)
    BATCH_SIZE = max(1, min(4, len(h5_paths) // 4))
    merged_paths = []

    for bi in range(0, len(h5_paths), BATCH_SIZE):
        batch = h5_paths[bi:bi + BATCH_SIZE]
        frames = [pd.read_hdf(p, key="chunk") for p in batch]
        batch_df = pd.concat(frames)
        del frames
        gc.collect()
        mp = str(tmp_dir / f"merged_{bi:03d}.h5")
        batch_df.to_hdf(mp, key="chunk", mode="w", complevel=1, complib="zlib")
        merged_paths.append(mp)
        del batch_df
        gc.collect()

    frames = [pd.read_hdf(mp, key="chunk") for mp in merged_paths]
    data = pd.concat(frames).sort_index()
    del frames
    gc.collect()

    # ── 写入 ──
    data.to_hdf(str(out_path), key="data", mode="w")
    print(f"写入完成: shape={data.shape}")
    print(f"  datetime 范围: {data.index.get_level_values('datetime').min()} ~ {data.index.get_level_values('datetime').max()}")
    insts = data.index.get_level_values("instrument").unique()
    print(f"  instrument 数量: {len(insts)}")

    # ── 清理 ──
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"清理临时目录: {tmp_dir}")
    print("完成。")


if __name__ == "__main__":
    main()
