"""
process_extra_data.py - 数据新鲜度检测、增量更新与因子特征构建

功能:
  1. 检测 extra_data 中每只股票的 CSV 数据是否最新
  2. 若不是最新，通过 tushare 接口对 extra_data(CSV) 做增量更新
  3. 从 extra_data CSV 全量转换为 qlib bin 格式 → cn_extra_data_improve
  4. 按 tushare/new_factor.md 定义的 30 个因子公式计算衍生因子，包括:
     - 动量/反转: momentum_5d/10d/20d, reversal_1d/2d/5d/20d
     - 波动率: realized_volatility_20d, intraday_volatility, avg_normalized_range_5d
     - 震荡: rsi_14d
     - 流动性: volume_change_5d, volume_ratio_5d, volume_ratio_5d_20d, avg_volume_ratio_20d, turnover_trend
     - 估值: pe_ttm (trailing_PE_ratio), pb_ratio (PB_Ratio), earnings_yield, book_to_price
     - 量价: obv_slope_10d, volume_weighted_momentum_5d, vwap_deviation_5d/10d
     - 风险调整: sharpe_10d, momentum_vol_adjusted_20, risk_adjusted_momentum_5d_20d
     - 质量: roe_yearly (roe), netprofit_margin (net_profit_margin)
     - 截面: sector_relative_pb (Sector_Relative_PB)
  5. 最终仅输出 30 个因子 bin 文件 + close，不含多余的中间特征

运行方式:
  docker run --rm -v $(pwd):/workspace -w /workspace zhuhai123/local_qlib:v1-tushare \\
    python3 process_extra_data.py --test          # 用 5 支股票验证
  docker run --rm -v $(pwd):/workspace -w /workspace zhuhai123/local_qlib:v1-tushare \\
    python3 process_extra_data.py --symbols SH600000 SZ000001  # 指定股票
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 避免本地目录遮蔽
for p in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
    while p in sys.path:
        sys.path.remove(p)

import sxsc_tushare as sx

# ============================================================
# 配置
# ============================================================
# Tushare API 三级限流: 20次/秒, 300次/分钟, 3000次/小时
# RateLimiter 使用安全阈值: 18/s, 280/min, 2800/h (留10%余量)
TUSHARE_TOKEN = "4cbb80cf41ae83b53f9bc431a502c328565e53938bce7cadce52bc2a"
BIN_SUFFIX = ".day.bin"
TEST_SYMBOLS = ["SH600000", "SH600004", "SH600006", "SZ000001", "SZ000002"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def symbol_to_ts_code(symbol):
    prefix = symbol[:2].upper()
    code = symbol[2:]
    return f"{code}.{prefix}"


# ============================================================
# API 限速器: 确保不超过 Tushare 三级限流
#   20次/秒, 300次/分钟, 3000次/小时
# ============================================================
class RateLimiter:
    """滑动窗口限速器，确保同时满足秒/分/时三级限制"""

    def __init__(self, max_per_sec=18, max_per_min=280, max_per_hour=2800):
        self.max_per_sec = max_per_sec
        self.max_per_min = max_per_min
        self.max_per_hour = max_per_hour
        self._timestamps = []

    def _trim_window(self, now):
        cutoff = now - 3600
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def acquire(self):
        now = time.time()
        self._trim_window(now)

        # 检查三级限流，计算需要等待的时间
        wait = 0.0

        # 秒级: 过去1秒内的调用次数
        sec_calls = sum(1 for t in self._timestamps if t > now - 1)
        if sec_calls >= self.max_per_sec:
            # 等到最早的那次调用过期
            recent = sorted([t for t in self._timestamps if t > now - 1])
            wait = max(wait, recent[0] + 1.001 - now)

        # 分钟级: 过去60秒内的调用次数
        min_calls = sum(1 for t in self._timestamps if t > now - 60)
        if min_calls >= self.max_per_min:
            recent = sorted([t for t in self._timestamps if t > now - 60])
            wait = max(wait, recent[0] + 60.001 - now)

        # 小时级: 过去3600秒内的调用次数
        hour_calls = len(self._timestamps)
        if hour_calls >= self.max_per_hour:
            wait = max(wait, self._timestamps[0] + 3600.001 - now)

        if wait > 0:
            time.sleep(wait)
            now = time.time()

        self._timestamps.append(now)

    @property
    def count_last_minute(self):
        now = time.time()
        return sum(1 for t in self._timestamps if t > now - 60)

    @property
    def count_last_hour(self):
        now = time.time()
        return sum(1 for t in self._timestamps if t > now - 3600)


# ============================================================
# API 封装
# ============================================================
class TushareAPI:
    def __init__(self):
        sx.set_token(TUSHARE_TOKEN)
        self.api = sx.get_api(env="prd")
        self._limiter = RateLimiter()

    def query(self, api_name, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                self._limiter.acquire()
                df = self.api.query(api_name, **kwargs)
                return df if df is not None else pd.DataFrame()
            except Exception as e:
                logger.warning(f"[{api_name}] attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
        return pd.DataFrame()

    def fetch_daily(self, ts_code, start_date, end_date):
        return self.query("daily", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_daily_basic(self, ts_code, start_date, end_date):
        return self.query("daily_basic", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_adj_factor(self, ts_code, start_date, end_date):
        """按年批量拉取复权因子，减少 API 调用次数"""
        all_dfs = []
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        for yr in range(start_year, end_year + 1):
            sd = f"{yr}0101"
            ed = f"{yr}1231"
            try:
                df = self.query("adj_factor", ts_code=ts_code, start_date=sd, end_date=ed)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"adj_factor {yr}: {e}")
        if not all_dfs:
            return pd.DataFrame()
        result = pd.concat(all_dfs, ignore_index=True)
        return result.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")

    def fetch_fina_indicator(self, ts_code, start_date, end_date):
        return self.query("fina_indicator", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_income(self, ts_code, start_date, end_date):
        return self.query("income", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_balancesheet(self, ts_code, start_date, end_date):
        return self.query("balancesheet", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_cashflow(self, ts_code, start_date, end_date):
        return self.query("cashflow", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_dividend(self, ts_code):
        return self.query("dividend", ts_code=ts_code)

    def fetch_stock_company(self, ts_code):
        return self.query("stock_company", ts_code=ts_code)


# ============================================================
# 数据新鲜度检测 (纯文件读取，无 API 调用)
# ============================================================
def check_data_freshness(extra_data_dir, symbol, today_str=None):
    """
    检测某只股票的数据是否最新，区分日频和季频数据。

    日频 (daily/daily_basic): 容忍 2 个自然日延迟 (覆盖周末)
    季频 (fina_indicator/income/balancesheet/cashflow): 容忍 100 天
        季报在报告期后 1-4 个月才公告，且不同公司公告时间不同

    返回: (is_fresh: bool, last_date: str, detail: dict)
      is_fresh = 日频新鲜 AND 季频新鲜
    """
    stock_dir = Path(extra_data_dir) / symbol
    daily_csv = stock_dir / "daily.csv"

    if not daily_csv.exists():
        return False, "20000101", {"status": "missing"}

    today = today_str or datetime.now().strftime("%Y%m%d")
    today_dt = datetime.strptime(today, "%Y%m%d")

    detail = {}
    daily_last = "20000101"
    quarterly_last = "20000101"

    # 日频数据: 文件名 -> 日期列
    DAILY_FILES = {"daily": "trade_date", "daily_basic": "trade_date"}
    # 季频数据: 文件名 -> 日期列 (ann_date = 公告日，避免 look-ahead bias)
    QUARTERLY_FILES = {
        "fina_indicator": "ann_date",
        "income": "ann_date",
        "balancesheet": "ann_date",
        "cashflow": "ann_date",
    }
    # 非时间序列数据
    STATIC_FILES = ["dividend", "stock_company"]

    for name, date_col in {**DAILY_FILES, **QUARTERLY_FILES}.items():
        csv_path = stock_dir / f"{name}.csv"
        if not csv_path.exists():
            detail[name] = {"status": "missing"}
            continue
        try:
            dates = pd.read_csv(csv_path, dtype=str, usecols=[date_col])[date_col].dropna()
            if dates.empty:
                detail[name] = {"status": "empty_dates"}
                continue

            detail[name] = {
                "min_date": dates.min(), "max_date": dates.max(), "rows": len(dates),
                "freq": "daily" if name in DAILY_FILES else "quarterly",
            }
            if name in DAILY_FILES and dates.max() > daily_last:
                daily_last = dates.max()
            if name in QUARTERLY_FILES and dates.max() > quarterly_last:
                quarterly_last = dates.max()
        except Exception as e:
            detail[name] = {"status": "error", "error": str(e)}

    for name in STATIC_FILES:
        csv_path = stock_dir / f"{name}.csv"
        detail[name] = {"status": "exists"} if csv_path.exists() else {"status": "missing"}

    # 分别判断日频和季频新鲜度
    daily_dt = datetime.strptime(daily_last, "%Y%m%d")
    daily_behind = (today_dt - daily_dt).days
    daily_fresh = daily_behind <= 2

    if quarterly_last == "20000101":
        quarterly_fresh = False
        quarterly_behind = 999
    else:
        quarterly_dt = datetime.strptime(quarterly_last, "%Y%m%d")
        quarterly_behind = (today_dt - quarterly_dt).days
        quarterly_fresh = quarterly_behind <= 100

    is_fresh = daily_fresh and quarterly_fresh
    last_date = max(daily_last, quarterly_last)

    if not daily_fresh:
        logger.info(f"  [{symbol}] 日频数据过期: last={daily_last}, behind={daily_behind}d")
    if not quarterly_fresh:
        logger.info(f"  [{symbol}] 季频数据过期: last={quarterly_last}, behind={quarterly_behind}d")

    return is_fresh, last_date, detail


# ============================================================
# 增量数据拉取 (仅拉取缺失的日期)
# ============================================================
def fetch_incremental_data(api, extra_data_dir, symbol, ts_code, start_date, end_date):
    """
    从 Tushare 拉取新数据并追加到 CSV，追加后按日期降序重排。
    CSV 约定: 最新数据在最前面 (trade_date/ann_date 降序)。

    日频 (daily/daily_basic): 仅拉取 start_date ~ end_date 区间的新日期
    季频 (fina_indicator/income/balancesheet/cashflow): 拉取全量并去重追加
        因为季报公告日期 (ann_date) 可能晚于已存储数据的日期

    返回: 成功拉取的记录总数
    """
    stock_dir = Path(extra_data_dir) / symbol
    stock_dir.mkdir(parents=True, exist_ok=True)

    # 日频: key=(拉取函数, 日期列, 是否仅拉取增量区间)
    # 季频: 拉取全量 (fetch_start="20100101")，因为季报可能有 retroactive 公告
    fetch_map = [
        ("daily", api.fetch_daily, "trade_date", False),
        ("daily_basic", api.fetch_daily_basic, "trade_date", False),
        ("fina_indicator", api.fetch_fina_indicator, "ann_date", True),
        ("income", api.fetch_income, "ann_date", True),
        ("balancesheet", api.fetch_balancesheet, "ann_date", True),
        ("cashflow", api.fetch_cashflow, "ann_date", True),
    ]

    total_new = 0

    for csv_name, fetch_fn, date_col, is_quarterly in fetch_map:
        csv_path = stock_dir / f"{csv_name}.csv"
        existing_dates = set()

        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path, dtype=str, usecols=[date_col])
                existing_dates = set(existing[date_col].dropna().unique())
            except Exception:
                pass

        fetch_start = "20100101" if is_quarterly else start_date

        try:
            new_df = fetch_fn(ts_code, fetch_start, end_date)
            if new_df is None or new_df.empty:
                continue

            # 过滤已存在的日期
            if existing_dates:
                new_df = new_df[~new_df[date_col].astype(str).isin(existing_dates)]

            if new_df.empty:
                continue

            # 追加写入
            write_header = not csv_path.exists()
            new_df.to_csv(csv_path, mode="a", header=write_header, index=False)

            # 追加后按日期降序重排，保持 newest-first 约定
            full_df = pd.read_csv(csv_path, dtype=str)
            full_df = full_df.sort_values(date_col, ascending=False)
            full_df.to_csv(csv_path, index=False)

            total_new += len(new_df)
            logger.info(f"  {csv_name}: +{len(new_df)} 条 (重新排序)")
        except Exception as e:
            logger.error(f"  {csv_name} 拉取失败: {e}")

    return total_new


# ============================================================
# CSV -> Qlib Bin 全量转换
# ============================================================
EXTRA_DAILY_FEATURES = {
    "pe": "pe", "pe_ttm": "pe_ttm", "pb": "pb",
    "ps": "ps", "ps_ttm": "ps_ttm",
    "dv_ratio": "dv_ratio", "dv_ttm": "dv_ttm",
    "turnover": "turnover_rate", "turnover_f": "turnover_rate_f",
    "vol_ratio": "volume_ratio",
    "total_mv": "total_mv", "circ_mv": "circ_mv",
    "total_sh": "total_share", "float_sh": "float_share", "free_sh": "free_share",
}

FUNDAMENTAL_FEATURES = {
    "eps": "eps", "dt_eps": "dt_eps", "bps": "bps",
    "ocfps": "ocfps", "cfps": "cfps",
    "revenue_ps": "revenue_ps", "undist_ps": "undist_profit_ps",
    "roe": "roe", "roe_yearly": "roe_yearly", "roa_yearly": "roa_yearly",
    "npta": "npta", "netprofit_margin": "netprofit_margin",
    "debt_to_assets": "debt_to_assets", "assets_to_eqt": "assets_to_eqt",
    "eps_yoy": "basic_eps_yoy", "netprofit_yoy": "netprofit_yoy",
    "roe_yoy": "roe_yoy", "bps_yoy": "bps_yoy",
    "assets_yoy": "assets_yoy", "revenue_yoy": "or_yoy",
}

FINANCIAL_STATEMENT_SOURCES = {
    "income": {
        "revenue": "total_revenue", "n_income": "n_income",
        "operate_profit": "operate_profit",
    },
    "balancesheet": {
        "total_assets": "total_assets", "total_liab": "total_liab",
        "total_equity": "total_hldr_eqy_exc_min_int",
    },
    "cashflow": {
        "ocf": "n_cashflow_act", "icf": "n_cashflow_inv_act",
        "fcf": "free_cashflow",
    },
}


def load_csvs_raw(input_dir):
    data = {}
    for name in ["daily", "daily_basic", "fina_indicator", "income",
                  "balancesheet", "cashflow"]:
        path = Path(input_dir) / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            for c in df.columns:
                if c not in (
                    "ts_code", "trade_date", "ann_date", "f_ann_date",
                    "end_date", "report_type", "comp_type", "end_type",
                    "div_proc", "record_date", "ex_date", "pay_date",
                    "div_listdate", "imp_ann_date", "setup_date",
                    "province", "city", "website", "email", "office",
                    "introduction", "main_business", "business_scope",
                    "chairman", "manager", "secretary", "exchange",
                ):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            data[name] = df
    return data


def build_calendar(daily_df):
    dates = sorted(daily_df["trade_date"].unique())
    return [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates]


def align_series_to_calendar(series, dates_col, calendar_compact):
    result = np.full(len(calendar_compact), np.nan, dtype=np.float32)
    date_to_val = dict(zip(dates_col, series.values))
    for i, d in enumerate(calendar_compact):
        if d in date_to_val:
            v = date_to_val[d]
            if pd.notna(v):
                result[i] = np.float32(v)
    return result


def normalize_market_data(daily_df, adj_factor_df, calendar_compact):
    merged = daily_df[["trade_date", "open", "high", "low", "close",
                        "vol", "amount"]].copy()
    merged = merged.sort_values("trade_date").reset_index(drop=True)

    if not adj_factor_df.empty:
        af = adj_factor_df[["trade_date", "adj_factor"]].copy()
        merged = merged.merge(af, on="trade_date", how="left")
        merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
    else:
        merged["adj_factor"] = 1.0

    merged["adj_close"] = merged["close"].astype(float) * merged["adj_factor"].astype(float)

    base_price = merged["adj_close"].iloc[0]
    if pd.isna(base_price) or base_price <= 0:
        base_price = merged["close"].iloc[0]

    adj_factor = merged["adj_factor"].values.astype(np.float64)
    close_raw = merged["close"].values.astype(np.float64)
    adj_close = merged["adj_close"].values.astype(np.float64)

    close_norm = adj_close / base_price
    open_norm = merged["open"].values.astype(np.float64) * adj_factor / base_price
    high_norm = merged["high"].values.astype(np.float64) * adj_factor / base_price
    low_norm = merged["low"].values.astype(np.float64) * adj_factor / base_price

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_vwap = np.where(
            merged["vol"].values.astype(np.float64) > 0,
            merged["amount"].values.astype(np.float64) / merged["vol"].values.astype(np.float64) * 10,
            np.nan,
        )
    close_raw_day1 = close_raw[0]
    if pd.isna(close_raw_day1) or close_raw_day1 <= 0:
        close_raw_day1 = base_price
    vwap_norm = raw_vwap / close_raw_day1

    change = np.full(len(close_norm), np.nan, dtype=np.float64)
    if len(close_norm) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            change[1:] = np.where(
                close_norm[:-1] != 0,
                (close_norm[1:] - close_norm[:-1]) / close_norm[:-1],
                np.nan,
            )

    features = {
        "open": open_norm.astype(np.float32),
        "close": close_norm.astype(np.float32),
        "high": high_norm.astype(np.float32),
        "low": low_norm.astype(np.float32),
        "vwap": vwap_norm.astype(np.float32),
        "volume": merged["vol"].values.astype(np.float32),
        "amount": merged["amount"].values.astype(np.float32),
        "adjclose": adj_close.astype(np.float32),
        "change": change.astype(np.float32),
        "factor": (adj_factor / base_price).astype(np.float32),
    }

    aligned = {}
    for name, vals in features.items():
        aligned[name] = align_series_to_calendar(
            pd.Series(vals), merged["trade_date"].values, calendar_compact
        )
    return aligned


def extract_daily_basic_features(daily_basic_df, calendar_compact):
    features = {}
    for feat_name, col_name in EXTRA_DAILY_FEATURES.items():
        if col_name in daily_basic_df.columns:
            features[feat_name] = align_series_to_calendar(
                daily_basic_df[col_name],
                daily_basic_df["trade_date"].values,
                calendar_compact,
            )
    if "dv_ttm" in features and "dv_ratio" in features:
        mask = np.isnan(features["dv_ttm"]) & ~np.isnan(features["dv_ratio"])
        if mask.any():
            features["dv_ttm"][mask] = features["dv_ratio"][mask]
    return features


def forward_fill_fundamental(fina_df, calendar_compact):
    features = {}
    for feat_name, col_name in FUNDAMENTAL_FEATURES.items():
        if col_name not in fina_df.columns:
            continue
        df = fina_df[["ann_date", col_name]].dropna(subset=[col_name, "ann_date"])
        if df.empty:
            continue
        df = df.sort_values("ann_date").drop_duplicates(subset=["ann_date"], keep="last")

        result = np.full(len(calendar_compact), np.nan, dtype=np.float32)
        ann_dates = df["ann_date"].astype(str).values
        values = df[col_name].values.astype(np.float32)

        current_val = np.nan
        ann_idx = 0
        for i, cal_date in enumerate(calendar_compact):
            cal_comp = cal_date.replace("-", "")
            while ann_idx < len(ann_dates) and ann_dates[ann_idx] <= cal_comp:
                current_val = values[ann_idx]
                ann_idx += 1
            result[i] = current_val

        if np.sum(~np.isnan(result)) > 0:
            features[feat_name] = result
    return features


def extract_financial_statement_features(data, calendar_compact):
    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b != 0, a / b, np.nan)

    combined_dfs = []
    for src_name, feat_map in FINANCIAL_STATEMENT_SOURCES.items():
        if src_name not in data:
            continue
        df = data[src_name].copy()
        cols_needed = ["ann_date"] + list(feat_map.values())
        available = [c for c in cols_needed if c in df.columns]
        if "ann_date" not in available:
            continue
        combined_dfs.append(df[available])

    if not combined_dfs:
        return {}

    merged = combined_dfs[0]
    for df in combined_dfs[1:]:
        merged = merged.merge(df, on="ann_date", how="outer")

    if "total_assets" in merged.columns and "total_liab" in merged.columns:
        merged["liab_to_eqty"] = safe_div(
            merged["total_liab"].values,
            (merged["total_assets"].values - merged["total_liab"].values),
        )
    if "operate_profit" in merged.columns and "total_revenue" in merged.columns:
        merged["op_to_revenue"] = safe_div(
            merged["operate_profit"].values, merged["total_revenue"].values
        ) * 100
    if "n_cashflow_act" in merged.columns and "n_income" in merged.columns:
        merged["ocf_to_profit"] = safe_div(
            merged["n_cashflow_act"].values, merged["n_income"].values
        )
    if "n_cashflow_act" in merged.columns and "total_assets" in merged.columns:
        merged["ocf_to_assets"] = safe_div(
            merged["n_cashflow_act"].values, merged["total_assets"].values
        ) * 100

    final_features = {
        "revenue": "total_revenue", "n_income": "n_income",
        "operate_profit": "operate_profit",
        "total_assets": "total_assets", "total_liab": "total_liab",
        "total_equity": "total_hldr_eqy_exc_min_int",
        "ocf": "n_cashflow_act", "icf": "n_cashflow_inv_act",
        "fcf": "free_cashflow",
        "liab_to_eqty": "liab_to_eqty",
        "op_to_revenue": "op_to_revenue",
        "ocf_to_profit": "ocf_to_profit",
        "ocf_to_assets": "ocf_to_assets",
    }

    features = {}
    for feat_name, col_name in final_features.items():
        if col_name not in merged.columns:
            continue
        df = merged[["ann_date", col_name]].dropna(subset=[col_name, "ann_date"])
        if df.empty:
            continue
        df = df.sort_values("ann_date").drop_duplicates(subset=["ann_date"], keep="last")

        result = np.full(len(calendar_compact), np.nan, dtype=np.float32)
        ann_dates = df["ann_date"].astype(str).values
        values = df[col_name].values.astype(np.float32)

        current_val = np.nan
        ann_idx = 0
        for i, cal_date in enumerate(calendar_compact):
            cal_comp = cal_date.replace("-", "")
            while ann_idx < len(ann_dates) and ann_dates[ann_idx] <= cal_comp:
                current_val = values[ann_idx]
                ann_idx += 1
            result[i] = current_val

        if np.sum(~np.isnan(result)) > 0:
            features[feat_name] = result
    return features


def write_bin(filepath, start_idx, data):
    header = np.array([float(start_idx)], dtype="<f4")
    np.concatenate([header, data.astype("<f4")]).tofile(str(filepath))


# new_factor.md 24 个因子所需的原始数据字段
# 这些是从 CSV 中提取写入 bin 的最小集合，衍生因子在此基础上计算
NEW_FACTOR_RAW_INPUTS = {
    "close", "high", "low", "vwap", "volume",   # 行情
    "pe_ttm", "pb", "turnover",                  # daily_basic
    "eps", "roe_yearly", "netprofit_margin",     # fina_indicator
}


def full_convert_csv_to_bin(symbol, extra_data_dir, output_dir, global_calendar,
                            keep_features=None):
    """
    全量从 CSV 转换为 bin (需要 API 获取 adj_factor)。
    keep_features: 若指定，仅写入这些特征名对应的 bin 文件；None 则写入全部。
    返回: 实际写入的特征名列表
    """
    input_dir = Path(extra_data_dir) / symbol
    ts_code = symbol_to_ts_code(symbol)

    data = load_csvs_raw(input_dir)
    if "daily" not in data:
        logger.error(f"{symbol}: 缺少 daily.csv")
        return []

    daily_df = data["daily"].copy()
    daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)
    start_date = daily_df["trade_date"].min()
    end_date = daily_df["trade_date"].max()

    # 获取 adj_factor (按年拉取以减少 API 调用)
    api = TushareAPI()
    adj_factor_df = api.fetch_adj_factor(ts_code, start_date, end_date)

    if global_calendar is not None:
        global_compact = [d.replace("-", "") for d in global_calendar]
        stock_calendar = build_calendar(daily_df)
        stock_compact = [d.replace("-", "") for d in stock_calendar]
        first_date = stock_compact[0]
        start_idx = global_compact.index(first_date) if first_date in global_compact else 0
        calendar = global_calendar
        calendar_compact = global_compact
    else:
        calendar = build_calendar(daily_df)
        calendar_compact = [d.replace("-", "") for d in calendar]
        start_idx = 0

    feat_dir = Path(output_dir) / "features" / symbol.lower()
    feat_dir.mkdir(parents=True, exist_ok=True)

    all_features = {}

    # 行情特征
    market_features = normalize_market_data(daily_df, adj_factor_df, calendar_compact)
    all_features.update(market_features)

    # 估值特征
    if "daily_basic" in data:
        daily_basic_df = data["daily_basic"].copy().sort_values("trade_date").reset_index(drop=True)
        all_features.update(extract_daily_basic_features(daily_basic_df, calendar_compact))

    # 财务指标
    if "fina_indicator" in data:
        fina_df = data["fina_indicator"].copy()
        all_features.update(forward_fill_fundamental(fina_df, calendar_compact))

    # 财务报表
    all_features.update(extract_financial_statement_features(data, calendar_compact))

    # 仅写入所需的原始特征（keep_features 过滤）
    written = []
    for name, arr in all_features.items():
        if keep_features is not None and name not in keep_features:
            continue
        write_bin(feat_dir / f"{name}{BIN_SUFFIX}", start_idx, arr)
        written.append(name)

    logger.info(f"  {symbol}: 全量转换 {len(written)}/{len(all_features)} 个特征写入")
    return written


# ============================================================
# 因子计算 — 基于 tushare/new_factor.md 定义的 24 个因子
# ============================================================
def compute_new_factor_factors(feat_dir, symbol):
    """
    按 new_factor.md 定义的公式，从基础 bin 特征计算衍生因子。
    基础特征 (close/high/low/vwap/volume/pe_ttm/turnover/eps/roe_yearly/netprofit_margin)
    已在 full_convert_csv_to_bin 中写入，本函数读取它们并计算 21 个衍生因子。
    另有 3 个直接因子 (pe_ttm/roe_yearly/netprofit_margin) 已在基础特征中。
    返回: 新增因子名列表
    """
    def read_feat(name):
        path = feat_dir / f"{name}{BIN_SUFFIX}"
        if not path.exists():
            return None
        data = np.fromfile(str(path), dtype="<f4")
        if len(data) < 2:
            return None
        return data[1:].astype(np.float64)

    close = read_feat("close")
    volume = read_feat("volume")
    high = read_feat("high")
    low = read_feat("low")
    vwap = read_feat("vwap")
    turnover = read_feat("turnover")
    eps = read_feat("eps")
    pb = read_feat("pb")

    if close is None:
        logger.warning(f"  {symbol}: 缺少 close 数据，跳过因子计算")
        return []

    n = len(close)

    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b != 0, a / b, np.nan)

    # 日收益率
    ret = np.full(n, np.nan, dtype=np.float64)
    ret[1:] = safe_div(close[1:] - close[:-1], close[:-1])

    new_factors = {}

    # ---- Factor 1: momentum_5d (5日动量) ----
    # MOM_t = close_t / close_{t-5} - 1
    mom5 = np.full(n, np.nan, dtype=np.float64)
    for i in range(5, n):
        if close[i] > 0 and close[i - 5] > 0:
            mom5[i] = close[i] / close[i - 5] - 1.0
    new_factors["momentum_5d"] = mom5

    # ---- Factor 2: momentum_20d (20日动量, MediumTermMomentum_20d) ----
    # M_t = close_t / close_{t-20} - 1
    mom20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(20, n):
        if close[i] > 0 and close[i - 20] > 0:
            mom20[i] = close[i] / close[i - 20] - 1.0
    new_factors["momentum_20d"] = mom20

    # ---- Factor 3: realized_volatility_20d (20日年化波动率) ----
    # sigma_t = std(r_{t-19..t}, ddof=1) * sqrt(252)
    rvol20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        seg = ret[i - 19 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= 5:
            rvol20[i] = np.std(valid, ddof=1) * np.sqrt(252)
    new_factors["realized_volatility_20d"] = rvol20

    # ---- Factor 4: rsi_14d (14日RSI) ----
    # RSI = 100 - 100/(1 + RS), RS = avg_gain / avg_loss
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(14, n):
        seg = ret[i - 13 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 5:
            continue
        gains = np.sum(np.maximum(valid, 0))
        losses = np.sum(np.abs(np.minimum(valid, 0)))
        if losses < 1e-12 and gains < 1e-12:
            continue
        if losses < 1e-12:
            rsi[i] = 100.0
        elif gains < 1e-12:
            rsi[i] = 0.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + gains / losses)
    new_factors["rsi_14d"] = rsi

    # ---- Factor 5: reversal_20d (20日反转) = -momentum_20d ----
    # R_t = -(P_t / P_{t-20} - 1)
    new_factors["reversal_20d"] = -mom20

    # ---- Factor 6: volume_change_5d (5日成交量变化) ----
    # V_t = volume_t / volume_{t-5} - 1
    if volume is not None:
        vol_chg5 = np.full(n, np.nan, dtype=np.float64)
        for i in range(5, n):
            if volume[i - 5] > 0:
                vol_chg5[i] = volume[i] / volume[i - 5] - 1.0
        new_factors["volume_change_5d"] = vol_chg5

    # ---- Factor 7: obv_slope_10d (OBV 10日斜率) ----
    # OBV_t = sum_{i=1..t} V_i * sign(P_i - P_{i-1})
    # slope = linear regression of OBV on day index (0..9)
    if volume is not None:
        obv = np.full(n, np.nan, dtype=np.float64)
        obv[0] = 0.0
        for i in range(1, n):
            prev = obv[i - 1] if not np.isnan(obv[i - 1]) else 0.0
            if np.isnan(volume[i]) or np.isnan(close[i]) or np.isnan(close[i - 1]):
                obv[i] = prev
            elif close[i] > close[i - 1]:
                obv[i] = prev + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = prev - volume[i]
            else:
                obv[i] = prev

        obv_slope = np.full(n, np.nan, dtype=np.float64)
        for i in range(9, n):
            seg = obv[i - 9 : i + 1]
            mask = ~np.isnan(seg)
            if mask.sum() < 5:
                continue
            x = np.arange(10, dtype=np.float64)
            xm, ym = x[mask], seg[mask]
            x_bar, y_bar = xm.mean(), ym.mean()
            num = np.sum((xm - x_bar) * (ym - y_bar))
            den = np.sum((xm - x_bar) ** 2)
            if den > 0:
                obv_slope[i] = num / den
        new_factors["obv_slope_10d"] = obv_slope

    # ---- Factor 8: sharpe_10d (10日Sharpe比率) ----
    # S_t = mean(r_{t-9..t}) / std(r_{t-9..t})
    sharpe10 = np.full(n, np.nan, dtype=np.float64)
    for i in range(9, n):
        seg = ret[i - 9 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= 5:
            mu = np.mean(valid)
            sd = np.std(valid, ddof=1)
            if sd > 0:
                sharpe10[i] = mu / sd
    new_factors["sharpe_10d"] = sharpe10

    # ---- Factor 9: reversal_1d (1日反转) = -日收益率 ----
    # REV_t = -(close_t / close_{t-1} - 1)
    new_factors["reversal_1d"] = -ret

    # ---- Factor 10: volume_ratio_5d (5日量比) ----
    # VR_t = volume_t / avg(volume_{t-1..t-5})
    if volume is not None:
        vr5 = np.full(n, np.nan, dtype=np.float64)
        for i in range(5, n):
            seg = volume[i - 5 : i]  # 前5日 (不含当日)
            valid = seg[~np.isnan(seg) & (seg > 0)]
            if len(valid) >= 3 and not np.isnan(volume[i]) and volume[i] > 0:
                vr5[i] = volume[i] / np.mean(valid)
        new_factors["volume_ratio_5d"] = vr5

    # ---- Factor 11: intraday_volatility (日内振幅) ----
    # IV_t = (high_t - low_t) / close_t
    if high is not None and low is not None:
        iv = safe_div(high - low, close)
        new_factors["intraday_volatility"] = iv

    # ---- Factor 12: volume_weighted_momentum_5d (成交量加权5日动量) ----
    # VWMOM_t = sum(vol_{t-i} * R_{t-i}) / sum(vol_{t-i}), i=1..5
    if volume is not None:
        vwmom5 = np.full(n, np.nan, dtype=np.float64)
        for i in range(5, n):
            vols = volume[i - 5 : i]
            rets = ret[i - 5 : i]
            mask = ~np.isnan(vols) & ~np.isnan(rets) & (vols > 0)
            if mask.sum() >= 3:
                vwmom5[i] = np.sum(vols[mask] * rets[mask]) / np.sum(vols[mask])
        new_factors["volume_weighted_momentum_5d"] = vwmom5

    # ---- Factor 13: earnings_yield (盈利收益率 = E/P) ----
    # EY_t = eps_t / close_t
    if eps is not None:
        ey = safe_div(eps, close)
        new_factors["earnings_yield"] = ey

    # ---- Factor 14: momentum_10d (10日动量) ----
    # MOM_{10,t} = close_t / close_{t-10} - 1
    mom10 = np.full(n, np.nan, dtype=np.float64)
    for i in range(10, n):
        if close[i] > 0 and close[i - 10] > 0:
            mom10[i] = close[i] / close[i - 10] - 1.0
    new_factors["momentum_10d"] = mom10

    # ---- Factor 15: vwap_deviation_10d (10日VWAP偏离) ----
    # VWAP_dev_{10,t} = close_t / avg(vwap_{t..t-9}) - 1
    if vwap is not None:
        vwap_dev10 = np.full(n, np.nan, dtype=np.float64)
        for i in range(9, n):
            seg = vwap[i - 9 : i + 1]
            valid = seg[~np.isnan(seg) & (seg > 0)]
            if len(valid) >= 3 and close[i] > 0:
                vwap_dev10[i] = close[i] / np.mean(valid) - 1.0
        new_factors["vwap_deviation_10d"] = vwap_dev10

    # ---- Factor 16: avg_normalized_range_5d (5日平均归一化振幅) ----
    # ANR_{5,t} = avg( (high-low)/close, 5d )
    if high is not None and low is not None:
        daily_range = safe_div(high - low, close)
        anr5 = np.full(n, np.nan, dtype=np.float64)
        for i in range(4, n):
            seg = daily_range[i - 4 : i + 1]
            valid = seg[~np.isnan(seg)]
            if len(valid) >= 3:
                anr5[i] = np.mean(valid)
        new_factors["avg_normalized_range_5d"] = anr5

    # ---- Factor 17: turnover_trend (换手率趋势) ----
    # TO_trend_t = (avg_TO_5d - avg_TO_20d) / avg_TO_20d
    if turnover is not None:
        to_trend = np.full(n, np.nan, dtype=np.float64)
        for i in range(19, n):
            seg5 = turnover[i - 4 : i + 1]
            seg20 = turnover[i - 19 : i + 1]
            v5 = seg5[~np.isnan(seg5)]
            v20 = seg20[~np.isnan(seg20)]
            if len(v5) >= 3 and len(v20) >= 5:
                avg20 = np.mean(v20)
                if avg20 > 0:
                    to_trend[i] = (np.mean(v5) - avg20) / avg20
        new_factors["turnover_trend"] = to_trend

    # ---- Factor 18: vwap_deviation_5d (5日VWAP偏离) ----
    # VWAP_dev_{5,t} = close_t / avg(vwap_{t..t-4}) - 1
    if vwap is not None:
        vwap_dev5 = np.full(n, np.nan, dtype=np.float64)
        for i in range(4, n):
            seg = vwap[i - 4 : i + 1]
            valid = seg[~np.isnan(seg) & (seg > 0)]
            if len(valid) >= 3 and close[i] > 0:
                vwap_dev5[i] = close[i] / np.mean(valid) - 1.0
        new_factors["vwap_deviation_5d"] = vwap_dev5

    # ---- Factor 19: reversal_2d (2日反转) ----
    # REV_{2,t} = -(close_t / close_{t-2} - 1)
    rev2 = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        if close[i - 2] > 0:
            rev2[i] = -(close[i] / close[i - 2] - 1.0)
    new_factors["reversal_2d"] = rev2

    # ---- Factor 20: volume_ratio_5d_20d (短期/长期成交量比) ----
    # VR_5_20_t = avg_vol_5d / avg_vol_20d
    if volume is not None:
        vr_5_20 = np.full(n, np.nan, dtype=np.float64)
        for i in range(19, n):
            seg5 = volume[i - 4 : i + 1]
            seg20 = volume[i - 19 : i + 1]
            v5 = seg5[~np.isnan(seg5) & (seg5 > 0)]
            v20 = seg20[~np.isnan(seg20) & (seg20 > 0)]
            if len(v5) >= 3 and len(v20) >= 5:
                avg20 = np.mean(v20)
                if avg20 > 0:
                    vr_5_20[i] = np.mean(v5) / avg20
        new_factors["volume_ratio_5d_20d"] = vr_5_20

    # ---- Factor 21: reversal_5d (5日反转) ----
    # REV_{5,t} = -(P_t / P_{t-5} - 1)
    rev5 = np.full(n, np.nan, dtype=np.float64)
    for i in range(5, n):
        if close[i - 5] > 0:
            rev5[i] = -(close[i] / close[i - 5] - 1.0)
    new_factors["reversal_5d"] = rev5

    # ---- Factor 22/25: pb_ratio (PB_Ratio, 市净率) ----
    # PB_t = $pb_t (直接读取 daily_basic 的 pb 字段，无需计算)

    # ---- Factor 23/26: momentum_vol_adjusted_20 (20日波动率调整动量) ----
    # MomVol_t = r_{t,20} / σ_{t,20}
    # r_{t,20} = (P_t - P_{t-20}) / P_{t-20} = momentum_20d
    # σ_{t,20} = std(r, 20d) — 日收益率标准差 (非年化)
    mom_vol_adj20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(20, n):
        if np.isnan(mom20[i]):
            continue
        seg = ret[i - 19 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= 5:
            vol = np.std(valid, ddof=1)
            if vol > 0:
                mom_vol_adj20[i] = mom20[i] / vol
    new_factors["momentum_vol_adjusted_20"] = mom_vol_adj20

    # ---- Factor 24/28: avg_volume_ratio_20d (20日平均成交量比率) ----
    # AVR20_t = V_t / avg(V_{t..t-19})
    if volume is not None:
        avr20 = np.full(n, np.nan, dtype=np.float64)
        for i in range(19, n):
            seg = volume[i - 19 : i + 1]
            valid = seg[~np.isnan(seg) & (seg > 0)]
            if len(valid) >= 5 and not np.isnan(volume[i]) and volume[i] > 0:
                avr20[i] = volume[i] / np.mean(valid)
        new_factors["avg_volume_ratio_20d"] = avr20

    # ---- Factor 25/29: book_to_price (账面市值比) ----
    # BP_t = 1 / $pb_t
    if pb is not None:
        bp = safe_div(np.ones(n, dtype=np.float64), pb)
        new_factors["book_to_price"] = bp

    # ---- Factor 26/30: risk_adjusted_momentum_5d_20d (5日风险调整动量) ----
    # RAMom_{5,20,t} = (close_t/close_{t-5} - 1) / (σ_{20d} * sqrt(252))
    # 分子: momentum_5d, 分母: 年化 20 日已实现波动率
    ramom_5_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(20, n):
        if np.isnan(mom5[i]):
            continue
        seg = ret[i - 19 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= 10:
            vol_ann = np.std(valid, ddof=1) * np.sqrt(252)
            if vol_ann > 0:
                ramom_5_20[i] = mom5[i] / vol_ann
    new_factors["risk_adjusted_momentum_5d_20d"] = ramom_5_20

    # ---- Factor 27: sector_relative_pb (行业相对PB) ----
    # 见 compute_cross_sectional_factors() — 需要所有股票截面数据，后处理计算

    # 以下因子已在 full_convert_csv_to_bin 中作为基础特征写入，无需重复计算:
    #   Factor 22: trailing_PE_ratio → pe_ttm (来自 daily_basic)
    #   Factor 23: roe → roe_yearly (来自 fina_indicator)
    #   Factor 24: net_profit_margin → netprofit_margin (来自 fina_indicator)
    #   Factor 25: PB_Ratio → pb (来自 daily_basic)

    # 读取 start_idx
    any_bin = next(feat_dir.glob(f"*{BIN_SUFFIX}"), None)
    start_idx = int(np.fromfile(str(any_bin), dtype="<f4", count=1)[0]) if any_bin else 0

    # 写入新因子 bin
    for name, arr in new_factors.items():
        write_bin(feat_dir / f"{name}{BIN_SUFFIX}", start_idx, arr.astype(np.float32))
        n_valid = np.sum(~np.isnan(arr))
        if n_valid > 0:
            logger.debug(f"    {name:35s}: {n_valid}/{len(arr)} 非空")

    return list(new_factors.keys())


def _stock_sector(symbol: str) -> str:
    """Map stock code to sector for cross-sectional grouping."""
    code = symbol[2:]
    if symbol.startswith("SH"):
        return "star" if code.startswith("688") else "sh_main"
    elif symbol.startswith("SZ"):
        if code.startswith("30"):
            return "chinext"
        if code.startswith("002") or code.startswith("003"):
            return "sme"
        return "sz_main"
    return "other"


def compute_cross_sectional_factors(feat_root: Path, stock_list: list[str] | None = None) -> list[str]:
    """Compute cross-sectional factors across all stocks at each date.

    Call AFTER all stocks have been individually processed by process_symbol().
    Currently computes: sector_relative_pb

    Returns list of factor names written.
    """
    if stock_list is None:
        stock_dirs = sorted(d for d in feat_root.iterdir() if d.is_dir())
    else:
        stock_dirs = sorted(feat_root / s.lower() for s in stock_list if (feat_root / s.lower()).is_dir())

    if not stock_dirs:
        return []

    # Read pb for all stocks
    stock_pb = {}  # symbol_lower -> np.array
    for d in stock_dirs:
        sym = d.name
        fpath = d / f"pb{BIN_SUFFIX}"
        if not fpath.exists():
            continue
        data = np.fromfile(str(fpath), dtype="<f4")
        if len(data) < 2:
            continue
        stock_pb[sym] = data[1:].astype(np.float64)

    if len(stock_pb) < 2:
        logger.warning("截面因子: 不足 2 只有 pb 数据的股票，跳过 sector_relative_pb")
        return []

    n = len(next(iter(stock_pb.values())))
    sectors = {sym: _stock_sector(sym) for sym in stock_pb}

    for sym, pb_arr in stock_pb.items():
        sector = sectors[sym]
        peers = [s for s, sec in sectors.items() if sec == sector and s != sym]

        rel_pb = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            peer_vals = []
            for p in peers:
                if i < len(stock_pb[p]) and not np.isnan(stock_pb[p][i]):
                    peer_vals.append(stock_pb[p][i])
            if len(peer_vals) >= 2 and not np.isnan(pb_arr[i]):
                rel_pb[i] = pb_arr[i] - np.median(peer_vals)

        any_bin = next((feat_root / sym).glob(f"*{BIN_SUFFIX}"), None)
        if any_bin is None:
            continue
        start_idx = int(np.fromfile(str(any_bin), dtype="<f4", count=1)[0])
        write_bin(feat_root / sym / f"sector_relative_pb{BIN_SUFFIX}", start_idx, rel_pb.astype(np.float32))
        n_valid = np.sum(~np.isnan(rel_pb))
        if n_valid > 0:
            logger.info(f"    sector_relative_pb ({sym}): {n_valid}/{len(rel_pb)} 非空")

    logger.info(f"截面因子 sector_relative_pb 完成 ({len(stock_pb)} 只股票)")
    return ["sector_relative_pb"]



# ============================================================
# 因子分类 (对应 tushare/new_factor.md 的 30 个因子)
# 运行时由 compute_new_factor_factors 和 process_symbol 填充
# ============================================================
FACTOR_CLASSIFICATION = {
    "动量/反转":    [],
    "波动率":       [],
    "震荡":         [],
    "流动性":       [],
    "估值":         ["pe_ttm", "pb"],
    "量价":         [],
    "风险调整":     [],
    "质量":         ["roe_yearly", "netprofit_margin"],
    "基本面/财务":  [],
    "截面":         [],
}


# ============================================================
# 单只股票处理
# ============================================================
def process_symbol(symbol, extra_data_dir, output_dir,
                   global_calendar, force_update=False, skip_bin=False):
    """
    处理单只股票:
      1. 检测 CSV 新鲜度
      2. 如有必要，增量拉取新数据到 extra_data CSV
      3. 从 CSV 全量转换为基础 bin 特征 → cn_extra_data_improve
      4. 按 new_factor.md 公式计算衍生因子
      5. 缩尾极端值
    """
    ts_code = symbol_to_ts_code(symbol)
    today_str = datetime.now().strftime("%Y%m%d")

    result = {"symbol": symbol, "status": "ok"}

    # ---- Step 1: 新鲜度检测 ----
    is_fresh, last_date, detail = check_data_freshness(extra_data_dir, symbol)
    logger.info(f"  [{symbol}] 新鲜度: {'fresh' if is_fresh else 'STALE'}  last={last_date}")

    # ---- Step 2: 增量拉取 ----
    # 日频数据的增量起点: 从 daily.csv 最后日期 + 1 天开始
    # 不能用 max(daily_last, quarterly_last)，因为季频可能比日频新
    if not is_fresh or force_update:
        # 从 detail 中提取日频数据的最后日期
        daily_last_date = detail.get("daily", {}).get("max_date", "20000101")
        fetch_start = (
            "20200101" if daily_last_date == "20000101"
            else (datetime.strptime(daily_last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        )
        logger.info(f"  [{symbol}] 拉取 {fetch_start} ~ {today_str} (日频起点={daily_last_date})")
        api = TushareAPI()
        new_count = fetch_incremental_data(
            api, extra_data_dir, symbol, ts_code, fetch_start, today_str
        )
        logger.info(f"  [{symbol}] 新增 {new_count} 条")
        result["new_rows"] = new_count
    else:
        result["new_rows"] = 0

    # csv-only 模式: 仅更新 CSV，跳过 bin 和因子构建
    if skip_bin:
        return result

    # ---- Step 3: 从 CSV 全量转换为 bin → cn_extra_data_improve ----
    dst_bin_dir = output_dir / "features" / symbol.lower()
    dst_bin_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [{symbol}] CSV → bin (仅 new_factor.md 所需原始字段)...")
    features = full_convert_csv_to_bin(
        symbol, extra_data_dir, output_dir, global_calendar,
        keep_features=NEW_FACTOR_RAW_INPUTS,
    )
    result["base_features"] = len(features)
    logger.info(f"  [{symbol}] 基础特征: {len(features)} 个")

    # ---- Step 4: 按 new_factor.md 公式计算衍生因子 ----
    logger.info(f"  [{symbol}] 计算 new_factor.md 因子...")
    derived = compute_new_factor_factors(dst_bin_dir, symbol)
    result["derived_factors"] = len(derived)
    logger.info(f"  [{symbol}] 衍生因子: {len(derived)} 个")

    # ---- Step 4b: 清理中间特征 (仅保留 new_factor.md 因子 + label 所需的 close) ----
    # close 保留给 label 公式 Ref($close, -2)/Ref($close, -1) - 1 使用
    _INTERMEDIATE_RAW = {"high", "low", "vwap", "volume", "turnover", "eps"}
    for fname in _INTERMEDIATE_RAW:
        fpath = dst_bin_dir / f"{fname}{BIN_SUFFIX}"
        if fpath.exists():
            fpath.unlink()

    # 记录衍生因子到分类 (只记录一次)
    FACTOR_CLASSIFICATION["动量/反转"] = [f for f in derived if f in (
        "momentum_5d", "momentum_10d", "momentum_20d",
        "reversal_1d", "reversal_2d", "reversal_5d", "reversal_20d",
    )]
    FACTOR_CLASSIFICATION["波动率"] = [f for f in derived if f in (
        "realized_volatility_20d", "intraday_volatility", "avg_normalized_range_5d",
    )]
    FACTOR_CLASSIFICATION["震荡"] = [f for f in derived if f in ("rsi_14d",)]
    FACTOR_CLASSIFICATION["流动性"] = [f for f in derived if f in (
        "volume_change_5d", "volume_ratio_5d", "volume_ratio_5d_20d",
        "avg_volume_ratio_20d", "turnover_trend",
    )]
    FACTOR_CLASSIFICATION["量价"] = [f for f in derived if f in (
        "obv_slope_10d", "volume_weighted_momentum_5d",
        "vwap_deviation_5d", "vwap_deviation_10d",
    )]
    FACTOR_CLASSIFICATION["风险调整"] = [f for f in derived if f in (
        "sharpe_10d", "momentum_vol_adjusted_20", "risk_adjusted_momentum_5d_20d",
    )]
    FACTOR_CLASSIFICATION["估值"].extend([f for f in derived if f in (
        "earnings_yield", "book_to_price",
    )])
    FACTOR_CLASSIFICATION["截面"] = [f for f in derived if f in ("sector_relative_pb",)]
    # 基本面/财务: 仅包含 new_factor.md 中的直接因子 (roe_yearly, netprofit_margin 已在质量中)
    FACTOR_CLASSIFICATION["基本面/财务"] = [f for f in features
        if f in ("pe_ttm", "pb", "roe_yearly", "netprofit_margin")]

    # ---- Step 5: 缩尾极端值 (1%/99%) ----
    # 对易出现极端值的基本面因子做缩尾，避免污染截面 ZScoreNorm
    _WINSORIZE_FIELDS = [
        "roe", "roe_yearly", "roa_yearly", "netprofit_margin", "npta",
        "eps_yoy", "netprofit_yoy", "roe_yoy", "bps_yoy", "revenue_yoy",
        "assets_yoy", "debt_to_assets", "ocf_to_profit", "ocf_to_assets",
    ]
    winsorized = 0
    for fname in _WINSORIZE_FIELDS:
        fpath = dst_bin_dir / f"{fname}{BIN_SUFFIX}"
        if not fpath.exists():
            continue
        data = np.fromfile(str(fpath), dtype="<f4")
        if len(data) < 2:
            continue
        header, vals = data[0], data[1:].copy()
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        lo, hi = np.nanquantile(valid, [0.01, 0.99])
        if lo < hi:
            np.concatenate([[header], np.clip(vals, lo, hi).astype("<f4")]).tofile(str(fpath))
            winsorized += 1
    if winsorized > 0:
        logger.debug(f"  [{symbol}] winsorize {winsorized} 个基本面因子")

    return result


# ============================================================
# 辅助函数
# ============================================================
def build_global_calendar(extra_data_dir):
    all_dates = set()
    for csv_path in Path(extra_data_dir).glob("*/daily.csv"):
        try:
            dates = pd.read_csv(csv_path, dtype=str, usecols=["trade_date"])["trade_date"]
            for d in dates.dropna():
                all_dates.add(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
        except Exception:
            pass
    return sorted(all_dates)


def write_instruments(output_dir, extra_data_dir):
    inst_dir = Path(output_dir) / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst_path = inst_dir / "all.txt"

    records = []
    for csv_path in sorted(Path(extra_data_dir).glob("*/daily.csv")):
        symbol = csv_path.parent.name
        try:
            dates = pd.read_csv(csv_path, dtype=str, usecols=["trade_date"])["trade_date"].dropna().sort_values()
            if dates.empty:
                continue
            start = f"{dates.iloc[0][:4]}-{dates.iloc[0][4:6]}-{dates.iloc[0][6:8]}"
            end = f"{dates.iloc[-1][:4]}-{dates.iloc[-1][4:6]}-{dates.iloc[-1][6:8]}"
            records.append((symbol, start, end))
        except Exception:
            pass

    with open(inst_path, "w") as f:
        for sym, sd, ed in sorted(records):
            f.write(f"{sym}\t{sd}\t{ed}\n")
    logger.info(f"写入 instruments: {len(records)} 只 -> {inst_path}")

    # 复制指数成分股文件 (csi300/500/800/1000/csiall)
    cn_data_inst = Path(output_dir).parent / "cn_data" / "instruments"
    if cn_data_inst.exists():
        for idx_file in cn_data_inst.glob("csi*.txt"):
            dest = inst_dir / idx_file.name
            if not dest.exists():
                shutil.copy2(str(idx_file), str(dest))
        logger.info(f"复制指数成分股文件 -> {inst_dir}")


def write_factor_manifest(output_dir, results):
    manifest = {
        "description": "cn_extra_data_improve 因子特征清单",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks_processed": len(results),
        "factor_categories": {},
    }
    for cat, factors in FACTOR_CLASSIFICATION.items():
        manifest["factor_categories"][cat] = {
            "count": len(factors),
            "factors": factors,
        }
    if not FACTOR_CLASSIFICATION.get("量价"):
        manifest["factor_categories"]["量价"] = {"count": 0, "factors": [], "description": "OBV斜率、成交量加权动量等"}

    manifest_path = Path(output_dir) / "factor_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"因子清单: {manifest_path}")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="process_extra_data - 数据检测、增量更新与因子构建")
    parser.add_argument("--test", action="store_true", help="用 5 支股票验证")
    parser.add_argument("--symbols", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--force", action="store_true", help="强制拉取最新数据")
    parser.add_argument("--csv-only", action="store_true", help="仅更新 CSV，跳过 bin 和因子")
    parser.add_argument("--mode", choices=["full", "improve", "improve-stock"], default="full",
                        help="full=完整流程, improve=全量构建improve, improve-stock=单只股票improve")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    extra_data_dir = script_dir / "extra_data"
    output_dir = script_dir / "cn_extra_data_improve"

    # ---- improve 模式: 从 extra_data CSV 构建 cn_extra_data_improve ----
    if args.mode == "improve":
        # 确定股票列表 (从 extra_data 目录获取)
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            symbols = sorted([
                d.name.upper() for d in extra_data_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".") and (d / "daily.csv").exists()
            ])

        # 构建/复制日历
        cal_src = output_dir / "calendars" / "day.txt"
        if cal_src.exists():
            with open(cal_src) as f:
                calendar = [line.strip() for line in f if line.strip()]
            logger.info(f"日历: {len(calendar)} 天")
        else:
            calendar = build_global_calendar(extra_data_dir)
            if calendar:
                cal_dst = output_dir / "calendars"
                cal_dst.mkdir(parents=True, exist_ok=True)
                with open(cal_dst / "day.txt", "w") as f:
                    for d in calendar:
                        f.write(d + "\n")

        logger.info(f"improve 模式: {len(symbols)} 只股票, 输出 {output_dir}")

        results = []
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] {symbol}")
            try:
                r = process_symbol(
                    symbol, extra_data_dir, output_dir,
                    calendar, force_update=False,
                )
                results.append(r)
            except Exception as e:
                logger.error(f"[{symbol}] 失败: {e}", exc_info=True)
                results.append({"symbol": symbol, "status": "error", "error": str(e)})

        success = sum(1 for r in results if r.get("status") == "ok")
        logger.info(f"improve 完成: {success}/{len(symbols)} 成功")

        write_instruments(output_dir, extra_data_dir)
        write_factor_manifest(output_dir, results)

        stock_dirs = list(Path(output_dir).glob("features/*"))
        if stock_dirs:
            sample = stock_dirs[0]
            n_bins = len(list(sample.glob("*.bin")))
            logger.info(f"每只股票: {n_bins} 特征 (new_factor.md 因子 + 基础行情/基本面)")
        logger.info(f"输出目录: {output_dir}")
        return

    # ---- improve-stock 模式: 单只股票 (并行调用，轻量) ----
    if args.mode == "improve-stock":
        if not args.symbols or len(args.symbols) != 1:
            logger.error("improve-stock 模式需要一个 --symbols 参数 (单只股票)")
            sys.exit(1)
        symbol = args.symbols[0].upper()

        # 读日历
        cal_src = output_dir / "calendars" / "day.txt"
        if cal_src.exists():
            with open(cal_src) as f:
                calendar = [line.strip() for line in f if line.strip()]
        else:
            calendar = build_global_calendar(extra_data_dir)

        r = process_symbol(
            symbol, extra_data_dir, output_dir,
            calendar, force_update=False,
        )
        if r.get("status") == "ok":
            n_base = r.get("base_features", 0)
            n_derived = r.get("derived_factors", 0)
            logger.info(f"improve-stock {symbol}: OK ({n_base} 基础 + {n_derived} 衍生因子)")
        else:
            logger.error(f"improve-stock {symbol}: {r.get('error', 'FAIL')}")
            sys.exit(1)
        return

    # ---- full 模式 (原有逻辑) ----
    # 确定股票列表
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.test:
        symbols = TEST_SYMBOLS
    else:
        symbols = sorted([
            d.name for d in extra_data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    logger.info("=" * 60)
    logger.info(f"股票数量: {len(symbols)}  模式: {'test' if args.test else 'full'}"
                f"  force={args.force}  csv_only={args.csv_only}")
    logger.info(f"输出目录:     {output_dir}")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建全局日历
    logger.info("构建交易日历...")
    calendar = build_global_calendar(extra_data_dir)
    if calendar:
        cal_dir = output_dir / "calendars"
        cal_dir.mkdir(parents=True, exist_ok=True)
        with open(cal_dir / "day.txt", "w") as f:
            for d in calendar:
                f.write(d + "\n")
        logger.info(f"日历: {len(calendar)} 天 ({calendar[0]} ~ {calendar[-1]})")

    # 逐只处理
    results = []
    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] {symbol}")
        try:
            r = process_symbol(
                symbol, extra_data_dir, output_dir,
                calendar, force_update=args.force,
                skip_bin=args.csv_only,
            )
            results.append(r)
        except Exception as e:
            logger.error(f"[{symbol}] 失败: {e}", exc_info=True)
            results.append({"symbol": symbol, "status": "error", "error": str(e)})

    success = sum(1 for r in results if r.get("status") == "ok")
    logger.info(f"处理完成: {success}/{len(symbols)} 成功")

    if not args.csv_only:
        # 计算截面因子 (需要所有股票数据)
        logger.info("计算截面因子 (sector_relative_pb)...")
        feat_root = Path(output_dir) / "features"
        try:
            cs_results = compute_cross_sectional_factors(feat_root)
            logger.info(f"截面因子完成: {cs_results}")
        except Exception as e:
            logger.error(f"截面因子失败: {e}", exc_info=True)

        write_instruments(output_dir, extra_data_dir)
        write_factor_manifest(output_dir, results)

        # 统计
        stock_dirs = list(Path(output_dir).glob("features/*"))
        sample = stock_dirs[0] if stock_dirs else None
        if sample:
            n_bins = len(list(sample.glob("*.bin")))
            logger.info(f"每只股票: {n_bins} 特征 (基础特征 + new_factor.md 衍生因子)")
            logger.info("")
            logger.info("因子分类:")
            for cat, factors in FACTOR_CLASSIFICATION.items():
                if factors:
                    logger.info(f"  {cat}: {factors}")

    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
