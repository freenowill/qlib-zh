"""
explore_extra_data.py - 将 extra_data (Tushare CSV) 转换为 qlib bin 格式

输入:  extra_data/{SYMBOL}/*.csv  (test_tushare.py 产出)
输出:  cn_extra_data/             (qlib 可直接加载的标准格式)

与 cn_data 的区别:
  - cn_data 仅含 10 个基础行情特征
  - cn_extra_data 额外包含 15+ 个基本面/估值特征，可直接用于多因子模型训练

运行方式:
  docker run --rm -v $(pwd):/workspace -w /workspace zhuhai123/local_qlib:v1-tushare \
    python3 explore_extra_data.py SZ000002
"""

import argparse
import os
import sys
import time
import struct
import logging
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
TUSHARE_TOKEN = "4cbb80cf41ae83b53f9bc431a502c328565e53938bce7cadce52bc2a"
RATE_LIMIT = 0.3


def symbol_to_ts_code(symbol):
    """SZ000001 -> 000001.SZ, SH600000 -> 600000.SH"""
    prefix = symbol[:2].upper()
    code = symbol[2:]
    return f"{code}.{prefix}"

BIN_SUFFIX = ".day.bin"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 读取 CSV 数据
# ============================================================
def load_csvs(input_dir):
    """读取 extra_data 下所有 CSV"""
    data = {}
    for name in ["daily", "daily_basic", "fina_indicator", "income",
                  "balancesheet", "cashflow", "dividend", "stock_company"]:
        path = input_dir / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path, dtype=str)
            # 数值列转 float
            for c in data[name].columns:
                if c not in ("ts_code", "trade_date", "ann_date", "f_ann_date",
                             "end_date", "report_type", "comp_type", "end_type",
                             "div_proc", "record_date", "ex_date", "pay_date",
                             "div_listdate", "imp_ann_date", "setup_date",
                             "province", "city", "website", "email", "office",
                             "introduction", "main_business", "business_scope",
                             "chairman", "manager", "secretary", "exchange"):
                    data[name][c] = pd.to_numeric(data[name][c], errors="coerce")
            logger.info(f"读取 {name}.csv: {len(data[name])} 行")
    return data


# ============================================================
# 2. 获取 adj_factor (复权因子)
# ============================================================
def fetch_adj_factor(ts_code, start_date, end_date):
    """从 Tushare API 获取复权因子"""
    sx.set_token(TUSHARE_TOKEN)
    api = sx.get_api(env="prd")

    all_dfs = []
    # 按月分批拉取，避免单次请求过大
    dates = pd.date_range(start_date, end_date, freq="MS")
    for d in dates:
        sd = d.strftime("%Y%m%d")
        ed = (d + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")
        try:
            time.sleep(RATE_LIMIT)
            df = api.query("adj_factor", ts_code=ts_code,
                           start_date=sd, end_date=ed)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"adj_factor {sd}~{ed} 失败: {e}")

    if not all_dfs:
        logger.error("未能获取 adj_factor 数据!")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
    logger.info(f"获取 adj_factor: {len(result)} 条")
    return result


# ============================================================
# 3. 构建日历与对齐
# ============================================================
def build_calendar(daily_df):
    """从 daily.csv 构建交易日历"""
    dates = sorted(daily_df["trade_date"].unique())
    # 转为 YYYY-MM-DD 格式
    cal = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates]
    return cal


def date_to_idx(cal_compact, date_str):
    """日期字符串 -> 日历索引 (YYYYMMDD)"""
    try:
        return cal_compact.index(date_str)
    except ValueError:
        return None


def align_series_to_calendar(series, dates_col, calendar_compact):
    """
    将带日期的 Series 对齐到日历。
    series: pd.Series, index 为原始行号
    dates_col: 对应的日期列 (YYYYMMDD)
    calendar_compact: list of YYYYMMDD
    返回: np.ndarray, 长度 = len(calendar), 缺失为 NaN
    """
    result = np.full(len(calendar_compact), np.nan, dtype=np.float32)
    date_to_val = dict(zip(dates_col, series.values))
    for i, d in enumerate(calendar_compact):
        if d in date_to_val:
            v = date_to_val[d]
            if pd.notna(v):
                result[i] = np.float32(v)
    return result


# ============================================================
# 4. 归一化 (参考 update_data.py)
# ============================================================
def normalize_market_data(daily_df, adj_factor_df, calendar_compact):
    """
    将 daily.csv + adj_factor 转换为 qlib 标准特征。

    返回 dict: feature_name -> np.ndarray (float32, 长度 = 日历长度)
    """
    # 合并 daily 和 adj_factor
    merged = daily_df[["trade_date", "open", "high", "low", "close",
                        "vol", "amount"]].copy()
    merged = merged.sort_values("trade_date").reset_index(drop=True)

    if not adj_factor_df.empty:
        af = adj_factor_df[["trade_date", "adj_factor"]].copy()
        merged = merged.merge(af, on="trade_date", how="left")
        # 前向填充缺失的 adj_factor
        merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
    else:
        merged["adj_factor"] = 1.0

    # 计算 adj_close
    merged["adj_close"] = merged["close"].astype(float) * merged["adj_factor"].astype(float)

    # base_price = 第一个交易日的 adj_close
    base_price = merged["adj_close"].iloc[0]
    if pd.isna(base_price) or base_price <= 0:
        base_price = merged["close"].iloc[0]
    logger.info(f"base_price = {base_price:.4f}")

    # 归一化
    adj_factor = merged["adj_factor"].values.astype(np.float64)
    close_raw = merged["close"].values.astype(np.float64)
    open_raw = merged["open"].values.astype(np.float64)
    high_raw = merged["high"].values.astype(np.float64)
    low_raw = merged["low"].values.astype(np.float64)
    vol_raw = merged["vol"].values.astype(np.float64)  # 手 (100股)
    amount_raw = merged["amount"].values.astype(np.float64)  # 千元
    adj_close = merged["adj_close"].values.astype(np.float64)

    close_norm = adj_close / base_price
    open_norm = open_raw * adj_factor / base_price
    high_norm = high_raw * adj_factor / base_price
    low_norm = low_raw * adj_factor / base_price

    # vwap = amount / volume * 10 (amount千元, vol手 -> 元/股)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_vwap = np.where(vol_raw > 0, amount_raw / vol_raw * 10, np.nan)
    # 用首日原始收盘价归一化，与 close (adjclose/base_price) 同量纲
    close_raw_day1 = close_raw[0]
    if pd.isna(close_raw_day1) or close_raw_day1 <= 0:
        close_raw_day1 = base_price
    vwap_norm = raw_vwap / close_raw_day1

    volume_raw = vol_raw  # 保留原始手数，不金额化
    factor_norm = adj_factor / base_price

    # change = daily return
    change = np.full(len(close_norm), np.nan, dtype=np.float64)
    if len(close_norm) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            change[1:] = np.where(
                close_norm[:-1] != 0,
                (close_norm[1:] - close_norm[:-1]) / close_norm[:-1],
                np.nan
            )

    features = {
        "open": open_norm.astype(np.float32),
        "close": close_norm.astype(np.float32),
        "high": high_norm.astype(np.float32),
        "low": low_norm.astype(np.float32),
        "vwap": vwap_norm.astype(np.float32),
        "volume": volume_raw.astype(np.float32),   # 原始手数
        "amount": amount_raw.astype(np.float32),    # 原始千元
        "adjclose": adj_close.astype(np.float32),
        "change": change.astype(np.float32),
        "factor": factor_norm.astype(np.float32),
    }

    # 对齐到日历
    aligned = {}
    for name, vals in features.items():
        aligned[name] = align_series_to_calendar(
            pd.Series(vals), merged["trade_date"].values, calendar_compact
        )

    return aligned


# ============================================================
# 5. 额外特征 (daily_basic)
# ============================================================
EXTRA_DAILY_FEATURES = {
    # 估值因子
    "pe":         "pe",
    "pe_ttm":     "pe_ttm",
    "pb":         "pb",
    "ps":         "ps",
    "ps_ttm":     "ps_ttm",
    # 红利因子
    "dv_ratio":   "dv_ratio",
    "dv_ttm":     "dv_ttm",
    # 流动性因子
    "turnover":   "turnover_rate",
    "turnover_f": "turnover_rate_f",
    "vol_ratio":  "volume_ratio",
    # 规模因子
    "total_mv":   "total_mv",
    "circ_mv":    "circ_mv",
    # 股本
    "total_sh":   "total_share",
    "float_sh":   "float_share",
    "free_sh":    "free_share",
}


def extract_daily_basic_features(daily_basic_df, calendar_compact):
    """从 daily_basic.csv 提取额外日频特征"""
    features = {}
    for feat_name, col_name in EXTRA_DAILY_FEATURES.items():
        if col_name in daily_basic_df.columns:
            features[feat_name] = align_series_to_calendar(
                daily_basic_df[col_name],
                daily_basic_df["trade_date"].values,
                calendar_compact
            )
    # dv_ttm 缺失时用 dv_ratio 回填
    if "dv_ttm" in features and "dv_ratio" in features:
        mask = np.isnan(features["dv_ttm"]) & ~np.isnan(features["dv_ratio"])
        if mask.any():
            features["dv_ttm"][mask] = features["dv_ratio"][mask]
            logger.info(f"  dv_ttm: 用 dv_ratio 回填 {mask.sum()} 个缺失值")
    return features


# ============================================================
# 6. 财务指标 (fina_indicator -> 前向填充到日频)
# ============================================================
# 选取覆盖率高、量化价值大的指标
FUNDAMENTAL_FEATURES = {
    # 每股指标
    "eps":              "eps",              # 每股收益
    "dt_eps":           "dt_eps",           # 扣非每股收益
    "bps":              "bps",              # 每股净资产
    "ocfps":            "ocfps",            # 每股经营现金流
    "cfps":             "cfps",             # 每股现金流
    "revenue_ps":       "revenue_ps",       # 每股营收
    "undist_ps":        "undist_profit_ps", # 每股未分配利润
    # 盈利能力
    "roe":              "roe",              # ROE
    "roe_yearly":       "roe_yearly",       # 年化 ROE
    "roa_yearly":       "roa_yearly",       # 年化 ROA
    "npta":             "npta",             # 净利润/总资产
    "netprofit_margin": "netprofit_margin", # 销售净利率
    # 杠杆
    "debt_to_assets":   "debt_to_assets",   # 资产负债率
    "assets_to_eqt":    "assets_to_eqt",    # 权益乘数
    # 成长
    "eps_yoy":          "basic_eps_yoy",    # EPS 同比
    "netprofit_yoy":    "netprofit_yoy",    # 净利润同比
    "roe_yoy":          "roe_yoy",          # ROE 同比
    "bps_yoy":          "bps_yoy",          # BPS 同比
    "assets_yoy":       "assets_yoy",       # 总资产同比
    "revenue_yoy":      "or_yoy",           # 营收同比
}


def forward_fill_fundamental(fina_df, calendar_compact):
    """
    将季报财务指标前向填充到日频。

    关键: 使用 ann_date (公告日) 而非 end_date (报告期) 作为数据可用时间，
    避免未来信息泄漏。
    """
    features = {}

    for feat_name, col_name in FUNDAMENTAL_FEATURES.items():
        if col_name not in fina_df.columns:
            continue

        # 按公告日排序
        df = fina_df[["ann_date", col_name]].dropna(subset=[col_name, "ann_date"])
        if df.empty:
            continue

        df = df.sort_values("ann_date").drop_duplicates(subset=["ann_date"], keep="last")

        # 对齐到日历: 公告日之后的交易日都使用该值
        result = np.full(len(calendar_compact), np.nan, dtype=np.float32)

        # 构建 ann_date -> value 的映射
        ann_dates = df["ann_date"].astype(str).values
        values = df[col_name].values.astype(np.float32)

        # 逐日填充: 找到当前公告日前最近的有数据的公告日
        current_val = np.nan
        ann_idx = 0
        for i, cal_date in enumerate(calendar_compact):
            cal_comp = cal_date.replace("-", "")
            # 检查是否有新公告
            while ann_idx < len(ann_dates) and ann_dates[ann_idx] <= cal_comp:
                current_val = values[ann_idx]
                ann_idx += 1
            result[i] = current_val

        non_nan = np.sum(~np.isnan(result))
        if non_nan > 0:
            features[feat_name] = result
            logger.info(f"  {feat_name}: {non_nan}/{len(result)} 非空")

    return features


# ============================================================
# 6b. 从 income/balancesheet/cashflow 提取特征
# ============================================================
def extract_financial_statement_features(data, calendar_compact):
    """
    从 income/balancesheet/cashflow 提取特征并前向填充到日频。
    使用 ann_date 避免未来信息泄漏。
    """
    source_features = {
        "income": {
            "revenue_yoy_ts":   "total_revenue",    # 营业总收入 (绝对值)
            "n_income":         "n_income",          # 净利润
            "operate_profit":   "operate_profit",    # 营业利润
        },
        "balancesheet": {
            "total_assets":     "total_assets",      # 总资产
            "total_liab":       "total_liab",        # 总负债
            "total_equity":     "total_hldr_eqy_exc_min_int",  # 归母股东权益
        },
        "cashflow": {
            "ocf":              "n_cashflow_act",    # 经营现金流净额
            "icf":              "n_cashflow_inv_act", # 投资现金流净额
            "fcf":              "free_cashflow",      # 自由现金流
        },
    }

    # 合并所有可用的财务报表
    combined_dfs = []
    for src_name, feat_map in source_features.items():
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

    # 合并所有报表 (按 ann_date + 顺序合并)
    merged = combined_dfs[0]
    for df in combined_dfs[1:]:
        merged = merged.merge(df, on="ann_date", how="outer")

    # 计算衍生比率特征
    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b != 0, a / b, np.nan)

    if "total_assets" in merged.columns and "total_liab" in merged.columns:
        merged["liab_to_eqty"] = safe_div(
            merged["total_liab"].values,
            (merged["total_assets"].values - merged["total_liab"].values)
        )

    if "operate_profit" in merged.columns and "total_revenue" in merged.columns:
        merged["op_to_revenue"] = safe_div(
            merged["operate_profit"].values,
            merged["total_revenue"].values
        ) * 100

    if "n_cashflow_act" in merged.columns and "n_income" in merged.columns:
        merged["ocf_to_profit"] = safe_div(
            merged["n_cashflow_act"].values,
            merged["n_income"].values
        )

    if "n_cashflow_act" in merged.columns and "total_assets" in merged.columns:
        merged["ocf_to_assets"] = safe_div(
            merged["n_cashflow_act"].values,
            merged["total_assets"].values
        ) * 100

    # 构建最终特征映射
    final_features = {
        # 利润表
        "revenue":          "total_revenue",
        "n_income":         "n_income",
        "operate_profit":   "operate_profit",
        # 资产负债表
        "total_assets":     "total_assets",
        "total_liab":       "total_liab",
        "total_equity":     "total_hldr_eqy_exc_min_int",
        # 现金流量表
        "ocf":              "n_cashflow_act",
        "icf":              "n_cashflow_inv_act",
        "fcf":              "free_cashflow",
        # 衍生比率
        "liab_to_eqty":     "liab_to_eqty",
        "op_to_revenue":    "op_to_revenue",
        "ocf_to_profit":    "ocf_to_profit",
        "ocf_to_assets":    "ocf_to_assets",
    }

    features = {}
    for feat_name, col_name in final_features.items():
        if col_name not in merged.columns:
            continue

        df = merged[["ann_date", col_name]].dropna(subset=[col_name, "ann_date"])
        if df.empty:
            continue

        df = df.sort_values("ann_date").drop_duplicates(
            subset=["ann_date"], keep="last"
        )

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

        non_nan = np.sum(~np.isnan(result))
        if non_nan > 0:
            features[feat_name] = result
            logger.info(f"  {feat_name}: {non_nan}/{len(result)} 非空")

    return features


# ============================================================
# 7. 写入 qlib bin 格式
# ============================================================
def write_bin(filepath, start_idx, data):
    """写入 qlib bin 文件"""
    header = np.array([float(start_idx)], dtype="<f4")
    np.concatenate([header, data.astype("<f4")]).tofile(str(filepath))


def write_calendar(path, dates):
    """写入日历文件"""
    with open(path, "w") as f:
        for d in dates:
            f.write(d + "\n")


def write_instruments(path, symbol, start_date, end_date):
    """写入/更新股票列表 (支持多股票追加)"""
    entries = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    entries[parts[0]] = (parts[1], parts[2])
    entries[symbol] = (start_date, end_date)
    with open(path, "w") as f:
        for sym, (sd, ed) in sorted(entries.items()):
            f.write(f"{sym}\t{sd}\t{ed}\n")


# ============================================================
# 主流程
# ============================================================
def load_global_calendar(calendar_path):
    """加载全局日历文件 (YYYY-MM-DD 格式，每行一个日期)"""
    with open(calendar_path) as f:
        dates = [line.strip() for line in f if line.strip()]
    return dates


def main():
    parser = argparse.ArgumentParser(description="extra_data -> cn_extra_data 转换")
    parser.add_argument("symbol", nargs="?", default="SZ000001",
                        help="股票代码，如 SZ000001、SH600000 (默认 SZ000001)")
    parser.add_argument("--calendar", default=None,
                        help="全局日历文件路径 (YYYY-MM-DD 格式)。"
                             "提供时数据对齐到全局日历，否则使用股票自身日历")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    ts_code = symbol_to_ts_code(symbol)
    input_dir = Path(__file__).parent / "extra_data" / symbol
    output_dir = Path(__file__).parent / "cn_extra_data"

    logger.info("=" * 60)
    logger.info(f"extra_data -> cn_extra_data 转换: {symbol} ({ts_code})")
    logger.info(f"输入: {input_dir}")
    logger.info(f"输出: {output_dir}")
    logger.info("=" * 60)

    # 1. 读取 CSV
    data = load_csvs(input_dir)
    if "daily" not in data:
        logger.error(f"缺少 daily.csv，跳过 {symbol}")
        sys.exit(1)
    daily_df = data["daily"].copy()
    daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)

    # 2. 获取 adj_factor
    start_date = daily_df["trade_date"].min()
    end_date = daily_df["trade_date"].max()
    logger.info(f"数据日期范围: {start_date} ~ {end_date}")

    adj_factor_df = fetch_adj_factor(ts_code, start_date, end_date)

    # 3. 构建日历
    if args.calendar and Path(args.calendar).exists():
        # 使用全局日历，计算 start_idx
        global_calendar = load_global_calendar(args.calendar)
        global_compact = [d.replace("-", "") for d in global_calendar]
        stock_calendar = build_calendar(daily_df)
        stock_compact = [d.replace("-", "") for d in stock_calendar]

        # 找到股票第一个交易日在全局日历中的位置
        first_date = stock_compact[0]
        if first_date in global_compact:
            start_idx = global_compact.index(first_date)
        else:
            logger.warning(f"股票首日 {first_date} 不在全局日历中，使用 start_idx=0")
            start_idx = 0

        calendar = global_calendar
        calendar_compact = global_compact
        logger.info(f"全局日历: {len(calendar)} 天, start_idx={start_idx}")
    else:
        # 使用股票自身日历 (向后兼容)
        calendar = build_calendar(daily_df)
        calendar_compact = [d.replace("-", "") for d in calendar]
        start_idx = 0
        logger.info(f"股票日历: {len(calendar)} 天 ({calendar[0]} ~ {calendar[-1]})")

    # 4. 创建输出目录 (日历和股票列表由 run_explore_data.sh 统一构建)
    feat_dir = output_dir / "features" / symbol.lower()
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 5. 生成标准行情特征
    logger.info("-" * 40)
    logger.info("生成标准行情特征 (10个)")
    market_features = normalize_market_data(daily_df, adj_factor_df, calendar_compact)

    for name, arr in market_features.items():
        write_bin(feat_dir / f"{name}{BIN_SUFFIX}", start_idx, arr)
        non_nan = np.sum(~np.isnan(arr))
        logger.info(f"  {name:10s}: {non_nan}/{len(arr)} 非空, "
                    f"range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")

    # 8. 生成日频额外特征
    if "daily_basic" in data:
        logger.info("-" * 40)
        logger.info("生成日频估值特征 ({}个)".format(len(EXTRA_DAILY_FEATURES)))
        daily_basic_df = data["daily_basic"].copy()
        daily_basic_df = daily_basic_df.sort_values("trade_date").reset_index(drop=True)

        extra_daily = extract_daily_basic_features(daily_basic_df, calendar_compact)
        for name, arr in extra_daily.items():
            write_bin(feat_dir / f"{name}{BIN_SUFFIX}", start_idx, arr)
            non_nan = np.sum(~np.isnan(arr))
            logger.info(f"  {name:15s}: {non_nan}/{len(arr)} 非空")
    else:
        logger.warning("缺少 daily_basic.csv，跳过估值特征")

    # 9. 生成基本面特征 (季报前向填充)
    if "fina_indicator" in data:
        logger.info("-" * 40)
        logger.info("生成基本面特征 (fina_indicator -> 日频)")
        fina_df = data["fina_indicator"].copy()
        fund_features = forward_fill_fundamental(fina_df, calendar_compact)
        for name, arr in fund_features.items():
            write_bin(feat_dir / f"{name}{BIN_SUFFIX}", start_idx, arr)

    # 9b. 从 income/balancesheet/cashflow 提取特征
    stmt_sources = [s for s in ["income", "balancesheet", "cashflow"] if s in data]
    if stmt_sources:
        logger.info("-" * 40)
        logger.info(f"生成财报特征 ({' + '.join(stmt_sources)} -> 日频)")
        stmt_features = extract_financial_statement_features(data, calendar_compact)
        for name, arr in stmt_features.items():
            write_bin(feat_dir / f"{name}{BIN_SUFFIX}", start_idx, arr)

    # 10. 汇总
    logger.info("=" * 60)
    logger.info("转换完成! 输出目录: cn_extra_data/")
    logger.info("-" * 40)
    all_bins = sorted(feat_dir.glob(f"*{BIN_SUFFIX}"))
    logger.info(f"特征总数: {len(all_bins)}")
    logger.info("")
    logger.info("标准行情特征 (10个):")
    for f in [x for x in all_bins if x.stem.replace(".day", "") in
              ["open", "close", "high", "low", "vwap", "volume",
               "amount", "adjclose", "change", "factor"]]:
        size = f.stat().st_size
        logger.info(f"  {f.name:30s}  {size:>8d} bytes")
    logger.info("")
    logger.info("额外特征 ({}个):".format(
        len(all_bins) - 10))
    for f in all_bins:
        feat = f.stem.replace(".day", "")
        if feat not in ["open", "close", "high", "low", "vwap", "volume",
                        "amount", "adjclose", "change", "factor"]:
            size = f.stat().st_size
            logger.info(f"  {f.name:30s}  {size:>8d} bytes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
