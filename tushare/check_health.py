"""
check_health.py - 检查 extra_data 数据是否健康、可用

用法:
  python3 check_health.py                  # 检查所有 extra_data/{SYMBOL}/ 目录
  python3 check_health.py SZ000001         # 只检查指定股票

检查项:
  1. 必需文件是否存在
  2. CSV 列名是否完整
  3. 数据行数是否合理
  4. 日期覆盖范围与一致性
  5. 数值列的缺失率与异常值
  6. 跨文件日期对齐
  7. 财报公告日 vs 报告期逻辑
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXTRA_DATA_DIR = Path(__file__).parent / "extra_data"

# ============================================================
# 期望的列定义
# ============================================================
REQUIRED_FILES = {
    "daily.csv": {
        "must_have": ["ts_code", "trade_date", "open", "high", "low", "close",
                       "vol", "amount"],
        "description": "日线行情",
    },
    "daily_basic.csv": {
        "must_have": ["ts_code", "trade_date", "pe", "pe_ttm", "pb",
                       "turnover_rate", "total_mv", "circ_mv"],
        "description": "每日指标",
    },
}

OPTIONAL_FILES = {
    "fina_indicator.csv": {
        "must_have": ["ts_code", "ann_date", "end_date", "eps", "roe",
                       "debt_to_assets"],
        "description": "财务指标 (季报)",
    },
    "income.csv": {
        "must_have": ["ts_code", "ann_date", "end_date", "total_revenue",
                       "n_income"],
        "description": "利润表",
    },
    "balancesheet.csv": {
        "must_have": ["ts_code", "ann_date", "end_date", "total_assets",
                       "total_liab"],
        "description": "资产负债表",
    },
    "cashflow.csv": {
        "must_have": ["ts_code", "ann_date", "end_date"],
        "description": "现金流量表",
    },
    "dividend.csv": {
        "must_have": ["ts_code", "end_date"],
        "description": "分红送股",
    },
}

# 数值列的合理范围 (min, max)，用于异常值检测
VALUE_RANGES = {
    "pe":           (-1000, 10000),
    "pe_ttm":       (-1000, 10000),
    "pb":           (-100, 1000),
    "turnover_rate": (0, 100),
    "roe":          (-100, 200),
    "eps":          (-50, 100),
    "debt_to_assets": (0, 200),
}


# ============================================================
# 检查函数
# ============================================================
class HealthReport:
    """汇总检查结果"""

    def __init__(self, symbol):
        self.symbol = symbol
        self.errors = []
        self.warnings = []
        self.info = []

    def error(self, msg):
        self.errors.append(msg)
        logger.error(f"  [ERROR] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        logger.warning(f"  [WARN]  {msg}")

    def ok(self, msg):
        self.info.append(msg)
        logger.info(f"  [OK]    {msg}")

    @property
    def healthy(self):
        return len(self.errors) == 0

    def summary(self):
        lines = [
            "",
            "=" * 60,
            f"健康检查报告: {self.symbol}",
            "=" * 60,
            f"  错误: {len(self.errors)}",
            f"  警告: {len(self.warnings)}",
            f"  状态: {'健康' if self.healthy else '异常'}",
        ]
        if self.errors:
            lines.append("")
            lines.append("错误详情:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("")
            lines.append("警告详情:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


def check_files_exist(data_dir, report):
    """检查文件是否存在"""
    logger.info(f"检查文件存在性: {data_dir.name}")
    found = {}
    for fname, meta in REQUIRED_FILES.items():
        path = data_dir / fname
        if path.exists():
            found[fname] = pd.read_csv(path, dtype=str)
            report.ok(f"{fname} 存在 ({meta['description']})")
        else:
            report.error(f"必需文件缺失: {fname} ({meta['description']})")

    for fname, meta in OPTIONAL_FILES.items():
        path = data_dir / fname
        if path.exists():
            found[fname] = pd.read_csv(path, dtype=str)
            report.ok(f"{fname} 存在 ({meta['description']})")
        else:
            report.warn(f"可选文件缺失: {fname} ({meta['description']})")

    return found


def check_columns(dfs, report):
    """检查 CSV 列名完整性"""
    logger.info("检查列名完整性")
    all_specs = {**REQUIRED_FILES, **OPTIONAL_FILES}
    for fname, df in dfs.items():
        if fname not in all_specs:
            continue
        spec = all_specs[fname]
        missing = [c for c in spec["must_have"] if c not in df.columns]
        if missing:
            report.error(f"{fname} 缺少必需列: {missing}")
        else:
            report.ok(f"{fname} 列名完整 ({len(df.columns)} 列)")


def check_row_counts(dfs, report):
    """检查行数是否合理"""
    logger.info("检查数据行数")
    for fname, df in dfs.items():
        n = len(df)
        if n == 0:
            report.error(f"{fname} 为空 (0 行)")
        elif n < 10:
            report.warn(f"{fname} 仅 {n} 行，数据可能不完整")
        else:
            report.ok(f"{fname} 有 {n} 行数据")


def check_date_coverage(dfs, report):
    """检查日期覆盖范围"""
    logger.info("检查日期覆盖范围")
    date_ranges = {}

    if "daily.csv" in dfs:
        df = dfs["daily.csv"]
        dates = sorted(df["trade_date"].unique())
        date_ranges["daily"] = (dates[0], dates[-1], len(dates))
        report.ok(f"daily 日期: {dates[0]} ~ {dates[-1]} ({len(dates)} 交易日)")

    if "daily_basic.csv" in dfs:
        df = dfs["daily_basic.csv"]
        dates = sorted(df["trade_date"].unique())
        date_ranges["daily_basic"] = (dates[0], dates[-1], len(dates))
        report.ok(f"daily_basic 日期: {dates[0]} ~ {dates[-1]} ({len(dates)} 交易日)")

    # 检查 daily 和 daily_basic 日期一致性
    if "daily" in date_ranges and "daily_basic" in date_ranges:
        d1 = set(dfs["daily.csv"]["trade_date"].unique())
        d2 = set(dfs["daily_basic.csv"]["trade_date"].unique())
        missing_in_basic = d1 - d2
        missing_in_daily = d2 - d1
        if missing_in_daily:
            report.warn(f"daily_basic 有 {len(missing_in_daily)} 个交易日不在 daily 中")
        if missing_in_basic:
            report.warn(f"daily 有 {len(missing_in_basic)} 个交易日不在 daily_basic 中")
        if not missing_in_daily and not missing_in_basic:
            report.ok("daily 与 daily_basic 日期完全对齐")

    return date_ranges


def check_numeric_values(dfs, report):
    """检查数值列的缺失率和异常值"""
    logger.info("检查数值质量")

    for fname, df in dfs.items():
        # 找出应该是数值的列
        non_numeric = {"ts_code", "trade_date", "ann_date", "f_ann_date",
                       "end_date", "report_type", "comp_type", "end_type",
                       "div_proc", "record_date", "ex_date", "pay_date",
                       "div_listdate", "imp_ann_date", "setup_date",
                       "province", "city", "website", "email", "office",
                       "introduction", "main_business", "business_scope",
                       "chairman", "manager", "secretary", "exchange"}
        num_cols = [c for c in df.columns if c not in non_numeric]

        for col in num_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            total = len(series)
            nan_count = series.isna().sum()
            nan_ratio = nan_count / total if total > 0 else 0

            if nan_ratio > 0.9:
                report.warn(f"{fname}/{col}: 缺失率 {nan_ratio:.1%} ({nan_count}/{total})")
            elif nan_ratio > 0.5:
                report.warn(f"{fname}/{col}: 缺失率较高 {nan_ratio:.1%}")

            # 异常值检查
            if fname in VALUE_RANGES or col in VALUE_RANGES:
                key = col if col in VALUE_RANGES else None
                if key and key in VALUE_RANGES:
                    lo, hi = VALUE_RANGES[key]
                    valid = series.dropna()
                    outliers = valid[(valid < lo) | (valid > hi)]
                    if len(outliers) > 0:
                        report.warn(
                            f"{fname}/{col}: {len(outliers)} 个值超出 [{lo}, {hi}] 范围"
                        )


def check_fundamental_dates(dfs, report):
    """检查财报公告日 vs 报告期逻辑"""
    logger.info("检查财报日期逻辑")
    for fname in ["fina_indicator.csv", "income.csv", "balancesheet.csv",
                   "cashflow.csv"]:
        if fname not in dfs:
            continue
        df = dfs[fname]
        if "ann_date" not in df.columns or "end_date" not in df.columns:
            continue

        ann = pd.to_numeric(df["ann_date"], errors="coerce")
        end = pd.to_numeric(df["end_date"], errors="coerce")

        # 公告日应 >= 报告期
        bad = df[(ann.notna()) & (end.notna()) & (ann < end)]
        if len(bad) > 0:
            report.warn(
                f"{fname}: {len(bad)} 条记录的 ann_date < end_date (可能有修正公告)"
            )

        # 检查是否有重复的 (ann_date, end_date) 组合
        if "report_type" in df.columns:
            dup_cols = ["ann_date", "end_date", "report_type"]
        else:
            dup_cols = ["ann_date", "end_date"]
        available = [c for c in dup_cols if c in df.columns]
        if available:
            dups = df.duplicated(subset=available, keep=False)
            if dups.sum() > 0:
                report.warn(f"{fname}: {dups.sum()} 条重复记录 (按 {available})")

        report.ok(f"{fname}: 日期逻辑检查通过 ({len(df)} 条)")


def check_ts_code_consistency(dfs, report):
    """检查 ts_code 是否一致"""
    logger.info("检查 ts_code 一致性")
    codes = set()
    for fname, df in dfs.items():
        if "ts_code" in df.columns:
            file_codes = set(df["ts_code"].unique())
            codes |= file_codes
            if len(file_codes) > 1:
                report.warn(f"{fname} 包含多个 ts_code: {file_codes}")

    if len(codes) > 1:
        report.warn(f"数据中包含多个股票: {codes}")
    elif len(codes) == 1:
        report.ok(f"ts_code 一致: {codes.pop()}")


def check_dividend(dfs, report):
    """检查分红数据"""
    if "dividend.csv" not in dfs:
        return
    logger.info("检查分红数据")
    df = dfs["dividend.csv"]
    if "cash_div" in df.columns:
        cash_div = pd.to_numeric(df["cash_div"], errors="coerce")
        has_div = (cash_div > 0).sum()
        report.ok(f"分红记录 {len(df)} 条, 其中现金分红 {has_div} 条")


# ============================================================
# 主流程
# ============================================================
def check_one_stock(data_dir):
    """检查单只股票的 extra_data"""
    symbol = data_dir.name
    report = HealthReport(symbol)

    logger.info(f"{'=' * 60}")
    logger.info(f"检查: {symbol} ({data_dir})")
    logger.info(f"{'=' * 60}")

    # 1. 文件存在性
    dfs = check_files_exist(data_dir, report)
    if not dfs:
        report.error("没有找到任何 CSV 文件")
        return report

    # 2. 列名完整性
    check_columns(dfs, report)

    # 3. 行数
    check_row_counts(dfs, report)

    # 4. 日期覆盖
    check_date_coverage(dfs, report)

    # 5. 数值质量
    check_numeric_values(dfs, report)

    # 6. 财报日期逻辑
    check_fundamental_dates(dfs, report)

    # 7. ts_code 一致性
    check_ts_code_consistency(dfs, report)

    # 8. 分红数据
    check_dividend(dfs, report)

    return report


def main():
    parser = argparse.ArgumentParser(description="检查 extra_data 数据健康状态")
    parser.add_argument("symbols", nargs="*",
                        help="股票代码列表，如 SZ000001 SZ000002 (留空则检查所有)")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        if not EXTRA_DATA_DIR.exists():
            logger.error(f"extra_data 目录不存在: {EXTRA_DATA_DIR}")
            sys.exit(1)
        symbols = sorted([d.name for d in EXTRA_DATA_DIR.iterdir() if d.is_dir()])

    if not symbols:
        logger.error("没有找到任何股票数据目录")
        sys.exit(1)

    reports = []
    for symbol in symbols:
        data_dir = EXTRA_DATA_DIR / symbol
        if not data_dir.exists():
            logger.error(f"目录不存在: {data_dir}")
            reports.append(HealthReport(symbol))
            reports[-1].error(f"目录不存在: {data_dir}")
            continue
        reports.append(check_one_stock(data_dir))

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for r in reports:
        status = "健康" if r.healthy else "异常"
        print(f"  {r.symbol:12s}  {status}  (错误:{len(r.errors)} 警告:{len(r.warnings)})")
    print("=" * 60)

    # 详细报告
    for r in reports:
        print(r.summary())

    # 退出码: 有任何错误则返回 1
    any_error = any(not r.healthy for r in reports)
    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
