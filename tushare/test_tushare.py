"""
test_tushare.py - 测试 sxsc_tushare 接口，拉取行情及基本面数据

运行方式 (在 zhuhai123/local_qlib:v1-tushare 容器中):
  docker run --rm -v $(pwd):/workspace -w /workspace zhuhai123/local_qlib:v1-tushare \
    python3 test_tushare.py SZ000002

结果保存在 ./extra_data/{SYMBOL}/ 目录下
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd

# 避免本地 tushare/ 目录遮蔽
for p in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
    while p in sys.path:
        sys.path.remove(p)

import sxsc_tushare as sx

# ============================================================
# 配置
# ============================================================
TUSHARE_TOKEN = "4cbb80cf41ae83b53f9bc431a502c328565e53938bce7cadce52bc2a"
RATE_LIMIT = 0.3               # Tushare API 调用间隔 (秒)


def symbol_to_ts_code(symbol):
    """SZ000001 -> 000001.SZ, SH600000 -> 600000.SH"""
    prefix = symbol[:2].upper()
    code = symbol[2:]
    return f"{code}.{prefix}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# API 封装
# ============================================================
def get_api():
    sx.set_token(TUSHARE_TOKEN)
    api = sx.get_api(env="prd")
    return api


def query_with_retry(api, api_name, max_retries=3, **kwargs):
    """带重试的 API 调用"""
    for attempt in range(max_retries):
        try:
            time.sleep(RATE_LIMIT)
            df = api.query(api_name, **kwargs)
            logger.info(f"[{api_name}] 获取 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.warning(f"[{api_name}] 第 {attempt+1} 次失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# ============================================================
# 数据拉取
# ============================================================
def fetch_daily(api, ts_code, start_date="20200101", end_date=None):
    """日线行情: 开盘/收盘/最高/最低/成交量/成交额/换手率等"""
    df = query_with_retry(
        api, "daily",
        ts_code=ts_code, start_date=start_date, end_date=end_date,
    )
    return df


def fetch_daily_basic(api, ts_code, start_date="20200101", end_date=None):
    """每日指标: PE/PB/PS/总市值/流通市值/换手率等"""
    df = query_with_retry(
        api, "daily_basic",
        ts_code=ts_code, start_date=start_date, end_date=end_date,
    )
    return df


def fetch_stock_company(api, ts_code):
    """上市公司基本信息: 行业/地区/上市日期/董事长/董秘/注册资本等"""
    df = query_with_retry(
        api, "stock_company",
        ts_code=ts_code,
    )
    return df


def fetch_fina_indicator(api, ts_code, start_date="20200101", end_date=None):
    """财务指标: ROE/ROA/毛利率/净利率/资产负债率/EPS/BPS 等"""
    df = query_with_retry(
        api, "fina_indicator",
        ts_code=ts_code, start_date=start_date, end_date=end_date,
    )
    return df


def fetch_income(api, ts_code, start_date="20200101", end_date=None):
    """利润表: 营业收入/营业成本/净利润 等"""
    df = query_with_retry(
        api, "income",
        ts_code=ts_code, start_date=start_date, end_date=end_date,
    )
    return df


def fetch_balancesheet(api, ts_code, start_date="20200101", end_date=None):
    """资产负债表: 总资产/总负债/股东权益 等"""
    df = query_with_retry(
        api, "balancesheet",
        ts_code=ts_code, start_date=start_date, end_date=end_date,
    )
    return df


def fetch_cashflow(api, ts_code, start_date="20200101", end_date=None):
    """现金流量表: 经营/投资/筹资 活动现金流"""
    df = query_with_retry(
        api, "cashflow",
        ts_code=ts_code, start_date=start_date, end_date=end_date,
    )
    return df


def fetch_dividend(api, ts_code):
    """分红送股记录"""
    df = query_with_retry(
        api, "dividend",
        ts_code=ts_code,
    )
    return df


# ============================================================
# 主流程
# ============================================================
def main():
    import datetime
    parser = argparse.ArgumentParser(description="从 Tushare 拉取股票数据")
    parser.add_argument("symbol", nargs="?", default="SZ000001",
                        help="股票代码，如 SZ000001、SH600000 (默认 SZ000001)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    ts_code = symbol_to_ts_code(symbol)
    output_dir = Path(__file__).parent / "extra_data" / symbol

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"目标: {symbol} ({ts_code}), 输出: {output_dir}")

    logger.info(f"初始化 sxsc_tushare API ...")
    api = get_api()

    # 默认拉取到今天
    today = datetime.date.today().strftime("%Y%m%d")

    # ------ 行情数据 ------
    logger.info("=" * 50)
    logger.info("拉取行情数据")
    logger.info("=" * 50)

    daily = fetch_daily(api, ts_code, end_date=today)
    if not daily.empty:
        path = output_dir / "daily.csv"
        daily.to_csv(path, index=False)
        logger.info(f"日线行情 -> {path}  ({len(daily)} 条)")

    daily_basic = fetch_daily_basic(api, ts_code, end_date=today)
    if not daily_basic.empty:
        path = output_dir / "daily_basic.csv"
        daily_basic.to_csv(path, index=False)
        logger.info(f"每日指标 -> {path}  ({len(daily_basic)} 条)")

    # ------ 基本面数据 ------
    logger.info("=" * 50)
    logger.info("拉取基本面数据")
    logger.info("=" * 50)

    company = fetch_stock_company(api, ts_code)
    if not company.empty:
        path = output_dir / "stock_company.csv"
        company.to_csv(path, index=False)
        logger.info(f"公司信息 -> {path}  ({len(company)} 条)")

    fina = fetch_fina_indicator(api, ts_code, end_date=today)
    if not fina.empty:
        path = output_dir / "fina_indicator.csv"
        fina.to_csv(path, index=False)
        logger.info(f"财务指标 -> {path}  ({len(fina)} 条)")

    income = fetch_income(api, ts_code, end_date=today)
    if not income.empty:
        path = output_dir / "income.csv"
        income.to_csv(path, index=False)
        logger.info(f"利润表   -> {path}  ({len(income)} 条)")

    balance = fetch_balancesheet(api, ts_code, end_date=today)
    if not balance.empty:
        path = output_dir / "balancesheet.csv"
        balance.to_csv(path, index=False)
        logger.info(f"资产负债 -> {path}  ({len(balance)} 条)")

    cashflow = fetch_cashflow(api, ts_code, end_date=today)
    if not cashflow.empty:
        path = output_dir / "cashflow.csv"
        cashflow.to_csv(path, index=False)
        logger.info(f"现金流量 -> {path}  ({len(cashflow)} 条)")

    dividend = fetch_dividend(api, ts_code)
    if not dividend.empty:
        path = output_dir / "dividend.csv"
        dividend.to_csv(path, index=False)
        logger.info(f"分红记录 -> {path}  ({len(dividend)} 条)")

    # ------ 汇总 ------
    logger.info("=" * 50)
    logger.info("数据拉取完成，结果汇总:")
    for f in sorted(output_dir.glob("*.csv")):
        df = pd.read_csv(f)
        logger.info(f"  {f.name:25s}  {len(df):>6d} 行  {len(df.columns):>3d} 列")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
