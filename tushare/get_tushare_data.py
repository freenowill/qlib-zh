#!/usr/bin/env python3
"""
get_tushare_data.py — 拉取 TuShare 基础数据、行情数据及基本面数据

设计策略:
  - 少量股票 (≤50): per-stock 全量区间拉取 (最少调用)
  - 大量股票 (>50): 行情数据按 trade_date 批量拉取 (by-date batch),
    财务/静态数据 per-stock (TuShare 要求 ts_code 参数)
  - 三级限流 + 限流错误自动识别等待
  - SQLite 断点续传: --resume 跳过已成功的调用
  - 拉取完成后校验数据完整性

用法:
  python3 get_tushare_data.py                           # 5 只测试股票
  python3 get_tushare_data.py --resume                  # 断点续传
  python3 get_tushare_data.py --symbols SH600000 SZ000001
  python3 get_tushare_data.py --symbols-file cn_data/instruments/all.txt
  python3 get_tushare_data.py --symbols-file cn_data/instruments/all.txt --resume

输出:
  extra_data/{SYMBOL}/*.csv  (与现有 pipeline 兼容)
"""

import argparse
import logging
import os
import socket
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 全局 socket 超时, 防止 TuShare HTTP 请求无限挂死
socket.setdefaulttimeout(30)

# 避免本地 tushare/ 目录遮蔽
for p in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
    while p in sys.path:
        sys.path.remove(p)

from api_utils import TushareAPI, symbol_to_ts_code, RateLimiter

# ============================================================
# 配置
# ============================================================
TUSHARE_TOKEN = "4cbb80cf41ae83b53f9bc431a502c328565e53938bce7cadce52bc2a"
START_DATE = "20100101"
TEST_SYMBOLS = ["SH600000", "SH600004", "SH600006", "SZ000001", "SZ000002"]

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "extra_data"
CHECKPOINT_DB = SCRIPT_DIR / ".get_data_checkpoint.db"

# 市场数据接口 (支持 trade_date 批量拉取)
MARKET_ENDPOINTS = ["daily", "daily_basic", "adj_factor"]
# 财务数据 (必须 per-stock)
FUNDAMENTAL_ENDPOINTS = ["fina_indicator", "income", "balancesheet", "cashflow"]
# 静态数据 (per-stock, 一次性)
STATIC_ENDPOINTS = ["dividend", "stock_company"]

BATCH_THRESHOLD = 50  # 超过此数量自动切换 batch 模式

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
# 强制行缓冲，确保 Docker logs 能实时看到输出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def ts_code_to_symbol(ts_code: str) -> str:
    """000001.SZ -> SZ000001"""
    code, suffix = ts_code.split(".")
    return f"{suffix.upper()}{code}"


# ============================================================
# 断点续传 (SQLite checkpoint)
# ============================================================
class Checkpoint:
    """统一的断点表: key=(endpoint, key), 其中 key 是 ts_code 或 trade_date"""
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                endpoint TEXT NOT NULL,
                key TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                rows INTEGER DEFAULT 0,
                status TEXT DEFAULT 'ok',
                PRIMARY KEY (endpoint, key)
            )
        """)
        self.conn.commit()

    def is_done(self, endpoint: str, key: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM fetch_log WHERE endpoint=? AND key=? AND status='ok'",
            (endpoint, key),
        )
        return cur.fetchone() is not None

    def mark_done(self, endpoint: str, key: str, rows: int):
        self.conn.execute(
            "INSERT OR REPLACE INTO fetch_log (endpoint, key, fetched_at, rows, status) VALUES (?,?,?,?,?)",
            (endpoint, key, datetime.now().isoformat(), rows, "ok"),
        )
        self.conn.commit()

    def mark_error(self, endpoint: str, key: str, error: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO fetch_log (endpoint, key, fetched_at, rows, status) VALUES (?,?,?,?,?)",
            (endpoint, key, datetime.now().isoformat(), 0, f"error: {error}"),
        )
        self.conn.commit()

    def show_stats(self):
        cur = self.conn.execute(
            "SELECT endpoint, COUNT(*), SUM(rows) FROM fetch_log WHERE status='ok' GROUP BY endpoint ORDER BY endpoint"
        )
        rows = cur.fetchall()
        if rows:
            logger.info("===== 已缓存记录 =====")
            for ep, cnt, nrows in rows:
                logger.info(f"  {ep}: {cnt} 次, {nrows or 0} 行")

    def close(self):
        self.conn.close()


# ============================================================
# 交易日历
# ============================================================
def get_trading_calendar(api: TushareAPI, start_date: str, end_date: str) -> list[str]:
    df = api.query("trade_cal", exchange="SSE", start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        logger.warning("交易日历为空")
        return []
    cal = df[df["is_open"] == 1]["cal_date"].tolist()
    cal = sorted(set(str(d).replace("-", "") for d in cal))
    logger.info(f"交易日历: {len(cal)} 天 ({cal[0]} ~ {cal[-1]})")
    return cal


# ============================================================
# 模式 A: per-stock (少量股票, 最省调用)
# ============================================================
def fetch_per_stock(api: TushareAPI, ck: Checkpoint, symbols: list[str],
                    start_date: str, end_date: str, resume: bool) -> int:
    """per-stock 模式: 每只股票全区间拉取所有数据"""
    total_calls = 0

    for i, symbol in enumerate(symbols):
        ts_code = symbol_to_ts_code(symbol)
        stock_dir = OUTPUT_DIR / symbol
        stock_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n[{i+1}/{len(symbols)}] {symbol} ({ts_code})")

        # 行情数据
        for ep in MARKET_ENDPOINTS:
            if resume and ck.is_done(ep, ts_code):
                logger.info(f"  {ep} 已缓存，跳过")
                continue
            logger.info(f"  {ep} ...")
            df = api.query(ep, ts_code=ts_code, start_date=start_date, end_date=end_date)
            if not df.empty:
                df.to_csv(stock_dir / f"{ep}.csv", index=False)
                ck.mark_done(ep, ts_code, len(df))
                total_calls += 1
                logger.info(f"    -> {len(df)} 行")
            else:
                ck.mark_error(ep, ts_code, "empty")
                logger.warning(f"    -> 空")

        # 财务数据
        for ep in FUNDAMENTAL_ENDPOINTS:
            if resume and ck.is_done(ep, ts_code):
                logger.info(f"  {ep} 已缓存，跳过")
                continue
            logger.info(f"  {ep} ...")
            df = api.query(ep, ts_code=ts_code, start_date=start_date, end_date=end_date)
            if not df.empty:
                df.to_csv(stock_dir / f"{ep}.csv", index=False)
                ck.mark_done(ep, ts_code, len(df))
                total_calls += 1
                logger.info(f"    -> {len(df)} 行")
            else:
                ck.mark_error(ep, ts_code, "empty")
                logger.warning(f"    -> 空")

        # 静态数据
        for ep in STATIC_ENDPOINTS:
            if resume and ck.is_done(ep, ts_code):
                logger.info(f"  {ep} 已缓存，跳过")
                continue
            logger.info(f"  {ep} ...")
            if ep == "dividend":
                df = api.query(ep, ts_code=ts_code)
            else:
                df = api.query(ep, ts_code=ts_code)
            if not df.empty:
                df.to_csv(stock_dir / f"{ep}.csv", index=False)
                ck.mark_done(ep, ts_code, len(df))
                total_calls += 1
                logger.info(f"    -> {len(df)} 行")
            else:
                ck.mark_error(ep, ts_code, "empty")
                logger.warning(f"    -> 空")

    return total_calls


# ============================================================
# 模式 B: by-date batch (大量股票, 行情数据按交易日批量)
# ============================================================
def fetch_market_by_date(api: TushareAPI, ck: Checkpoint, trading_days: list[str],
                         resume: bool, db_conn: sqlite3.Connection):
    """按交易日批量拉取行情数据 → SQLite 临时表"""
    for ep in MARKET_ENDPOINTS:
        table_name = f"{ep}_raw"
        db_conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ts_code TEXT, trade_date TEXT,
                fetched_at TEXT DEFAULT '',
                PRIMARY KEY (ts_code, trade_date)
            )
        """)
        db_conn.commit()

    for i, day in enumerate(trading_days):
        for ep in MARKET_ENDPOINTS:
            if resume and ck.is_done(f"{ep}_by_date", day):
                continue
            try:
                df = api.query(ep, trade_date=day)
                rows = len(df) if not df.empty else 0
                if rows > 0:
                    table_name = f"{ep}_raw"
                    # 动态加列
                    _ensure_table_columns(db_conn, table_name, df)
                    df.to_sql(table_name, db_conn, if_exists="append", index=False)
                ck.mark_done(f"{ep}_by_date", day, rows)
                if (i + 1) % 200 == 0:
                    logger.info(f"  [{i+1}/{len(trading_days)}] {ep}({day}): {rows} 行, 进度 {100*(i+1)//len(trading_days)}%")
            except Exception as e:
                logger.warning(f"  {ep}({day}) 失败: {e}")
                ck.mark_error(f"{ep}_by_date", day, str(e)[:100])


def _ensure_table_columns(db_conn, table_name, df):
    """动态为 SQLite 表添加缺失列"""
    cur = db_conn.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cur.fetchall()}
    for col in df.columns:
        if col not in existing:
            try:
                db_conn.execute(f"ALTER TABLE {table_name} ADD COLUMN \"{col}\" TEXT")
            except Exception:
                pass


def export_market_csvs(db_conn: sqlite3.Connection, symbols: list[str], output_dir: Path):
    """从 SQLite 导出 market 数据到 per-stock CSV"""
    for ep in MARKET_ENDPOINTS:
        table_name = f"{ep}_raw"
        cur = db_conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cur.fetchone():
            continue
        logger.info(f"导出 {ep} -> per-stock CSV...")
        ok = 0
        for symbol in symbols:
            ts_code = symbol_to_ts_code(symbol)
            stock_dir = output_dir / symbol
            stock_dir.mkdir(parents=True, exist_ok=True)
            csv_path = stock_dir / f"{ep}.csv"
            # 已有则跳过
            if csv_path.exists() and csv_path.stat().st_size > 0:
                ok += 1
                continue
            try:
                df = pd.read_sql(
                    f"SELECT * FROM {table_name} WHERE ts_code=? ORDER BY trade_date",
                    db_conn, params=(ts_code,)
                )
                if not df.empty:
                    df.to_csv(csv_path, index=False)
                    ok += 1
            except Exception:
                pass
            if ok % 1000 == 0:
                logger.info(f"  {ep}: {ok}/{len(symbols)}")
        logger.info(f"  {ep}: 导出完成 {ok}/{len(symbols)} 只")


def fetch_financial_per_stock(api: TushareAPI, ck: Checkpoint, symbols: list[str],
                              start_date: str, end_date: str, resume: bool) -> int:
    """per-stock 拉取财务数据 (必须 per-stock, TuShare 要求 ts_code)"""
    total = 0
    for i, symbol in enumerate(symbols):
        ts_code = symbol_to_ts_code(symbol)
        stock_dir = OUTPUT_DIR / symbol
        stock_dir.mkdir(parents=True, exist_ok=True)

        for ep in FUNDAMENTAL_ENDPOINTS:
            if resume and ck.is_done(ep, ts_code):
                continue
            csv_path = stock_dir / f"{ep}.csv"
            if resume and csv_path.exists() and csv_path.stat().st_size > 0:
                ck.mark_done(ep, ts_code, 1)
                continue
            try:
                df = api.query(ep, ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df.to_csv(csv_path, index=False)
                    ck.mark_done(ep, ts_code, len(df))
                    total += 1
            except Exception as e:
                ck.mark_error(ep, ts_code, str(e)[:100])

        for ep in STATIC_ENDPOINTS:
            if resume and ck.is_done(ep, ts_code):
                continue
            csv_path = stock_dir / f"{ep}.csv"
            if resume and csv_path.exists() and csv_path.stat().st_size > 0:
                ck.mark_done(ep, ts_code, 1)
                continue
            try:
                df = api.query(ep, ts_code=ts_code)
                if not df.empty:
                    df.to_csv(csv_path, index=False)
                    ck.mark_done(ep, ts_code, len(df))
                    total += 1
            except Exception as e:
                ck.mark_error(ep, ts_code, str(e)[:100])

        if (i + 1) % 200 == 0:
            logger.info(f"  财务/静态: {i+1}/{len(symbols)} 完成 ({100*(i+1)//len(symbols)}%)")

    return total


def fetch_batch(api: TushareAPI, ck: Checkpoint, symbols: list[str],
                start_date: str, end_date: str, resume: bool) -> tuple[int, int]:
    """Batch 模式: 行情 by-date + 财务/静态 per-stock"""
    # Phase 1: 交易日历
    trading_days = get_trading_calendar(api, start_date, end_date)
    if not trading_days:
        return 0, 0

    # Phase 2: 行情数据 by-date → SQLite
    logger.info(f"\n===== Phase 1: 行情数据 (by-date, {len(trading_days)} 交易日) =====")
    db_path = OUTPUT_DIR / ".market_batch.db"
    mkt_conn = sqlite3.connect(str(db_path))
    try:
        fetch_market_by_date(api, ck, trading_days, resume, mkt_conn)
    finally:
        mkt_conn.close()

    # Phase 3: 导出到 per-stock CSV
    logger.info("\n===== Phase 2: 导出行情 CSV =====")
    mkt_conn = sqlite3.connect(str(db_path))
    try:
        export_market_csvs(mkt_conn, symbols, OUTPUT_DIR)
    finally:
        mkt_conn.close()

    # Phase 4: 财务 + 静态 per-stock
    logger.info(f"\n===== Phase 3: 财务与静态数据 (per-stock, {len(symbols)} 只) =====")
    fin_calls = fetch_financial_per_stock(api, ck, symbols, start_date, end_date, resume)

    # 估算行情调用次数
    mkt_calls = len(trading_days) * len(MARKET_ENDPOINTS)
    return mkt_calls, fin_calls


# ============================================================
# 数据完整性校验
# ============================================================
REQUIRED_FILES = [
    "daily.csv", "daily_basic.csv", "adj_factor.csv",
    "fina_indicator.csv", "income.csv", "balancesheet.csv",
    "cashflow.csv", "dividend.csv", "stock_company.csv",
]


def verify_data_completeness(symbols: list[str]):
    logger.info("\n" + "=" * 50)
    logger.info("数据完整性校验")
    logger.info("=" * 50)

    ok_count = 0
    missing_count = 0
    for symbol in symbols:
        stock_dir = OUTPUT_DIR / symbol
        missing = [f for f in REQUIRED_FILES
                   if not (stock_dir / f).exists() or (stock_dir / f).stat().st_size == 0]
        if missing:
            missing_count += 1
            if missing_count <= 10:  # 只打印前 10 个
                logger.warning(f"  {symbol}: 缺失 {', '.join(missing)}")
        else:
            ok_count += 1
    logger.info(f"完整: {ok_count}/{len(symbols)}, 不完整: {missing_count}/{len(symbols)}")


# ============================================================
# 加载股票列表
# ============================================================
def load_symbols_from_file(filepath: str) -> list[str]:
    """从 instruments 文件 (tab-separated: code, start_date, end_date) 加载股票代码

    支持格式: SH600000, SZ000001, 000001.SZ, 600000.SH
    自动过滤: 北交所 (BJ/4/8 开头), 指数 (SH000xxx, SZ399xxx)
    """
    symbols = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            code = parts[0].strip().upper()
            if not code:
                continue
            # ts_code 格式: 000001.SZ → SZ000001
            if "." in code:
                if code.endswith((".SZ", ".SH")):
                    symbols.append(ts_code_to_symbol(code))
                # .BJ / .NQ 等跳过
                continue
            # 已经是 symbol 格式: SH600000 / SZ000001
            if code.startswith("BJ"):
                continue
            if code.startswith(("SH", "SZ")):
                num = code[2:]
                # 过滤指数: SH000xxx (上证指数), SZ399xxx (深证成指等)
                if code.startswith("SH") and num.startswith("000"):
                    continue
                if code.startswith("SZ") and num.startswith("399"):
                    continue
                # 过滤 4/8 开头 (北交所/新三板, 但 SH/SZ 前缀的一般不会有)
                if num.startswith(("4", "8")):
                    continue
                symbols.append(code)
    return sorted(set(symbols))


def get_symbols(args) -> list[str]:
    """解析股票列表"""
    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
        logger.info(f"从 {args.symbols_file} 加载 {len(symbols)} 只股票")
        return symbols
    if args.symbols:
        return [s.upper() for s in args.symbols]
    return TEST_SYMBOLS


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="拉取 TuShare 基础数据、行情数据及基本面数据"
    )
    parser.add_argument("--symbols", nargs="+", default=None,
                        help=f"股票代码 (默认测试: {' '.join(TEST_SYMBOLS)})")
    parser.add_argument("--symbols-file", default=None,
                        help="从文件加载股票列表 (如 cn_data/instruments/all.txt)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传: 跳过已成功的")
    parser.add_argument("--start-date", default=START_DATE,
                        help=f"起始日期 YYYYMMDD (默认 {START_DATE})")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过完整性校验")
    parser.add_argument("--checkpoint-db", default=str(CHECKPOINT_DB),
                        help="断点数据库路径")
    parser.add_argument("--force-per-stock", action="store_true",
                        help="强制使用 per-stock 模式 (即使股票很多)")
    args = parser.parse_args()

    symbols = get_symbols(args)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = args.start_date

    use_batch = len(symbols) > BATCH_THRESHOLD and not args.force_per_stock

    logger.info("=" * 60)
    logger.info("TuShare 数据拉取")
    logger.info(f"股票: {len(symbols)} 只")
    logger.info(f"日期: {start_date} ~ {end_date}")
    logger.info(f"模式: {'by-date batch (行情批量+财务per-stock)' if use_batch else 'per-stock'}")
    logger.info(f"断点续传: {'是' if args.resume else '否'}")
    logger.info(f"输出: {OUTPUT_DIR}")
    logger.info("=" * 60)

    api = TushareAPI()
    ck = Checkpoint(Path(args.checkpoint_db))
    ck.show_stats()

    total_start = time.time()

    if use_batch:
        mkt_calls, fin_calls = fetch_batch(api, ck, symbols, start_date, end_date, args.resume)
        total_calls = mkt_calls + fin_calls
    else:
        total_calls = fetch_per_stock(api, ck, symbols, start_date, end_date, args.resume)

    elapsed = time.time() - total_start

    logger.info("\n" + "=" * 50)
    logger.info(f"拉取完成: ~{total_calls} API 调用, 耗时 {elapsed/60:.1f} 分钟")
    logger.info("=" * 50)

    ck.show_stats()
    ck.close()

    if not args.no_verify:
        verify_data_completeness(symbols)

    logger.info("\n完成.")


if __name__ == "__main__":
    main()
