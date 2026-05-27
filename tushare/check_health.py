#!/usr/bin/env python3
"""
check_health.py - CSI300 / CSI1000 股票数据完整性检查与自动补全

功能:
  1. 读取 cn_data/instruments 下的 csi300.txt 和 csi1000.txt，取并集唯一股票
  2. 均分到 10 个 worker 进程
  3. 每个 worker 串行检查每只股票:
     - CSV 文件完整性 (9 个文件)
     - 文件内每年数据完整性 (日频 >= 200 天, 财务 >= 3 条)
  4. 缺失部分自动通过 TuShare API 拉取补全

用法:
  # 仅检查 (dry-run)
  docker run --rm -v $(pwd)/tushare:/workspace -w /workspace \\
    zhuhai123/local_qlib:v1-tushare \\
    python3 check_health.py --check-only

  # 检查并补全
  docker run --rm -v $(pwd)/tushare:/workspace -w /workspace \\
    zhuhai123/local_qlib:v1-tushare \\
    python3 check_health.py
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from api_utils import TushareAPI, symbol_to_ts_code

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = Path(__file__).parent
EXTRA_DATA_DIR = SCRIPT_DIR / "extra_data"
INSTRUMENTS_DIR = SCRIPT_DIR / "cn_data" / "instruments"
LOG_DIR = EXTRA_DATA_DIR / ".logs"

EXPECTED_CSV = [
    "daily.csv", "daily_basic.csv", "adj_factor.csv",
    "fina_indicator.csv", "income.csv", "balancesheet.csv", "cashflow.csv",
    "dividend.csv", "stock_company.csv",
]

DAILY_FILES = {
    "daily.csv":        ("daily",        "trade_date", "fetch_daily"),
    "daily_basic.csv":  ("daily_basic",  "trade_date", "fetch_daily_basic"),
    "adj_factor.csv":   ("adj_factor",   "trade_date", "fetch_adj_factor"),
}

FINANCIAL_FILES = {
    "fina_indicator.csv": ("fina_indicator", "ann_date", "fetch_fina_indicator"),
    "income.csv":         ("income",         "ann_date", "fetch_income"),
    "balancesheet.csv":   ("balancesheet",   "ann_date", "fetch_balancesheet"),
    "cashflow.csv":       ("cashflow",       "ann_date", "fetch_cashflow"),
}

STATIC_FILES = {
    "dividend.csv":      "fetch_dividend",
    "stock_company.csv": "fetch_stock_company",
}

YEAR_THRESHOLD = 200


# ============================================================
# 股票列表
# ============================================================
def load_index_symbols(*index_names: str) -> list[str]:
    """从 cn_data/instruments/<name>.txt 加载指数成分股 (取并集去重)"""
    symbols = set()
    for name in index_names:
        idx_file = INSTRUMENTS_DIR / f"{name}.txt"
        if not idx_file.exists():
            logging.warning("指数文件不存在: %s", idx_file)
            continue
        with open(idx_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                code = line.split("\t")[0].strip().upper()
                if code:
                    symbols.add(code)
    return sorted(symbols)


# ============================================================
# 检查函数
# ============================================================
def get_listing_date(stock_dir: Path) -> str:
    """获取上市日期: stock_company.csv > daily.csv > 默认"""
    sc = stock_dir / "stock_company.csv"
    if sc.exists():
        try:
            df = pd.read_csv(sc)
            if "list_date" in df.columns and not df.empty:
                ld = str(df["list_date"].dropna().iloc[0])
                if ld and ld != "nan":
                    return ld.replace("-", "")[:8]
        except Exception:
            pass

    daily = stock_dir / "daily.csv"
    if daily.exists():
        try:
            df = pd.read_csv(daily, usecols=["trade_date"])
            if not df.empty:
                dates = sorted(df["trade_date"].dropna().astype(str))
                return dates[0]
        except Exception:
            pass

    return "20000101"


def check_csv_existence(stock_dir: Path) -> list[str]:
    """返回缺失的 CSV 文件名列表"""
    missing = []
    for fname in EXPECTED_CSV:
        fp = stock_dir / fname
        if not fp.exists() or fp.stat().st_size == 0:
            missing.append(fname)
    return missing


def check_year_completeness(csv_path: Path, date_col: str,
                             list_year: int, current_year: int,
                             threshold: int = YEAR_THRESHOLD,
                             max_stale_days: int = 1) -> list[int]:
    """返回日频数据缺失的年份

    对于当前年份，不仅检查条数，还检查最新日期是否在 max_stale_days 天内。
    默认 1 天，要求最新数据为昨天或今天。
    """
    if not csv_path.exists():
        return list(range(list_year, current_year + 1))

    try:
        df = pd.read_csv(csv_path, usecols=[date_col])
        if df.empty:
            return list(range(list_year, current_year + 1))

        dates = df[date_col].dropna().astype(str)
        year_counts = defaultdict(int)
        latest = ""
        for d in dates:
            year_counts[int(d[:4])] += 1
            if d > latest:
                latest = d

        missing = []
        for y in range(list_year, current_year + 1):
            cnt = year_counts.get(y, 0)
            if y == list_year:
                need = min(threshold, 150)
            elif y == current_year:
                # 检查新鲜度: 最新日期距今是否超过 max_stale_days
                if latest:
                    latest_dt = datetime.strptime(latest[:8], "%Y%m%d")
                    stale_days = (datetime.now() - latest_dt).days
                    need = 0 if stale_days <= max_stale_days else threshold + 1  # 强制标记缺失
                else:
                    need = 1
            else:
                need = threshold
            if cnt < need:
                missing.append(y)
        return missing
    except Exception:
        return list(range(list_year, current_year + 1))


def check_financial_completeness(csv_path: Path, date_col: str,
                                  list_year: int, current_year: int,
                                  max_stale_days: int = 120) -> list[int]:
    """返回财务数据缺失的年份

    当前年份除条数检查外，还验证最新公告日期是否在 max_stale_days 天内。
    财报季频公布，容忍 120 天。
    """
    start_year = max(list_year, 2010)
    if not csv_path.exists():
        return list(range(start_year, current_year + 1))

    try:
        df = pd.read_csv(csv_path, usecols=[date_col])
        if df.empty:
            return list(range(start_year, current_year + 1))

        dates = df[date_col].dropna().astype(str)
        year_counts = defaultdict(int)
        latest = ""
        for d in dates:
            year_counts[int(d[:4])] += 1
            if d > latest:
                latest = d

        missing = []
        for y in range(start_year, current_year + 1):
            need = 1 if y == current_year else 3
            cnt = year_counts.get(y, 0)
            if cnt < need:
                missing.append(y)
            elif y == current_year and latest:
                # 即使当年有条数，也检查是否太旧
                latest_dt = datetime.strptime(latest[:8], "%Y%m%d")
                if (datetime.now() - latest_dt).days > max_stale_days:
                    missing.append(y)
        return missing
    except Exception:
        return list(range(start_year, current_year + 1))


# ============================================================
# 拉取函数
# ============================================================
def _read_csv_safe(csv_path: Path) -> pd.DataFrame:
    """安全读取 CSV，失败返回空 DataFrame"""
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def _normalize_date_col(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """统一日期列为 int64 类型，避免 str/int 混合导致 sort_values 失败"""
    if df.empty or date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_numeric(df[date_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[date_col])
    return df


def _write_csv_safe(df: pd.DataFrame, csv_path: Path):
    """安全写入 CSV，日期列转为纯字符串避免类型漂移"""
    df.to_csv(csv_path, index=False)


def pull_missing_file(api: TushareAPI, ts_code: str, stock_dir: Path,
                       fname: str, log: logging.Logger) -> bool:
    """拉取缺失的单个 CSV 文件"""
    try:
        today = datetime.now().strftime("%Y%m%d")

        if fname in DAILY_FILES:
            _, date_col, method_name = DAILY_FILES[fname]
            if method_name == "fetch_adj_factor":
                df = api.fetch_adj_factor(ts_code, "20000101", today, cache_dir=stock_dir)
            else:
                df = getattr(api, method_name)(ts_code=ts_code, start_date="20000101", end_date=today)
            if df is not None and not df.empty:
                df = _normalize_date_col(df, date_col)
                _write_csv_safe(df, stock_dir / fname)
                return True

        elif fname in FINANCIAL_FILES:
            _, date_col, method_name = FINANCIAL_FILES[fname]
            df = getattr(api, method_name)(ts_code=ts_code, start_date="20100101", end_date=today)
            if df is not None and not df.empty:
                df = _normalize_date_col(df, date_col)
                _write_csv_safe(df, stock_dir / fname)
                return True

        elif fname in STATIC_FILES:
            df = getattr(api, STATIC_FILES[fname])(ts_code=ts_code)
            if df is not None and not df.empty:
                _write_csv_safe(df, stock_dir / fname)
                return True

        return False
    except Exception as e:
        log.warning("  拉取 %s 失败: %s", fname, e)
        return False


def _years_to_ranges(years: list[int]) -> list[tuple[int, int]]:
    """将年份列表合并为连续区间. [2004,2005,2006,2008,2009] -> [(2004,2006), (2008,2009)]"""
    if not years:
        return []
    yrs = sorted(years)
    ranges = []
    start = end = yrs[0]
    for y in yrs[1:]:
        if y == end + 1:
            end = y
        else:
            ranges.append((start, end))
            start = end = y
    ranges.append((start, end))
    return ranges


def pull_missing_daily_years(api: TushareAPI, ts_code: str, stock_dir: Path,
                              fname: str, missing_years: list[int],
                              log: logging.Logger) -> int:
    """合并连续年份为区间，批量补全日频数据。返回补全的年份数。"""
    _, date_col, method = DAILY_FILES[fname]
    fetch_fn = getattr(api, method)
    csv_path = stock_dir / fname

    existing = _read_csv_safe(csv_path)
    existing = _normalize_date_col(existing, date_col)

    ranges = _years_to_ranges(missing_years)
    fixed = 0
    for yr_start, yr_end in ranges:
        sd, ed = f"{yr_start}0101", f"{yr_end}1231"
        try:
            if method == "fetch_adj_factor":
                df = api.fetch_adj_factor(ts_code, sd, ed, cache_dir=stock_dir)
            else:
                df = fetch_fn(ts_code=ts_code, start_date=sd, end_date=ed)
            if df is not None and not df.empty:
                df = _normalize_date_col(df, date_col)
                existing = pd.concat([existing, df], ignore_index=True)
                fixed += (yr_end - yr_start + 1)
            else:
                log.debug("  %s %s-%s 返回空", fname, yr_start, yr_end)
        except Exception as e:
            log.warning("  %s %s-%s 失败: %s", fname, yr_start, yr_end, e)

    if fixed > 0:
        existing = existing.drop_duplicates(subset=[date_col]).sort_values(date_col)
        _write_csv_safe(existing, csv_path)
    return fixed


def pull_missing_financial_years(api: TushareAPI, ts_code: str, stock_dir: Path,
                                  fname: str, missing_years: list[int],
                                  log: logging.Logger) -> int:
    """合并连续年份为区间，批量补全财务数据。返回补全的年份数。"""
    _, date_col, method = FINANCIAL_FILES[fname]
    fetch_fn = getattr(api, method)
    csv_path = stock_dir / fname

    existing = _read_csv_safe(csv_path)
    existing = _normalize_date_col(existing, date_col)

    ranges = _years_to_ranges(missing_years)
    fixed = 0
    for yr_start, yr_end in ranges:
        try:
            df = fetch_fn(ts_code=ts_code, start_date=f"{yr_start}0101", end_date=f"{yr_end}1231")
            if df is not None and not df.empty:
                df = _normalize_date_col(df, date_col)
                existing = pd.concat([existing, df], ignore_index=True)
                fixed += (yr_end - yr_start + 1)
        except Exception as e:
            log.warning("  %s %s-%s 失败: %s", fname, yr_start, yr_end, e)

    if fixed > 0:
        existing = existing.drop_duplicates(subset=[date_col]).sort_values(date_col)
        _write_csv_safe(existing, csv_path)
    return fixed


# ============================================================
# Worker
# ============================================================
def worker(worker_id: int, symbols: list[str], check_only: bool,
            year_threshold: int):
    """每个 worker 串行处理分配到的股票，结果写入日志文件"""
    log = _setup_worker_logger(worker_id)
    api = None if check_only else TushareAPI()

    current_year = datetime.now().year
    total = len(symbols)

    stats = {"checked": 0, "ok": 0, "csv_missing": 0, "csv_fixed": 0,
             "years_missing": 0, "years_fixed": 0, "errors": 0,
             "api_calls": 0, "planned_calls": 0}
    details = []

    log.info("开始: %d 只股票", total)

    for i, symbol in enumerate(symbols):
        stock_dir = EXTRA_DATA_DIR / symbol
        ts_code = symbol_to_ts_code(symbol)
        stock_dir.mkdir(parents=True, exist_ok=True)

        stock_issues = {"symbol": symbol, "missing_files": [], "missing_years": {}}

        try:
            # 1. CSV 文件完整性
            missing_csv = check_csv_existence(stock_dir)
            if missing_csv:
                stats["csv_missing"] += len(missing_csv)
                stock_issues["missing_files"] = missing_csv
                log.info("[%d/%d] %s 缺文件: %s", i + 1, total, symbol, missing_csv)

                if not check_only:
                    for fname in missing_csv:
                        if pull_missing_file(api, ts_code, stock_dir, fname, log):
                            stats["csv_fixed"] += 1
                            stats["api_calls"] += 1
                        else:
                            stats["errors"] += 1
                else:
                    stats["planned_calls"] += len(missing_csv)

            # 2. 年份完整性
            list_date = get_listing_date(stock_dir)
            list_year = int(list_date[:4])

            # 日频
            for fname, (_, date_col, _) in DAILY_FILES.items():
                if not (stock_dir / fname).exists():
                    continue
                missing_yrs = check_year_completeness(
                    stock_dir / fname, date_col, list_year, current_year, year_threshold)
                if missing_yrs:
                    nranges = len(_years_to_ranges(missing_yrs))
                    stock_issues["missing_years"][fname] = missing_yrs
                    stats["years_missing"] += len(missing_yrs)
                    stats["planned_calls"] += nranges

                    if not check_only:
                        n = pull_missing_daily_years(
                            api, ts_code, stock_dir, fname, missing_yrs, log)
                        stats["years_fixed"] += n
                        stats["api_calls"] += n

            # 财务
            for fname, (_, date_col, _) in FINANCIAL_FILES.items():
                if not (stock_dir / fname).exists():
                    continue
                missing_yrs = check_financial_completeness(
                    stock_dir / fname, date_col, list_year, current_year)
                if missing_yrs:
                    nranges = len(_years_to_ranges(missing_yrs))
                    stock_issues["missing_years"][fname] = missing_yrs
                    stats["years_missing"] += len(missing_yrs)
                    stats["planned_calls"] += nranges

                    if not check_only:
                        n = pull_missing_financial_years(
                            api, ts_code, stock_dir, fname, missing_yrs, log)
                        stats["years_fixed"] += n
                        stats["api_calls"] += n

            if not stock_issues["missing_files"] and not stock_issues["missing_years"]:
                stats["ok"] += 1
            else:
                details.append(stock_issues)

            stats["checked"] += 1

        except Exception as e:
            stats["errors"] += 1
            log.error("[%d/%d] %s 异常: %s", i + 1, total, symbol, e)
            traceback.print_exc(file=sys.stderr)

        # 进度 (每 50 只)
        if (i + 1) % 50 == 0:
            log.info("--- %d/%d (ok=%d, fix_csv=%d, fix_yr=%d, err=%d) ---",
                     i + 1, total, stats["ok"], stats["csv_fixed"],
                     stats["years_fixed"], stats["errors"])

    # 写结果文件
    result_file = LOG_DIR / f"check_health_w{worker_id:02d}_result.json"
    with open(result_file, "w") as f:
        json.dump({"worker_id": worker_id, "stats": stats, "details": details},
                  f, ensure_ascii=False, indent=2, default=str)

    log.info("完成: check=%d ok=%d csv_miss=%d csv_fix=%d yr_miss=%d yr_fix=%d api=%d err=%d",
             stats["checked"], stats["ok"], stats["csv_missing"], stats["csv_fixed"],
             stats["years_missing"], stats["years_fixed"], stats["api_calls"], stats["errors"])


def _setup_worker_logger(worker_id: int) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"w{worker_id:02d}")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    fh = logging.FileHandler(LOG_DIR / f"check_health_w{worker_id:02d}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)

    # 控制台: 输出 INFO+ 让用户看到实时进度
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(f"[W{worker_id:02d}] %(message)s"))
    log.addHandler(ch)

    # 让 api_utils 的限流日志也写入 worker 文件
    au_log = logging.getLogger("api_utils")
    au_log.setLevel(logging.INFO)
    au_log.handlers.clear()
    au_log.addHandler(fh)
    au_log.propagate = False

    return log


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CSI300/CSI1000 数据完整性检查与补全")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--check-only", action="store_true", help="仅检查 (dry-run)")
    parser.add_argument("--year-threshold", type=int, default=YEAR_THRESHOLD,
                        help=f"日频数据最低天数/年 (默认 {YEAR_THRESHOLD})")
    parser.add_argument("--index", nargs="+", default=["csi300", "csi1000"],
                        help="指数名称 (默认 csi300 csi1000)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    main_log = logging.getLogger("main")

    # 加载股票列表
    symbols = load_index_symbols(*args.index)
    total = len(symbols)
    n_workers = min(args.workers, total) if total > 0 else 1

    # 清空旧 worker 日志，避免进度监控读到上次运行的数据
    for w_id in range(n_workers):
        (LOG_DIR / f"check_health_w{w_id:02d}.log").write_text("")
        (LOG_DIR / f"check_health_w{w_id:02d}_result.json").unlink(missing_ok=True)

    main_log.info("=" * 60)
    main_log.info("check_health: %s 成分股", ", ".join(args.index))
    main_log.info("股票: %d 只, Workers: %d, 模式: %s, 阈值: %d 天/年",
                  total, n_workers,
                  "仅检查" if args.check_only else "检查+补全",
                  args.year_threshold)
    main_log.info("=" * 60)

    # 均分股票
    chunk_size = (total + n_workers - 1) // n_workers
    processes = []
    for w_id in range(n_workers):
        start = w_id * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            break
        chunk = symbols[start:end]
        p = Process(target=worker, args=(w_id, chunk, args.check_only, args.year_threshold))
        p.start()
        processes.append(p)

    main_log.info("已启动 %d 个 worker, 分配: %s",
                  len(processes),
                  [len(symbols[i * chunk_size:min((i + 1) * chunk_size, total)])
                   for i in range(len(processes))])

    # 进度监控线程: 每 30s 汇总各 worker 进度到控制台
    import threading
    stop_monitor = threading.Event()
    monitor_start = time.time()

    def monitor_progress():
        first_run = True
        while not stop_monitor.is_set():
            stop_monitor.wait(10 if first_run else 30)
            first_run = False
            if stop_monitor.is_set():
                break
            parts = []
            for w_id in range(len(processes)):
                log_file = LOG_DIR / f"check_health_w{w_id:02d}.log"
                if not log_file.exists():
                    continue
                try:
                    if log_file.stat().st_mtime < monitor_start:
                        continue
                    with open(log_file) as lf:
                        lines = lf.readlines()

                    # 找最后一条进度行
                    prog_line = ""
                    for line in reversed(lines):
                        if "---" in line and "/" in line:
                            prog_line = line.strip().split("---", 1)[-1].strip()
                            break

                    # 检查最近是否在限流等待 (日志末尾有 "限流等待")
                    rate_limited = any("限流等待" in l for l in lines[-5:])

                    if prog_line:
                        tag = "[限流中]" if rate_limited else ""
                        parts.append(f"W{w_id}:{prog_line} {tag}")
                except Exception:
                    pass
            if parts:
                main_log.info("进度 | %s", " | ".join(parts))

    monitor = threading.Thread(target=monitor_progress, daemon=True)
    monitor.start()

    for p in processes:
        p.join()
    stop_monitor.set()

    # 汇总所有 worker 结果
    total_stats = {"checked": 0, "ok": 0, "csv_missing": 0, "csv_fixed": 0,
                   "years_missing": 0, "years_fixed": 0, "errors": 0,
                   "api_calls": 0, "planned_calls": 0}
    all_details = []

    for w_id in range(len(processes)):
        rf = LOG_DIR / f"check_health_w{w_id:02d}_result.json"
        if rf.exists():
            with open(rf) as f:
                r = json.load(f)
            for k in total_stats:
                total_stats[k] += r["stats"].get(k, 0)
            all_details.extend(r.get("details", []))

    # 报告
    main_log.info("=" * 60)
    main_log.info("汇总")
    main_log.info("  检查: %d  正常: %d  错误: %d",
                  total_stats["checked"], total_stats["ok"], total_stats["errors"])
    main_log.info("  缺 CSV 文件: %d  已补全: %d",
                  total_stats["csv_missing"], total_stats["csv_fixed"])
    main_log.info("  缺年份:      %d  已补全: %d",
                  total_stats["years_missing"], total_stats["years_fixed"])

    total_calls = total_stats["api_calls"] if total_stats["api_calls"] > 0 else total_stats["planned_calls"]
    if total_calls > 0:
        main_log.info("  API 调用:    %d 次 (合并连续年份后)", total_calls)
        main_log.info("  预估耗时:    %.1f 小时 (按 2800 次/小时)", total_calls / 2800)
    main_log.info("=" * 60)

    # 有问题股票摘要
    if all_details:
        main_log.info("问题股票: %d 只", len(all_details))
        for d in all_details[:20]:
            parts = [d["symbol"]]
            if d["missing_files"]:
                parts.append(f"缺文件({len(d['missing_files'])}):{','.join(d['missing_files'])}")
            if d["missing_years"]:
                yr_sum = " ".join(f"{k}:{len(v)}y" for k, v in d["missing_years"].items())
                parts.append(yr_sum)
            main_log.info("  %s", " | ".join(parts))
        if len(all_details) > 20:
            main_log.info("  ... 等 %d 只", len(all_details) - 20)

    # 详细报告
    report_path = LOG_DIR / "check_health_report.json"
    with open(report_path, "w") as f:
        json.dump({"time": datetime.now().isoformat(), "index": args.index,
                    "stats": total_stats, "problem_count": len(all_details),
                    "problems": all_details[:100]},
                  f, ensure_ascii=False, indent=2, default=str)
    main_log.info("报告: %s", report_path)

    if total_stats["errors"] > 0:
        main_log.warning("有 %d 个错误, 查看 worker 日志", total_stats["errors"])


if __name__ == "__main__":
    main()
