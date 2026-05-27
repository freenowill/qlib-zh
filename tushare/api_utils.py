"""
api_utils.py — TuShare API 封装工具类

提供:
  - symbol_to_ts_code(): 股票代码格式转换
  - RateLimiter: 滑动窗口三级限速器 (单进程)
  - DistributedRateLimiter: 跨容器共享文件限速器 (fcntl.flock)
  - TushareAPI: TuShare API 封装 + 限速 + 自动重试 + fetch 快捷方法

用法:
  from api_utils import TushareAPI, symbol_to_ts_code

  api = TushareAPI()
  df = api.query("daily", ts_code="000001.SZ", start_date="20240101", end_date="20241231")
"""

import fcntl
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# 避免本地目录遮蔽，确保 import sxsc_tushare 找到全局包
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

logger = logging.getLogger(__name__)


def symbol_to_ts_code(symbol):
    """SZ000001 -> 000001.SZ, SH600000 -> 600000.SH"""
    prefix = symbol[:2].upper()
    code = symbol[2:]
    return f"{code}.{prefix}"


# ============================================================
# API 限速器: 确保不超过 Tushare 三级限流
#   20次/秒, 300次/分钟, 3000次/小时
#
# DistributedRateLimiter: 跨 Docker 容器的共享文件限速器
#   通过 Docker volume 挂载的共享文件 + fcntl.flock 实现全局协调。
#   当 RATE_LIMIT_FILE 环境变量不存在或文件锁不可用时自动回退为
#   进程内存模式。
# ============================================================
class RateLimiter:
    """滑动窗口限速器 (单进程模式)，确保同时满足秒/分/时三级限制"""

    def __init__(self, max_per_sec=None, max_per_min=None, max_per_hour=None):
        self.max_per_sec = max_per_sec or int(os.environ.get("TUSHARE_RATE_PS", "18"))
        self.max_per_min = max_per_min or int(os.environ.get("TUSHARE_RATE_PM", "280"))
        self.max_per_hour = max_per_hour or int(os.environ.get("TUSHARE_RATE_PH", "2800"))
        self._timestamps = []

    def _trim_window(self, now):
        cutoff = now - 3600
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def _check_wait(self, timestamps, now):
        """检查三级限流，返回需要等待的秒数 (不修改 timestamps)"""
        wait = 0.0

        sec_calls = sum(1 for t in timestamps if t > now - 1)
        if sec_calls >= self.max_per_sec:
            recent = sorted([t for t in timestamps if t > now - 1])
            wait = max(wait, recent[0] + 1.001 - now)

        min_calls = sum(1 for t in timestamps if t > now - 60)
        if min_calls >= self.max_per_min:
            recent = sorted([t for t in timestamps if t > now - 60])
            wait = max(wait, recent[0] + 60.001 - now)

        hour_calls = len(timestamps)
        if hour_calls >= self.max_per_hour:
            wait = max(wait, timestamps[0] + 3600.001 - now)

        return wait

    def acquire(self):
        now = time.time()
        self._trim_window(now)
        wait = self._check_wait(self._timestamps, now)
        if wait > 0:
            if wait > 5:
                logging.getLogger(__name__).info("限流等待 %.0fs (已调用 ~%d 次/小时)", wait, len(self._timestamps))
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


class DistributedRateLimiter(RateLimiter):
    """跨 Docker 容器的共享文件限速器。

    使用 fcntl.flock 对共享文件加锁，确保所有容器看到同一份
    API 调用时间戳，从而实现全局限流。

    当 RATE_LIMIT_FILE 未设置或锁不可用时，自动回退为进程内存模式。
    """

    def __init__(self, max_per_sec=None, max_per_min=None, max_per_hour=None):
        super().__init__(max_per_sec, max_per_min, max_per_hour)
        self._limiter_path = os.environ.get("RATE_LIMIT_FILE", "")
        self._use_file = bool(self._limiter_path)
        self._locked = False

    def _load_timestamps(self):
        """从共享文件读取所有时间戳"""
        try:
            with open(self._limiter_path, "r") as f:
                ts = [float(line.strip()) for line in f if line.strip()]
                return ts
        except (FileNotFoundError, ValueError):
            return []

    def _save_timestamps(self, timestamps):
        """写入共享文件 (调用者需持有锁)"""
        with open(self._limiter_path, "w") as f:
            for t in timestamps:
                f.write(f"{t:.6f}\n")

    def acquire(self):
        if not self._use_file:
            return super().acquire()

        # Step 1: 获取文件锁，读取全局时间戳，计算等待时间
        wait = 0.0
        timestamps = []
        try:
            self._lock()
            timestamps = self._load_timestamps()
            now = time.time()
            # 清理过期 (>1小时)
            cutoff = now - 3600
            timestamps = [t for t in timestamps if t > cutoff]
            wait = self._check_wait(timestamps, now)
            self._unlock()
        except Exception:
            self._use_file = False
            return super().acquire()

        # Step 2: 等待 (释放锁后等待，不阻塞其他进程)
        if wait > 0:
            if wait > 5:
                logging.getLogger(__name__).info("限流等待 %.0fs (已调用 ~%d 次/小时)", wait, len(timestamps))
            time.sleep(wait)

        # Step 3: 重新加锁，追加当前时间戳
        try:
            self._lock()
            timestamps = self._load_timestamps()
            cutoff = time.time() - 3600
            timestamps = [t for t in timestamps if t > cutoff]
            timestamps.append(time.time())
            self._save_timestamps(timestamps)
            self._unlock()
        except Exception:
            self._use_file = False
            super().acquire()

    def _lock(self):
        if self._locked:
            return
        self._lock_fd = os.open(self._limiter_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            self._locked = True
        except Exception:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise

    def _unlock(self):
        if not self._locked:
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._locked = False

    def __del__(self):
        try:
            self._unlock()
        except Exception:
            pass


# ============================================================
# API 封装
# ============================================================
class TushareAPI:
    def __init__(self, rates_ps=None, rates_pm=None, rates_ph=None):
        sx.set_token(TUSHARE_TOKEN)
        self.api = sx.get_api(env="prd")
        # 优先使用分布式限速器 (跨容器共享)，回退为进程内存模式
        if os.environ.get("RATE_LIMIT_FILE"):
            self._limiter = DistributedRateLimiter(rates_ps, rates_pm, rates_ph)
        else:
            self._limiter = RateLimiter(rates_ps, rates_pm, rates_ph)

    def query(self, api_name, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                self._limiter.acquire()
                df = self.api.query(api_name, **kwargs)
                if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                    return df
                if df is None:
                    return pd.DataFrame()
                return df
            except Exception as e:
                msg = str(e)
                logger.warning(f"[{api_name}] attempt {attempt+1}/{max_retries}: {msg}")
                # 检测 Tushare 限流错误:
                #  "抱歉，您每分钟最多访问该接口300次"
                #  "抱歉，您每小时最多访问该接口3000次"
                is_rate_limit = any(kw in msg for kw in ("每分钟最多访问", "每小时最多访问", "频率", "该接口每分钟"))
                if is_rate_limit:
                    if "小时" in msg or "3000次" in msg:
                        # 小时级限流 — 需要等待 3600s 窗口重置
                        wait_s = 3600 + 60
                        logger.warning(f"[{api_name}] 小时级限流，等待 {wait_s/60:.0f} 分钟后重试")
                    elif "分钟" in msg or "300次" in msg:
                        # 分次级限流 — 等待 60s
                        wait_s = 60 + 5
                        logger.warning(f"[{api_name}] 分次级限流，等待 {wait_s}s")
                    else:
                        wait_s = (2 ** attempt) * 15
                        logger.warning(f"[{api_name}] 疑似限流，等待 {wait_s}s")
                    time.sleep(wait_s)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return pd.DataFrame()

    def fetch_daily(self, ts_code, start_date, end_date):
        return self.query("daily", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_daily_basic(self, ts_code, start_date, end_date):
        return self.query("daily_basic", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def fetch_adj_factor(self, ts_code, start_date, end_date, cache_dir=None):
        """按年批量拉取复权因子，支持 CSV 缓存以消除冗余 API 调用。

        adj_factor 是历史数据不会变，缓存在 cache_dir/<symbol>/adj_factor.csv。
        若缓存覆盖所需日期范围则直接返回，否则仅拉取缺失年份并追加。
        """
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / "adj_factor.csv"

        # 读取缓存
        cached_df = pd.DataFrame()
        if cache_path and cache_path.exists():
            try:
                cached_df = pd.read_csv(cache_path, dtype=str)
                cached_df["trade_date"] = cached_df["trade_date"].astype(str)
                cached_df["adj_factor"] = pd.to_numeric(cached_df["adj_factor"], errors="coerce")
                if not cached_df.empty:
                    cached_dates = set(cached_df["trade_date"].unique())
                    cached_min = min(cached_dates)
                    cached_max = max(cached_dates)
                    if cached_min <= start_date and cached_max >= end_date:
                        logger.info(f"  adj_factor 命中缓存 ({cached_min}~{cached_max})")
                        return cached_df.sort_values("trade_date").reset_index(drop=True)
                    logger.info(f"  adj_factor 缓存不完整 ({cached_min}~{cached_max}), "
                                f"需拉取 {start_date}~{end_date}")
            except Exception as e:
                logger.warning(f"  adj_factor 缓存损坏: {e}, 重新拉取")
                cached_df = pd.DataFrame()

        # 拉取数据: 先尝试全区间一次性拉取，失败则按年回退
        all_dfs = [cached_df] if not cached_df.empty else []
        existing_years = set()
        if not cached_df.empty:
            for d in cached_df["trade_date"]:
                existing_years.add(int(d[:4]))

        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        years_needed = [yr for yr in range(start_year, end_year + 1) if yr not in existing_years]
        fetched = 0

        if years_needed:
            # 全区间一次性拉取 (比按年拉取减少 ~7x API 调用)
            try:
                df = self.query("adj_factor", ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    all_dfs.append(df)
                    fetched = len(years_needed)  # 全区间覆盖所有缺失年份
                else:
                    years_needed = []  # API 返回空则跳过
            except Exception as e:
                logger.warning(f"adj_factor 全区间拉取失败: {e}, 按年回退...")
                for yr in years_needed:
                    sd = f"{yr}0101"
                    ed = f"{yr}1231"
                    try:
                        df = self.query("adj_factor", ts_code=ts_code, start_date=sd, end_date=ed)
                        if not df.empty:
                            all_dfs.append(df)
                            fetched += 1
                    except Exception as e:
                        logger.warning(f"adj_factor {yr}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        if len(all_dfs) == 1:
            result = all_dfs[0]
        else:
            result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")

        # 写回缓存
        if cache_path and fetched > 0:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(cache_path, index=False)
                logger.info(f"  adj_factor 缓存更新: {len(result)} 行 (新增 {fetched} 年)")
            except Exception as e:
                logger.warning(f"  adj_factor 缓存写入失败: {e}")

        return result

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

    def fetch_namechange(self, ts_code):
        """获取股票名称变更历史.

        返回列: ts_code, name, start_date, end_date, ann_date, change_reason
        """
        return self.query("namechange", ts_code=ts_code)
