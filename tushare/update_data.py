"""
update_data.py - 通过 Tushare API 补充 cn_data 缺失数据

功能:
  1. 读取现有 cn_data，识别缺失的交易日和股票
  2. 从 Tushare API 获取最新行情数据
  3. 按 cn_data 的归一化方案处理数据
  4. 输出完整的 qlib bin 格式到 cn_data_update/

用法:
  python update_data.py --tushare_token YOUR_TOKEN
  python update_data.py --tushare_token YOUR_TOKEN --end_date 2026-05-15
  python update_data.py --tushare_token YOUR_TOKEN --mode full --start_date 2025-01-01
"""

import os
import sys
import time
import struct
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import fire

# 避免本地 tushare/ 目录遮蔽已安装的 tushare 包
_script_dir = os.path.dirname(os.path.abspath(__file__))
_cwd = os.getcwd()
for _p in [_script_dir, _cwd]:
    while _p in sys.path:
        sys.path.remove(_p)

# ============================================================
# Constants
# ============================================================

FEATURES = ['open', 'close', 'high', 'low', 'vwap', 'volume', 'amount', 'adjclose', 'change', 'factor']
BIN_SUFFIX = '.day.bin'
RATE_LIMIT = 0.3  # seconds between Tushare API calls
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# TushareClient - API 封装
# ============================================================

class TushareClient:
    def __init__(self, token, rate_limit=RATE_LIMIT, server=None):
        self.rate_limit = rate_limit
        self._last_call = 0
        # 尝试 sxsc_tushare (山西证券定制版)，回退到标准 tushare
        try:
            import sxsc_tushare as sx
            sx.set_token(token)
            self.pro = sx.get_api(env="prd")
            logger.info("使用 sxsc_tushare 接口")
        except ImportError:
            import tushare as ts
            ts.set_token(token)
            if server:
                # 自定义服务器: monkey-patch DataApi 的 HTTP 地址
                from tushare.pro import client
                client.DataApi._DataApi__http_url = server
                logger.info(f"使用自定义服务器: {server}")
            self.pro = ts.pro_api()
            logger.info("使用标准 tushare 接口")

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call = time.time()

    def _retry(self, func, *args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                self._wait()
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API 调用失败 (尝试 {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def get_trade_calendar(self, start_date, end_date):
        """获取交易日历，返回 YYYYMMDD 格式列表"""
        df = self._retry(
            self.pro.trade_cal,
            exchange='SSE', is_open='1',
            start_date=start_date, end_date=end_date,
            fields='cal_date'
        )
        return sorted(df['cal_date'].tolist())

    def get_daily_with_adj(self, trade_date):
        """获取某日所有股票行情 + 复权因子，返回 DataFrame"""
        for _ in range(MAX_RETRIES):
            try:
                self._wait()
                price_df = self.pro.daily(trade_date=trade_date)
                self._wait()
                adj_df = self.pro.adj_factor(trade_date=trade_date)
                if price_df is None or price_df.empty:
                    return pd.DataFrame()
                if adj_df is None or adj_df.empty:
                    return pd.DataFrame()
                merged = pd.merge(price_df, adj_df, on='ts_code', how='inner')
                merged['adj_close'] = merged['close'] * merged['adj_factor']
                return merged
            except Exception as e:
                logger.warning(f"获取 {trade_date} 数据失败: {e}")
                time.sleep(1)
        return pd.DataFrame()

    def get_index_daily(self, ts_code, start_date, end_date):
        """获取指数日线行情"""
        df = self._retry(
            self.pro.index_daily,
            ts_code=ts_code,
            start_date=start_date, end_date=end_date
        )
        return df if df is not None else pd.DataFrame()

    def get_stock_list(self):
        """获取全部股票列表 (上市+退市)"""
        l_data = self._retry(self.pro.stock_basic, list_status='L',
                             fields='ts_code,symbol,exchange,list_date,delist_date')
        d_data = self._retry(self.pro.stock_basic, list_status='D',
                             fields='ts_code,symbol,exchange,list_date,delist_date')
        return pd.concat([l_data, d_data], ignore_index=True)

    def get_index_weight(self, index_code, start_date, end_date):
        """获取指数成分股权重，按 15 天窗口分批获取，返回 DataFrame"""
        time_step = timedelta(days=15)
        cur_start = datetime.strptime(start_date, '%Y%m%d')
        cur_end_dt = datetime.strptime(end_date, '%Y%m%d')
        result_dfs = []
        empty_count = 0
        while cur_start < cur_end_dt:
            window_end = min(cur_start + time_step, cur_end_dt)
            df = self._retry(
                self.pro.index_weight,
                index_code=index_code,
                start_date=cur_start.strftime('%Y%m%d'),
                end_date=window_end.strftime('%Y%m%d')
            )
            cur_start = window_end
            if df is not None and not df.empty:
                result_dfs.append(df)
                empty_count = 0
            else:
                empty_count += 1
                if empty_count >= 20:
                    logger.info(f"  {index_code} 连续 20 次空数据，停止获取")
                    break
        if result_dfs:
            return pd.concat(result_dfs, ignore_index=True)
        return pd.DataFrame()

    def get_full_year_calendar(self, year):
        """获取某年完整交易日历，返回 YYYYMMDD 格式列表"""
        df = self._retry(
            self.pro.trade_cal,
            exchange='SSE', is_open='1',
            start_date=f'{year}0101', end_date=f'{year}1231',
            fields='cal_date'
        )
        return sorted(df['cal_date'].tolist())


# ============================================================
# ExistingDataReader - 读取 cn_data
# ============================================================

class ExistingDataReader:
    def __init__(self, cn_data_dir):
        self.cn_data_dir = Path(cn_data_dir)

    def read_calendar(self):
        """读取交易日历，返回 ['YYYY-MM-DD', ...] 列表"""
        cal_file = self.cn_data_dir / 'calendars' / 'day.txt'
        with open(cal_file) as f:
            return [line.strip() for line in f if line.strip()]

    def read_instruments(self):
        """读取股票列表，返回 {symbol: (start_date, end_date), ...}"""
        inst_file = self.cn_data_dir / 'instruments' / 'all.txt'
        instruments = {}
        with open(inst_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    instruments[parts[0]] = (parts[1], parts[2])
        return instruments

    def get_all_symbols(self):
        """列出 features/ 下所有股票目录，返回大写 symbol"""
        features_dir = self.cn_data_dir / 'features'
        if not features_dir.exists():
            return []
        return sorted([d.name.upper() for d in features_dir.iterdir() if d.is_dir()])

    def read_bin(self, symbol, feature):
        """读取 bin 文件，返回 (start_idx, np.ndarray)"""
        path = self.cn_data_dir / 'features' / symbol.lower() / f'{feature}{BIN_SUFFIX}'
        if not path.exists():
            return None, None
        data = np.fromfile(str(path), dtype='<f4')
        if len(data) < 1:
            return None, None
        start_idx = int(data[0])
        values = data[1:]
        return start_idx, values

    def read_base_price(self, symbol):
        """读取 adjclose 的第一个非 NaN 值作为归一化基准"""
        start_idx, values = self.read_bin(symbol, 'adjclose')
        if values is None or len(values) == 0:
            return None
        for v in values:
            if not np.isnan(v) and v > 0:
                return float(v)
        return None

    def get_stock_end_idx(self, symbol):
        """获取某只股票在日历中的结束索引"""
        start_idx, values = self.read_bin(symbol, 'close')
        if values is None:
            return None
        return start_idx + len(values) - 1


# ============================================================
# DataNormalizer - 归一化处理
# ============================================================

class DataNormalizer:
    @staticmethod
    def ts_code_to_symbol(ts_code):
        """600000.SH -> SH600000"""
        code, exchange = ts_code.split('.')
        return f'{exchange.upper()}{code}'

    @staticmethod
    def ts_code_to_symbol_upper(ts_code):
        """600000.SH -> SH600000"""
        code, exchange = ts_code.split('.')
        return f'{exchange.upper()}{code}'

    @staticmethod
    def symbol_to_ts_code(symbol):
        """SH600000 -> 600000.SH (accepts both upper and lower case)"""
        exchange = symbol[:2].upper()
        code = symbol[2:]
        return f'{code}.{exchange}'

    @staticmethod
    def normalize(raw_df, base_price):
        """
        对 Tushare 日线数据做归一化。

        raw_df 必须包含列: open, high, low, close, vol, amount, adj_factor, adj_close
        base_price: 从 cn_data 读取的 adjclose[first_day]

        返回 dict of np.ndarray, key 为 feature 名。
        """
        adj_factor = raw_df['adj_factor'].values.astype(np.float64)
        close_raw = raw_df['close'].values.astype(np.float64)
        open_raw = raw_df['open'].values.astype(np.float64)
        high_raw = raw_df['high'].values.astype(np.float64)
        low_raw = raw_df['low'].values.astype(np.float64)
        vol_raw = raw_df['vol'].values.astype(np.float64)  # Tushare vol 单位: 手(100股)
        amount_raw = raw_df['amount'].values.astype(np.float64)  # Tushare amount 单位: 千元
        adj_close = raw_df['adj_close'].values.astype(np.float64)

        close_norm = adj_close / base_price
        open_norm = open_raw * adj_factor / base_price
        high_norm = high_raw * adj_factor / base_price
        low_norm = low_raw * adj_factor / base_price

        # vwap = amount / volume * 10，再归一化
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_vwap = np.where(vol_raw > 0, amount_raw / vol_raw * 10, np.nan)
        vwap_norm = raw_vwap / base_price

        volume_norm = vol_raw * base_price
        factor_norm = adj_factor / base_price

        # change = daily return of normalized close
        change = np.full(len(close_norm), np.nan, dtype=np.float64)
        if len(close_norm) > 1:
            change[1:] = (close_norm[1:] - close_norm[:-1]) / close_norm[:-1]

        return {
            'open': open_norm.astype(np.float32),
            'close': close_norm.astype(np.float32),
            'high': high_norm.astype(np.float32),
            'low': low_norm.astype(np.float32),
            'vwap': vwap_norm.astype(np.float32),
            'volume': volume_norm.astype(np.float32),
            'amount': amount_raw.astype(np.float32),
            'adjclose': adj_close.astype(np.float32),
            'change': change.astype(np.float32),
            'factor': factor_norm.astype(np.float32),
        }

    @staticmethod
    def compute_base_price(adj_close_series):
        """从 adjclose 序列计算 base_price (第一个非 NaN 的正值)"""
        for v in adj_close_series:
            if not np.isnan(v) and v > 0:
                return float(v)
        return None


# ============================================================
# BinaryWriter - 写入 qlib bin 格式
# ============================================================

class BinaryWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def setup(self):
        """创建输出目录结构"""
        (self.output_dir / 'calendars').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'instruments').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'features').mkdir(parents=True, exist_ok=True)

    def write_calendar(self, dates):
        """写入日历文件"""
        path = self.output_dir / 'calendars' / 'day.txt'
        with open(path, 'w') as f:
            for d in dates:
                f.write(d + '\n')
        logger.info(f"写入日历: {len(dates)} 天 -> {path}")

    def write_instruments(self, instruments):
        """写入股票列表文件"""
        path = self.output_dir / 'instruments' / 'all.txt'
        with open(path, 'w') as f:
            for symbol, (start, end) in sorted(instruments.items()):
                f.write(f'{symbol}\t{start}\t{end}\n')
        logger.info(f"写入股票列表: {len(instruments)} 只 -> {path}")

    def write_bin_new(self, symbol, feature, start_idx, data):
        """写入新的 bin 文件 (header + data)"""
        feat_dir = self.output_dir / 'features' / symbol.lower()
        feat_dir.mkdir(parents=True, exist_ok=True)
        path = feat_dir / f'{feature}{BIN_SUFFIX}'
        header = np.array([float(start_idx)], dtype='<f4')
        np.hstack([header, data.astype('<f4')]).tofile(str(path))

    def write_bin_append(self, symbol, feature, data):
        """向已有 bin 文件追加数据"""
        path = self.output_dir / 'features' / symbol.lower() / f'{feature}{BIN_SUFFIX}'
        if not path.exists():
            logger.warning(f"追加失败，文件不存在: {path}")
            return
        with open(str(path), 'ab') as f:
            data.astype('<f4').tofile(f)

    def copy_stock_from_source(self, symbol, source_dir):
        """从源目录复制某只股票的全部 bin 文件"""
        src = Path(source_dir) / 'features' / symbol.lower()
        dst = self.output_dir / 'features' / symbol.lower()
        if not src.exists():
            return
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix == '.bin':
                shutil.copy2(str(f), str(dst / f.name))

    def write_day_future(self, dates):
        """写入 day_future.txt (全年交易日历)"""
        path = self.output_dir / 'calendars' / 'day_future.txt'
        with open(path, 'w') as f:
            for d in dates:
                f.write(d + '\n')
        logger.info(f"写入全年日历: {len(dates)} 天 -> {path}")

    def write_index_instruments(self, index_name, records):
        """写入指数成分股文件 (instruments/csi*.txt)
        records: list of (symbol_upper, start_date, end_date)
        """
        path = self.output_dir / 'instruments' / f'{index_name}.txt'
        with open(path, 'w') as f:
            for symbol, start, end in sorted(records):
                f.write(f'{symbol}\t{start}\t{end}\n')
        logger.info(f"写入 {index_name}.txt: {len(records)} 条记录 -> {path}")


# ============================================================
# UpdateOrchestrator - 主流程
# ============================================================

class UpdateOrchestrator:
    def __init__(self, cn_data_dir, output_dir, tushare_token,
                 start_date=None, end_date=None, mode='incremental',
                 rate_limit=RATE_LIMIT, server=None):
        self.cn_data_dir = Path(cn_data_dir)
        self.output_dir = Path(output_dir)
        self.token = tushare_token
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.mode = mode
        self.rate_limit = rate_limit
        self.server = server

        self.reader = ExistingDataReader(cn_data_dir)
        self.writer = BinaryWriter(output_dir)
        self.normalizer = DataNormalizer()
        self.client = None  # 延迟初始化

    def _init_client(self):
        if self.client is None:
            self.client = TushareClient(self.token, self.rate_limit, self.server)

    def _date_to_compact(self, d):
        """'2026-05-08' -> '20260508'"""
        return d.replace('-', '')

    def _compact_to_date(self, d):
        """'20260508' -> '2026-05-08'"""
        return f'{d[:4]}-{d[4:6]}-{d[6:8]}'

    def _build_full_calendar(self, existing_cal, new_dates):
        """合并现有日历和新日期，去重排序"""
        all_dates = sorted(set(existing_cal + new_dates))
        return all_dates

    def _align_data_to_calendar(self, raw_df, calendar_dates, date_col='trade_date'):
        """
        将 Tushare 返回的数据对齐到日历。
        Tushare date 格式: '20260508'，calendar 格式: '2026-05-08'
        返回按日历顺序排列的 DataFrame，缺失日期填充 NaN。
        """
        if raw_df.empty:
            n = len(calendar_dates)
            return pd.DataFrame({
                'open': [np.nan]*n, 'high': [np.nan]*n, 'low': [np.nan]*n,
                'close': [np.nan]*n, 'vol': [np.nan]*n, 'amount': [np.nan]*n,
                'adj_factor': [np.nan]*n, 'adj_close': [np.nan]*n,
            }, index=calendar_dates)

        # 转换日期格式
        raw_df = raw_df.copy()
        raw_df['_date'] = raw_df[date_col].apply(self._compact_to_date)
        raw_df = raw_df.set_index('_date')

        # 按日历重索引
        result = raw_df.reindex(calendar_dates)
        return result

    def run(self):
        """主入口"""
        self._init_client()
        self.writer.setup()

        # 1. 读取现有数据
        logger.info("读取现有 cn_data ...")
        existing_cal = self.reader.read_calendar()
        existing_instruments = self.reader.read_instruments()
        existing_symbols = set(self.reader.get_all_symbols())
        logger.info(f"  现有日历: {len(existing_cal)} 天 ({existing_cal[0]} ~ {existing_cal[-1]})")
        logger.info(f"  现有股票: {len(existing_symbols)} 只")

        # 2. 确定缺失日期
        existing_cal_set = set(existing_cal)
        last_existing_date = existing_cal[-1]
        target_end = self.end_date

        if self.mode == 'incremental':
            fetch_start = last_existing_date
        else:
            fetch_start = self.start_date or last_existing_date

        logger.info(f"从 Tushare 获取交易日历: {fetch_start} ~ {target_end} ...")
        new_trade_dates_compact = self.client.get_trade_calendar(
            self._date_to_compact(fetch_start),
            self._date_to_compact(target_end)
        )
        new_trade_dates = [self._compact_to_date(d) for d in new_trade_dates_compact]

        # 过滤掉已有日期 (incremental 模式下)
        if self.mode == 'incremental':
            dates_to_fetch = [d for d in new_trade_dates if d not in existing_cal_set]
        else:
            dates_to_fetch = new_trade_dates

        if not dates_to_fetch:
            logger.info("没有需要更新的日期，数据已是最新。")
            # 直接复制现有数据到输出目录
            self._copy_existing_data(existing_symbols)
            self.writer.write_calendar(existing_cal)
            self.writer.write_instruments(existing_instruments)
            self._update_day_future()
            self._update_index_instruments()
            return

        logger.info(f"需要获取 {len(dates_to_fetch)} 个交易日数据: {dates_to_fetch[0]} ~ {dates_to_fetch[-1]}")

        # 3. 逐日获取新数据
        # new_data[ts_code] = list of (date, row_dict)
        new_data_by_symbol = {}
        index_symbols = {'000300.SH', '399300.SZ', '000905.SH', '000906.SH', '000852.SH', '000985.SH'}

        for i, date in enumerate(dates_to_fetch):
            compact = self._date_to_compact(date)
            logger.info(f"[{i+1}/{len(dates_to_fetch)}] 获取 {date} ...")

            # 获取股票行情
            df = self.client.get_daily_with_adj(compact)
            if not df.empty:
                for _, row in df.iterrows():
                    ts_code = row['ts_code']
                    symbol = self.normalizer.ts_code_to_symbol(ts_code)
                    if symbol not in new_data_by_symbol:
                        new_data_by_symbol[symbol] = []
                    new_data_by_symbol[symbol].append({
                        'date': date,
                        'open': row['open'], 'high': row['high'],
                        'low': row['low'], 'close': row['close'],
                        'vol': row['vol'], 'amount': row['amount'],
                        'adj_factor': row['adj_factor'],
                        'adj_close': row['adj_close'],
                    })

            # 获取指数行情 (指数没有 adj_factor，用 close 作为 adj_close)
            for idx_code in index_symbols:
                try:
                    idx_df = self.client.get_index_daily(idx_code, compact, compact)
                    if idx_df is not None and not idx_df.empty:
                        row = idx_df.iloc[0]
                        symbol = self.normalizer.ts_code_to_symbol(idx_code)
                        if symbol not in new_data_by_symbol:
                            new_data_by_symbol[symbol] = []
                        new_data_by_symbol[symbol].append({
                            'date': date,
                            'open': row['open'], 'high': row['high'],
                            'low': row['low'], 'close': row['close'],
                            'vol': row.get('vol', np.nan),
                            'amount': row.get('amount', np.nan),
                            'adj_factor': 1.0,  # 指数不复权
                            'adj_close': row['close'],
                        })
                except Exception as e:
                    logger.warning(f"获取指数 {idx_code} {date} 失败: {e}")

        logger.info(f"获取到 {len(new_data_by_symbol)} 只股票/指数的新数据")

        # 4. 数据完整性校验，仅保留有数据的日期
        dates_with_data = set()
        for records in new_data_by_symbol.values():
            for r in records:
                dates_with_data.add(r['date'])
        missing_dates = [d for d in dates_to_fetch if d not in dates_with_data]
        if missing_dates:
            logger.warning(f"以下日期未获取到任何数据，将从日历中剔除: {missing_dates}")
        valid_new_dates = sorted(dates_with_data)

        if not valid_new_dates:
            logger.info("没有任何有效新数据，直接复制现有数据。")
            self._copy_existing_data(existing_symbols)
            self.writer.write_calendar(existing_cal)
            self.writer.write_instruments(existing_instruments)
            self._update_day_future()
            self._update_index_instruments()
            return

        # 5. 合并日历 (仅包含有数据的日期)
        full_calendar = self._build_full_calendar(existing_cal, valid_new_dates)
        calendar_to_idx = {d: i for i, d in enumerate(full_calendar)}
        logger.info(f"完整日历: {len(full_calendar)} 天 (新增 {len(valid_new_dates)} 天)")

        # 6. 复制现有数据到输出目录 (incremental 模式)
        if self.mode == 'incremental':
            logger.info("复制现有股票数据到输出目录 ...")
            for sym in existing_symbols:
                self.writer.copy_stock_from_source(sym, self.cn_data_dir)

        # 7. 处理每只股票
        updated_count = 0
        new_count = 0

        for symbol, records in new_data_by_symbol.items():
            # 按日期排序
            records.sort(key=lambda r: r['date'])
            dates_in_records = [r['date'] for r in records]

            if symbol in existing_symbols and self.mode == 'incremental':
                # 已有股票：追加新数据
                base_price = self.reader.read_base_price(symbol)
                if base_price is None:
                    logger.warning(f"无法读取 {symbol} 的 base_price，跳过")
                    continue

                # 构建新数据 DataFrame
                raw_df = pd.DataFrame(records)

                # 对齐到新日期范围 (仅新日期)
                new_cal_dates = [d for d in dates_in_records if d not in existing_cal_set]
                if not new_cal_dates:
                    continue

                # 归一化
                features = self.normalizer.normalize(raw_df, base_price)

                # 计算追加时的 change (需要前一个 close)
                _, existing_close = self.reader.read_bin(symbol, 'close')
                if existing_close is not None and len(existing_close) > 0:
                    last_close = existing_close[-1]
                    if not np.isnan(last_close) and len(features['close']) > 0:
                        first_new_close = features['close'][0]
                        if not np.isnan(first_new_close) and last_close != 0:
                            features['change'][0] = (first_new_close - last_close) / last_close

                # 追加到 bin 文件
                for feat in FEATURES:
                    self.writer.write_bin_append(symbol, feat, features[feat])

                # 更新 instruments 中已有股票的 end_date
                last_new_date = new_cal_dates[-1]
                if symbol in existing_instruments:
                    old_start, old_end = existing_instruments[symbol]
                    if last_new_date > old_end:
                        existing_instruments[symbol] = (old_start, last_new_date)

                updated_count += 1
            else:
                # 新股票：创建完整 bin 文件
                raw_df = pd.DataFrame(records)
                base_price = self.normalizer.compute_base_price(raw_df['adj_close'].values)
                if base_price is None:
                    logger.warning(f"无法计算 {symbol} 的 base_price，跳过")
                    continue

                features = self.normalizer.normalize(raw_df, base_price)

                # 确定在完整日历中的起始索引
                first_date = dates_in_records[0]
                start_idx = calendar_to_idx.get(first_date, 0)

                for feat in FEATURES:
                    self.writer.write_bin_new(symbol, feat, start_idx, features[feat])

                # 更新 instruments
                last_date = dates_in_records[-1]
                existing_instruments[symbol] = (first_date, last_date)
                new_count += 1

        logger.info(f"更新: {updated_count} 只已有股票, {new_count} 只新股票")

        # 8. 写入日历和股票列表
        self.writer.write_calendar(full_calendar)
        self.writer.write_instruments(existing_instruments)

        # 9. 更新 day_future.txt (全年交易日历)
        self._update_day_future()

        # 10. 更新指数成分股文件
        self._update_index_instruments()

        logger.info(f"完成! 输出目录: {self.output_dir}")

    def _copy_existing_data(self, symbols):
        """复制所有现有股票数据到输出目录"""
        for sym in symbols:
            self.writer.copy_stock_from_source(sym, self.cn_data_dir)

    def _update_day_future(self):
        """获取从 cn_data 首日到年底的完整交易日历并写入 day_future.txt"""
        existing_cal = self.reader.read_calendar()
        first_date = existing_cal[0]  # e.g. '2000-01-04'
        year = datetime.now().year
        logger.info(f"获取交易日历: {first_date} ~ {year}-12-31 ...")
        dates_compact = self.client.get_trade_calendar(
            self._date_to_compact(first_date),
            f'{year}1231'
        )
        dates = [self._compact_to_date(d) for d in dates_compact]
        self.writer.write_day_future(dates)

    INDEX_MAPPING = {
        'csi300': '399300.SZ',
        'csi500': '000905.SH',
        'csi800': '000906.SH',
        'csi1000': '000852.SH',
        'csiall': '000985.SH',
    }

    def _update_index_instruments(self):
        """获取指数成分股权重并写入 instruments/csi*.txt"""
        # 复制现有指数成分股文件作为基础
        src_instruments = self.cn_data_dir / 'instruments'
        for index_name in self.INDEX_MAPPING:
            src_file = src_instruments / f'{index_name}.txt'
            if src_file.exists():
                shutil.copy2(str(src_file), str(self.output_dir / 'instruments' / f'{index_name}.txt'))

        # 从 API 获取最新成分股权重并更新
        for index_name, ts_code in self.INDEX_MAPPING.items():
            logger.info(f"更新指数成分股: {index_name} ({ts_code}) ...")
            try:
                df = self.client.get_index_weight(
                    ts_code,
                    self._date_to_compact(self.end_date),
                    self._date_to_compact(self.end_date)
                )
                if df.empty:
                    logger.info(f"  {index_name} 今日无成分股数据")
                    continue

                # 将新数据转为 (symbol_upper, trade_date, trade_date) 格式
                new_records = {}
                for _, row in df.iterrows():
                    sym = self.normalizer.ts_code_to_symbol_upper(row['con_code'])
                    trade_date = self._compact_to_date(row['in_date'])
                    if sym not in new_records or trade_date > new_records[sym]:
                        new_records[sym] = trade_date

                # 读取现有文件并更新
                inst_file = self.output_dir / 'instruments' / f'{index_name}.txt'
                existing = {}
                if inst_file.exists():
                    with open(inst_file) as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                sym, start, end = parts[0], parts[1], parts[2]
                                if sym not in existing or end > existing[sym][1]:
                                    existing[sym] = (start, end)

                # 合并: 更新已有股票的 end_date，添加新股票
                for sym, trade_date in new_records.items():
                    if sym in existing:
                        old_start, old_end = existing[sym]
                        if trade_date > old_end:
                            existing[sym] = (old_start, trade_date)
                    else:
                        existing[sym] = (trade_date, trade_date)

                # 写回
                records = [(sym, start, end) for sym, (start, end) in existing.items()]
                self.writer.write_index_instruments(index_name, records)

            except Exception as e:
                logger.warning(f"获取 {index_name} 成分股失败: {e}")


# ============================================================
# CLI 入口
# ============================================================

def update(cn_data_dir='./cn_data', output_dir='./cn_data_update',
           tushare_token=None, start_date=None, end_date=None,
           mode='incremental', rate_limit=RATE_LIMIT, server=None):
    """
    通过 Tushare API 补充 cn_data 缺失数据。

    Args:
        cn_data_dir: 现有 cn_data 目录路径
        output_dir: 输出目录路径
        tushare_token: Tushare API token (也可通过 TUSHARE_TOKEN 环境变量设置)
        start_date: 更新起始日期 (YYYY-MM-DD)，默认自动检测
        end_date: 更新截止日期 (YYYY-MM-DD)，默认今天
        mode: 'incremental' 增量更新 或 'full' 全量刷新
        rate_limit: API 调用间隔秒数
        server: 自定义 Tushare API 服务器地址 (如山西证券: http://221.204.19.233:7173/dataapi)
    """
    token = tushare_token or os.environ.get('TUSHARE_TOKEN') or os.environ.get('TUSHARE')
    if not token:
        logger.error("请提供 Tushare token: --tushare_token=YOUR_TOKEN 或设置 TUSHARE_TOKEN 环境变量")
        sys.exit(1)

    orchestrator = UpdateOrchestrator(
        cn_data_dir=cn_data_dir,
        output_dir=output_dir,
        tushare_token=token,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        rate_limit=rate_limit,
        server=server,
    )
    orchestrator.run()


if __name__ == '__main__':
    fire.Fire(update)
