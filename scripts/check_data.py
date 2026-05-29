"""
check_data.py - 统计 cn_data 每个日期的数据缺失条数

直接读取 qlib bin 格式文件，不依赖 qlib 初始化，速度快。

用法:
  python check_data.py --cn_data_dir /path/to/cn_data
  python check_data.py --cn_data_dir /path/to/cn_data --feature close
  python check_data.py --cn_data_dir /path/to/cn_data --output missing_by_date.csv
"""

import os
import numpy as np
import pandas as pd
import fire
from pathlib import Path


BIN_SUFFIX = '.day.bin'


def check_missing_by_date(cn_data_dir: str, feature: str = 'close', output: str = None):
    """
    统计 cn_data 每个交易日的数据缺失条数。

    Args:
        cn_data_dir: cn_data 根目录路径
        feature: 用于检查的特征列，默认 close
        output: 可选，输出 CSV 文件路径
    """
    cn_data_dir = Path(cn_data_dir)

    # 1. 读取交易日历
    cal_file = cn_data_dir / 'calendars' / 'day.txt'
    with open(cal_file) as f:
        calendar = [line.strip() for line in f if line.strip()]
    print(f"交易日历: {len(calendar)} 天, {calendar[0]} ~ {calendar[-1]}")

    # 2. 列出所有股票
    features_dir = cn_data_dir / 'features'
    symbols = sorted([d.name for d in features_dir.iterdir() if d.is_dir()])
    print(f"股票数量: {len(symbols)}")

    # 3. 统计每个日期的缺失数
    # missing_count[i] = 第 i 个交易日缺失的股票数
    missing_count = np.zeros(len(calendar), dtype=np.int64)
    # total_count[i] = 第 i 个交易日应有数据的股票数
    total_count = np.zeros(len(calendar), dtype=np.int64)

    for symbol in symbols:
        bin_path = features_dir / symbol / f'{feature}{BIN_SUFFIX}'
        if not bin_path.exists():
            continue

        data = np.fromfile(str(bin_path), dtype='<f4')
        if len(data) < 1:
            continue

        start_idx = int(data[0])
        values = data[1:]
        end_idx = start_idx + len(values)

        # 标记该股票覆盖的日期范围
        total_count[start_idx:end_idx] += 1

        # 统计 NaN
        nan_mask = np.isnan(values)
        missing_count[start_idx:end_idx] += nan_mask.astype(np.int64)

    # 4. 输出结果
    df = pd.DataFrame({
        'date': calendar,
        'total_stocks': total_count,
        'missing_count': missing_count,
    })

    # 添加缺失率
    df['missing_ratio'] = np.where(
        df['total_stocks'] > 0,
        df['missing_count'] / df['total_stocks'],
        0.0
    )

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"数据缺失统计 (特征: {feature})")
    print(f"{'='*60}")
    print(f"总交易日数: {len(df)}")
    print(f"有缺失数据的交易日数: {(df['missing_count'] > 0).sum()}")
    print(f"无缺失数据的交易日数: {(df['missing_count'] == 0).sum()}")
    print(f"\n缺失条数统计:")
    print(f"  最大缺失: {df['missing_count'].max()} 条")
    print(f"  平均缺失: {df['missing_count'].mean():.1f} 条")
    print(f"  中位缺失: {df['missing_count'].median():.1f} 条")

    # 打印缺失最多的前20天
    top_missing = df.nlargest(20, 'missing_count')
    print(f"\n缺失最多的前20个交易日:")
    print(top_missing[['date', 'total_stocks', 'missing_count', 'missing_ratio']].to_string(index=False))

    # 打印最近30天的缺失情况
    print(f"\n最近30个交易日缺失情况:")
    recent = df.tail(30)
    print(recent[['date', 'total_stocks', 'missing_count', 'missing_ratio']].to_string(index=False))

    # 保存完整结果
    if output:
        df.to_csv(output, index=False)
        print(f"\n完整结果已保存到: {output}")



if __name__ == '__main__':
    fire.Fire(check_missing_by_date)
