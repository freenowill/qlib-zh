#!/usr/bin/env python3
"""
fetch_sw_industry.py — 从 TuShare 获取申万一级行业分类 (SW2021)

通过 index_classify 获取 28 个申万一级行业指数代码，
再通过 index_member 逐一获取成分股，生成 stock→industry 映射文件。

用法:
  python3 tushare/fetch_sw_industry.py
  python3 tushare/fetch_sw_industry.py --output tushare/cn_data/sw_industry.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from api_utils import TushareAPI


def ts_code_to_symbol(ts_code: str) -> str:
    """000001.SZ -> SZ000001"""
    code, suffix = ts_code.split(".")
    return f"{suffix.upper()}{code}"


def fetch_sw_industry(api: TushareAPI) -> pd.DataFrame:
    """获取申万一级行业分类成分股映射。"""

    # Step 1: 获取申万一级行业指数列表
    print("获取申万一级行业指数列表 (index_classify, level=L1, src=SW2021) ...")
    classify = api.query("index_classify", level="L1", src="SW2021")

    if classify.empty:
        print("ERROR: index_classify 返回空，尝试 ths_index 回退 ...")
        return _fetch_ths_fallback(api)

    print(f"  获取到 {len(classify)} 个行业指数:")
    for _, row in classify.iterrows():
        print(f"    {row.get('index_code', '?')}  {row.get('industry_name', '?')}")

    # Step 2: 逐一获取成分股
    all_members = []
    index_codes = classify["index_code"].tolist()

    for i, idx_code in enumerate(index_codes):
        print(f"  [{i+1}/{len(index_codes)}] {idx_code} ...", end=" ", flush=True)
        members = api.query("index_member", index_code=idx_code)
        if members.empty:
            # 部分申万指数代码格式可能有差异，尝试 .SI 后缀
            if not idx_code.endswith(".SI"):
                members = api.query("index_member", index_code=f"{idx_code}.SI")
        if members.empty:
            print(f"空 (无成分股数据)")
            continue
        print(f"{len(members)} 只成分股")
        # 保留 is_new='Y' 的最新成分股
        members["index_code"] = idx_code
        all_members.append(members)

    if not all_members:
        print("ERROR: 所有行业指数成分股均为空")
        return pd.DataFrame()

    members_df = pd.concat(all_members, ignore_index=True)

    # Step 3: 合并行业名称
    industry_names = {}
    for _, row in classify.iterrows():
        industry_names[row["index_code"]] = row.get("industry_name", row["index_code"])

    members_df["industry_name"] = members_df["index_code"].map(industry_names)

    # Step 4: 只保留当前成分股 (is_new='Y' 或 out_date 为空/NaT)
    if "is_new" in members_df.columns:
        members_df = members_df[members_df["is_new"] == "Y"].copy()
    elif "out_date" in members_df.columns:
        members_df["out_date"] = pd.to_datetime(members_df["out_date"], errors="coerce")
        members_df = members_df[members_df["out_date"].isna()].copy()

    # Step 5: 格式转换
    con_col = "con_code" if "con_code" in members_df.columns else "ts_code"
    if con_col not in members_df.columns:
        print(f"ERROR: 无法找到成分股代码列, columns={members_df.columns.tolist()}")
        return pd.DataFrame()

    members_df["symbol"] = members_df[con_col].apply(ts_code_to_symbol)
    members_df = members_df[["symbol", "industry_name"]].drop_duplicates()

    # 一只股票不应该属于多个申万一级行业，如有重复取第一个
    members_df = members_df.drop_duplicates(subset=["symbol"], keep="first")

    print(f"\n总计: {len(members_df)} 只股票, {members_df['industry_name'].nunique()} 个行业")
    print(f"行业分布:")
    for name, count in members_df["industry_name"].value_counts().items():
        print(f"  {name}: {count}")

    return members_df


def _fetch_ths_fallback(api: TushareAPI) -> pd.DataFrame:
    """index_classify 不可用时的回退方案: 使用同花顺行业分类."""
    print("使用同花顺 (THS) 行业分类作为回退 ...")
    ths_index = api.query("ths_index", level="L1")
    if ths_index.empty:
        print("ERROR: ths_index 也返回空")
        return pd.DataFrame()

    print(f"  获取到 {len(ths_index)} 个同花顺一级行业")

    all_members = []
    for i, (_, row) in enumerate(ths_index.iterrows()):
        ts_code = row["ts_code"]
        name = row.get("industry_name", ts_code)
        print(f"  [{i+1}/{len(ths_index)}] {ts_code} {name} ...", end=" ", flush=True)
        members = api.query("ths_member", ts_code=ts_code)
        if members.empty:
            print("空")
            continue
        print(f"{len(members)} 只")
        members["industry_name"] = name
        all_members.append(members)

    if not all_members:
        return pd.DataFrame()

    members_df = pd.concat(all_members, ignore_index=True)
    con_col = "con_code" if "con_code" in members_df.columns else "ts_code"
    members_df["symbol"] = members_df[con_col].apply(ts_code_to_symbol)
    members_df = members_df[["symbol", "industry_name"]].drop_duplicates()
    members_df = members_df.drop_duplicates(subset=["symbol"], keep="first")
    print(f"\n总计: {len(members_df)} 只股票, {members_df['industry_name'].nunique()} 个行业")
    return members_df


def main():
    parser = argparse.ArgumentParser(description="获取申万一级行业分类成分股映射")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "cn_data" / "sw_industry.csv"),
                        help="输出 CSV 文件路径")
    args = parser.parse_args()

    api = TushareAPI()
    df = fetch_sw_industry(api)

    if df.empty:
        print("ERROR: 未获取到任何行业数据")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n已写入: {output_path} ({len(df)} 条记录)")


if __name__ == "__main__":
    main()
