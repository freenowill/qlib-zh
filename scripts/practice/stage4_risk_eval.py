#!/usr/bin/env python3
"""
stage4_risk_eval.py
对初筛的股票进行：
    1. 财务标签  (优 / 良 / 中 / 差)
    2. 估值标签  (高 / 中 / 低)
    3. 风险标签  (高 / 中 / 低)
    4. 额外输出连续化的 financial_score / valuation_score / risk_score，
       更接近机构实盘里“先约束、再排序”的做法
数据源：baostock 公共接口
说明：
  - 利润、偿债、营运、成长、杜邦、业绩快报用于财务评分
  - 估值指标采用历史日频中的 PE/PB/PS
  - 近期负面公告数量采用业绩预告/业绩快报中的负面关键词与负增长代理
输出: <output>/risk_eval.csv
"""
from __future__ import annotations

import argparse
import importlib
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _bs():
    try:
        return importlib.import_module("baostock")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"baostock 未安装或无法导入: {exc}") from exc


NEGATIVE_KEYWORDS = ("预亏", "预减", "亏损", "下滑", "下降", "减少", "低于", "转亏", "恶化")


@dataclass
class Quarter:
    year: int
    quarter: int


def _to_6digit(code: str) -> str:
    code = str(code).strip()
    if code.count(".") == 1:
        left, right = code.split(".")
        code = right if right.isdigit() else left
    if code.upper().startswith(("SH", "SZ")):
        code = code[2:]
    return code.zfill(6)


def _to_bs_code(code: str) -> str:
    code = _to_6digit(code)
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}.{code}"


def _quarter_of_date(pred_date: str) -> Quarter:
    dt = pd.to_datetime(pred_date)
    # 使用最近已披露的完整季度
    if dt.month <= 4:
        return Quarter(dt.year - 1, 4)
    if dt.month <= 7:
        return Quarter(dt.year, 1)
    if dt.month <= 10:
        return Quarter(dt.year, 2)
    return Quarter(dt.year, 3)


def _score_bucket(value: float, thresholds: list[tuple[float, int]], reverse: bool = False) -> int:
    if pd.isna(value):
        return 50
    val = float(value)
    if reverse:
        thresholds = list(reversed(thresholds))
    for bound, score in thresholds:
        if (not reverse and val >= bound) or (reverse and val <= bound):
            return score
    return thresholds[-1][1]


def _percentify(value):
    if pd.isna(value):
        return np.nan
    v = float(value)
    if abs(v) <= 1.5:
        return v * 100
    return v


def _financial_label_from_score(score: float) -> str:
    if score >= 80:
        return "优"
    if score >= 65:
        return "良"
    if score >= 50:
        return "中"
    return "差"


def _valuation_label(pe: float, pb: float, ps: float) -> str:
    high = 0
    low = 0
    total = 0
    for value, high_th, low_th in [(pe, 60, 15), (pb, 8, 1.5), (ps, 10, 2.5)]:
        if pd.notna(value):
            total += 1
            if value > high_th:
                high += 1
            elif value < low_th:
                low += 1
    if total == 0:
        return "中"
    if high / total >= 0.5:
        return "高"
    if low / total >= 0.5:
        return "低"
    return "中"


def _cross_sectional_valuation_score(frame: pd.DataFrame, industry_col: str = "industry") -> pd.Series:
    """Cross-sectional valuation score: higher is better / cheaper.

    Uses industry-relative percentiles when an industry group is large enough;
    otherwise falls back to the full candidate pool.
    """
    if frame.empty:
        return pd.Series(dtype=float)

    def _score_one(group: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=group.index, dtype=float)
        total_weight = 0.0
        # Lower is better for valuation multiples.
        for col, weight in [("pe_ttm", 0.40), ("pb", 0.35), ("ps", 0.25)]:
            s = pd.to_numeric(group.get(col), errors="coerce")
            if s.notna().sum() < 3:
                continue
            pct = s.rank(pct=True, method="average")
            # Cheap -> higher score.
            comp = (1.0 - pct).clip(0.0, 1.0).fillna(0.5)
            score = score + comp * weight
            total_weight += weight
        if total_weight <= 0:
            return pd.Series(50.0, index=group.index, dtype=float)
        return (score / total_weight * 100).clip(0.0, 100.0)

    global_score = _score_one(frame)
    out = global_score.copy()
    if industry_col in frame.columns:
        for _, idx in frame.groupby(frame[industry_col].fillna("未知")).groups.items():
            if len(idx) >= 5:
                out.loc[idx] = _score_one(frame.loc[idx])
    return out


def _continuous_risk_score(frame: pd.DataFrame) -> pd.Series:
    """Higher score means higher risk."""
    if frame.empty:
        return pd.Series(dtype=float)

    fin = pd.to_numeric(frame.get("financial_score"), errors="coerce").fillna(50.0)
    val = pd.to_numeric(frame.get("valuation_score"), errors="coerce").fillna(50.0)
    news = pd.to_numeric(frame.get("negative_news_count"), errors="coerce").fillna(0.0).clip(0.0, 3.0)
    is_st = frame.get("is_st", pd.Series(False, index=frame.index)).astype(bool)
    is_suspended = frame.get("is_suspended", pd.Series(False, index=frame.index)).astype(bool)

    fin_penalty = (1.0 - fin / 100.0).clip(0.0, 1.0)
    val_penalty = (1.0 - val / 100.0).clip(0.0, 1.0)
    news_penalty = (news / 3.0).clip(0.0, 1.0)
    st_penalty = is_st.astype(float)
    trade_penalty = is_suspended.astype(float)

    risk = 100.0 * (
        0.35 * fin_penalty
        + 0.25 * val_penalty
        + 0.20 * news_penalty
        + 0.10 * st_penalty
        + 0.10 * trade_penalty
    )
    return risk.clip(0.0, 100.0)


def _risk_label(financial_label: str, valuation_label: str, negative_news_count: int, is_st: bool, is_suspended: bool, tradestatus: bool) -> str:
    score = 0
    if financial_label == "差":
        score += 2
    if valuation_label == "高":
        score += 1
    if negative_news_count >= 3:
        score += 2
    elif negative_news_count == 2:
        score += 1
    if is_st:
        score += 3
    if is_suspended or not tradestatus:
        score += 3
    if score >= 5:
        return "高"
    if score >= 2:
        return "中"
    return "低"


def _login():
    bs = _bs()
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")


def _logout():
    try:
        bs = _bs()
        bs.logout()
    except Exception:
        pass


def _rs_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=getattr(rs, "fields", []))
    return pd.DataFrame(rows, columns=rs.fields)


def _latest_daily_snapshot(code: str, pred_date: str) -> dict:
    bs = _bs()
    bs_code = _to_bs_code(code)
    start_date = (pd.to_datetime(pred_date) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,code,close,peTTM,pbMRQ,psTTM,tradestatus,isST",
        start_date=start_date,
        end_date=pred_date,
        frequency="d",
        adjustflag="3",
    )
    df = _rs_to_df(rs)
    if df.empty:
        return {
            "close": np.nan,
            "pe_ttm": np.nan,
            "pb": np.nan,
            "ps": np.nan,
            "tradestatus": True,
            "is_st": False,
        }
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    last = df.iloc[-1]
    return {
        "close": pd.to_numeric(last.get("close"), errors="coerce"),
        "pe_ttm": pd.to_numeric(last.get("peTTM"), errors="coerce"),
        "pb": pd.to_numeric(last.get("pbMRQ"), errors="coerce"),
        "ps": pd.to_numeric(last.get("psTTM"), errors="coerce"),
        "tradestatus": str(last.get("tradestatus", "1")) == "1",
        "is_st": str(last.get("isST", "0")) == "1",
    }


def _latest_stock_basic(code: str) -> dict:
    bs = _bs()
    rs = bs.query_stock_basic(code=_to_bs_code(code))
    df = _rs_to_df(rs)
    if df.empty:
        return {"ipo_date": pd.NaT}
    row = df.iloc[0]
    return {"ipo_date": pd.to_datetime(row.get("ipoDate"), errors="coerce")}


def _latest_stock_industry(code: str) -> str:
    bs = _bs()
    bs_code = _to_bs_code(code)
    query = getattr(bs, "query_stock_industry", None)
    if query is None:
        return "未知"
    try:
        rs = query(code=bs_code)
        df = _rs_to_df(rs)
        if df.empty:
            return "未知"
        row = df.iloc[-1]
        for col in ["industry", "industryName", "industry_type", "industryType", "classify"]:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
        if len(row) > 0:
            return str(row.iloc[-1]).strip() or "未知"
    except Exception:
        pass
    return "未知"


def _latest_financial_bundle(code: str, q: Quarter, pred_date: str) -> dict:
    bs = _bs()
    bs_code = _to_bs_code(code)
    out = {
        "roeAvg": np.nan,
        "npMargin": np.nan,
        "gpMargin": np.nan,
        "netProfit": np.nan,
        "currentRatio": np.nan,
        "quickRatio": np.nan,
        "cashRatio": np.nan,
        "liabilityToAsset": np.nan,
        "assetToEquity": np.nan,
        "NRTurnRatio": np.nan,
        "INVTurnRatio": np.nan,
        "CATurnRatio": np.nan,
        "AssetTurnRatio": np.nan,
        "YOYEquity": np.nan,
        "YOYAsset": np.nan,
        "YOYNI": np.nan,
        "YOYEPSBasic": np.nan,
        "YOYPNI": np.nan,
        "dupontROE": np.nan,
        "dupontAssetTurn": np.nan,
        "performanceExpressROEWa": np.nan,
        "performanceExpressGRYOY": np.nan,
        "performanceExpressOPYOY": np.nan,
        "performanceExpressEPSChgPct": np.nan,
        "negative_news_count": 0,
    }

    def _first_row(rs):
        df = _rs_to_df(rs)
        if df.empty:
            return None
        return df.iloc[-1]

    row = _first_row(bs.query_profit_data(bs_code, q.year, q.quarter))
    if row is not None:
        for k in ["roeAvg", "npMargin", "gpMargin", "netProfit"]:
            out[k] = pd.to_numeric(row.get(k), errors="coerce")

    row = _first_row(bs.query_balance_data(bs_code, q.year, q.quarter))
    if row is not None:
        for k in ["currentRatio", "quickRatio", "cashRatio", "liabilityToAsset", "assetToEquity"]:
            out[k] = pd.to_numeric(row.get(k), errors="coerce")

    row = _first_row(bs.query_operation_data(bs_code, q.year, q.quarter))
    if row is not None:
        for k in ["NRTurnRatio", "INVTurnRatio", "CATurnRatio", "AssetTurnRatio"]:
            out[k] = pd.to_numeric(row.get(k), errors="coerce")

    row = _first_row(bs.query_growth_data(bs_code, q.year, q.quarter))
    if row is not None:
        for k in ["YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI"]:
            out[k] = pd.to_numeric(row.get(k), errors="coerce")

    row = _first_row(bs.query_dupont_data(bs_code, q.year, q.quarter))
    if row is not None:
        for k in ["dupontROE", "dupontAssetTurn"]:
            out[k] = pd.to_numeric(row.get(k), errors="coerce")

    row = _first_row(bs.query_performance_express_report(bs_code, start_date=f"{q.year}-01-01", end_date=f"{q.year}-12-31"))
    if row is not None:
        for k in ["performanceExpressROEWa", "performanceExpressGRYOY", "performanceExpressOPYOY", "performanceExpressEPSChgPct"]:
            out[k] = pd.to_numeric(row.get(k), errors="coerce")

    # 负面公告 proxy：业绩预告 + 业绩快报
    ref_dt = pd.to_datetime(pred_date)
    start_30 = (ref_dt - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end_30 = ref_dt.strftime("%Y-%m-%d")
    neg_cnt = 0
    for rs in [
        bs.query_forecast_report(bs_code, start_date=start_30, end_date=end_30),
        bs.query_performance_express_report(bs_code, start_date=start_30, end_date=end_30),
    ]:
        df = _rs_to_df(rs)
        if df.empty:
            continue
        for _, r in df.iterrows():
            text = " ".join(str(v) for v in r.tolist())
            if any(k in text for k in NEGATIVE_KEYWORDS):
                neg_cnt += 1
            # perf report numeric negative proxy
            for fld in ["performanceExpressGRYOY", "performanceExpressOPYOY", "performanceExpressEPSChgPct", "profitForcastChgPctDwn"]:
                if fld in df.columns:
                    val = pd.to_numeric(r.get(fld), errors="coerce")
                    if pd.notna(val) and val < 0:
                        neg_cnt += 1
                        break
    out["negative_news_count"] = int(neg_cnt)
    return out


def _financial_score(bundle: dict) -> tuple[float, dict[str, float]]:
    prof = np.nanmean([
        _score_bucket(_percentify(bundle.get("roeAvg")), [(15, 100), (10, 80), (5, 60), (0, 40)], reverse=False),
        _score_bucket(_percentify(bundle.get("npMargin")), [(20, 100), (10, 80), (5, 60), (0, 40)], reverse=False),
        _score_bucket(_percentify(bundle.get("gpMargin")), [(30, 100), (20, 80), (10, 60), (0, 40)], reverse=False),
    ])
    solv = np.nanmean([
        _score_bucket(bundle.get("liabilityToAsset"), [(0.3, 100), (0.5, 80), (0.7, 60), (0.85, 40)], reverse=True),
        _score_bucket(bundle.get("assetToEquity"), [(1, 100), (2, 80), (3, 60), (5, 40)], reverse=True),
        _score_bucket(bundle.get("currentRatio"), [(2, 100), (1.5, 80), (1, 60), (0.8, 40)], reverse=False),
    ])
    oper = np.nanmean([
        _score_bucket(bundle.get("AssetTurnRatio"), [(1, 100), (0.5, 80), (0.2, 60), (0.1, 40)], reverse=False),
        _score_bucket(bundle.get("NRTurnRatio"), [(5, 100), (2, 80), (1, 60), (0.5, 40)], reverse=False),
        _score_bucket(bundle.get("CATurnRatio"), [(1, 100), (0.5, 80), (0.2, 60), (0.1, 40)], reverse=False),
    ])
    grow = np.nanmean([
        _score_bucket(_percentify(bundle.get("YOYNI")), [(20, 100), (10, 80), (0, 60), (-10, 40)], reverse=False),
        _score_bucket(_percentify(bundle.get("YOYEPSBasic")), [(20, 100), (10, 80), (0, 60), (-10, 40)], reverse=False),
        _score_bucket(_percentify(bundle.get("YOYPNI")), [(20, 100), (10, 80), (0, 60), (-10, 40)], reverse=False),
        _score_bucket(_percentify(bundle.get("performanceExpressGRYOY")), [(20, 100), (10, 80), (0, 60), (-10, 40)], reverse=False),
        _score_bucket(_percentify(bundle.get("performanceExpressOPYOY")), [(20, 100), (10, 80), (0, 60), (-10, 40)], reverse=False),
    ])
    score = float(np.nanmean([prof, solv, oper, grow]))
    return score, {"prof": prof, "solv": solv, "oper": oper, "grow": grow}


def _build_fallback_result(pool: pd.DataFrame, pred_date: str) -> pd.DataFrame:
    """Build a neutral risk-eval table when baostock is temporarily unavailable."""
    result = pool.copy()
    if "pred_date" not in result.columns:
        result["pred_date"] = pred_date
    if "date" not in result.columns:
        result["date"] = pred_date

    defaults = {
        "ipo_date": pd.NaT,
        "close": np.nan,
        "pe_ttm": np.nan,
        "pb": np.nan,
        "ps": np.nan,
        "industry": "未知",
        "financial_score": 50.0,
        "financial_label": "中",
        "valuation_score": 50.0,
        "valuation_label": "中",
        "negative_news_count": 0,
        "risk_score": 35.0,
        "risk_label": "中",
        "is_st": False,
        "is_suspended": False,
        "prof_score": 50.0,
        "solv_score": 50.0,
        "oper_score": 50.0,
        "grow_score": 50.0,
    }
    for col, default in defaults.items():
        if col not in result.columns:
            result[col] = default
        else:
            result[col] = result[col].where(result[col].notna(), default)

    result["is_st"] = pd.Series(result.get("is_st", False), index=result.index).fillna(False).astype(bool)
    result["is_suspended"] = pd.Series(result.get("is_suspended", False), index=result.index).fillna(False).astype(bool)
    for col in [
        "financial_score",
        "valuation_score",
        "risk_score",
        "negative_news_count",
        "prof_score",
        "solv_score",
        "oper_score",
        "grow_score",
    ]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def risk_eval(input_csv: str, output_dir: str, pred_date: str):
    login_error = None
    try:
        _login()
    except Exception as exc:
        login_error = exc
        print(f"⚠ baostock 登录失败，stage4 切换到本地降级模式: {exc}")

    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        pool = pd.read_csv(input_csv)
        if "pred_date" in pool.columns and not pool["pred_date"].dropna().empty:
            file_pred_date = str(pool["pred_date"].dropna().astype(str).iloc[0])
            if file_pred_date != pred_date:
                print(f"  ⚠ 输入 pred_date={pred_date} 与初筛文件中的 pred_date={file_pred_date} 不一致，已自动对齐为后者")
                pred_date = file_pred_date
        if "code" not in pool.columns and "instrument" in pool.columns:
            pool["code"] = pool["instrument"].astype(str)
        pool["code"] = pool["code"].astype(str).map(_to_6digit)
        print(f"✓ 加载初筛结果: {len(pool)} 只股票")

        if login_error is not None:
            result = _build_fallback_result(pool, pred_date)
            out_csv = out_path / "risk_eval.csv"
            result.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print("✓ 已输出 stage4 降级结果（中性财务/估值/风险标签）")
            print(f"✓ 风险评估保存: {out_csv}")
            return

        q = _quarter_of_date(pred_date)
        rows = []
        t0 = time.perf_counter()
        for i, row in pool.iterrows():
            code = str(row["code"]).zfill(6)
            print(f"  [{i+1}/{len(pool)}] 评估 {code}...")

            daily = _latest_daily_snapshot(code, pred_date)
            basic = _latest_stock_basic(code)
            industry = _latest_stock_industry(code)
            bundle = _latest_financial_bundle(code, q, pred_date)
            fin_score, dims = _financial_score(bundle)
            financial_label = _financial_label_from_score(fin_score)
            valuation_label = _valuation_label(daily["pe_ttm"], daily["pb"], daily["ps"])
            risk_label = _risk_label(
                financial_label,
                valuation_label,
                int(bundle.get("negative_news_count", 0) or 0),
                bool(daily.get("is_st", False)),
                not bool(daily.get("tradestatus", True)),
                bool(daily.get("tradestatus", True)),
            )

            rows.append({
                **row.to_dict(),
                "ipo_date": basic.get("ipo_date", pd.NaT),
                "close": daily["close"],
                "pe_ttm": daily["pe_ttm"],
                "pb": daily["pb"],
                "ps": daily["ps"],
                "industry": industry,
                "financial_score": round(fin_score, 2) if pd.notna(fin_score) else "",
                "financial_label": financial_label,
                "valuation_label": valuation_label,
                "negative_news_count": int(bundle.get("negative_news_count", 0) or 0),
                "risk_label": risk_label,
                "is_st": bool(daily.get("is_st", False)),
                "is_suspended": not bool(daily.get("tradestatus", True)),
                "prof_score": round(dims["prof"], 2) if pd.notna(dims["prof"]) else "",
                "solv_score": round(dims["solv"], 2) if pd.notna(dims["solv"]) else "",
                "oper_score": round(dims["oper"], 2) if pd.notna(dims["oper"]) else "",
                "grow_score": round(dims["grow"], 2) if pd.notna(dims["grow"]) else "",
            })

        result = pd.DataFrame(rows)
        if result.empty:
            out_csv = out_path / "risk_eval.csv"
            result.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"\n⚠ 风险评估结果为空，已保存空表: {out_csv}")
            return

        for col in ["financial_score", "pe_ttm", "pb", "ps", "negative_news_count"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
        result["is_st"] = pd.Series(result.get("is_st", False), index=result.index).fillna(False).astype(bool)
        result["is_suspended"] = pd.Series(result.get("is_suspended", False), index=result.index).fillna(False).astype(bool)

        # Institutional-style continuous scores: cheap + quality + news + tradability.
        result["valuation_score"] = _cross_sectional_valuation_score(result)
        result["risk_score"] = _continuous_risk_score(result)

        # Keep legacy labels for downstream stage5 compatibility.
        result["valuation_label"] = np.where(
            result["valuation_score"] >= 70,
            "低",
            np.where(result["valuation_score"] <= 30, "高", "中"),
        )
        result["risk_label"] = np.where(
            result["risk_score"] >= 60,
            "高",
            np.where(result["risk_score"] >= 30, "中", "低"),
        )

        out_csv = out_path / "risk_eval.csv"
        result.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"\n✓ 风险评估耗时 {time.perf_counter() - t0:.1f}s")
        print("\n◆ 风险评估完毕:")
        display_cols = [c for c in ["code", "financial_label", "valuation_label", "risk_label", "financial_score", "valuation_score", "risk_score", "negative_news_count", "pe_ttm", "pb", "ps"] if c in result.columns]
        print(result[display_cols].to_string(index=False))
        print(f"\n✓ 风险评估保存: {out_csv}")
    finally:
        _logout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pred-date", required=True, dest="pred_date")
    args = ap.parse_args()
    risk_eval(args.input, args.output, args.pred_date)


if __name__ == "__main__":
    main()
