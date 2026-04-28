#!/usr/bin/env python3
"""
gen_practice_yaml.py
根据动态日期范围，从模板 YAML 生成本次实盘实验专用的 workflow_config.yaml。
"""
import argparse
import re
import sys
import os
from pathlib import Path

import pandas as pd
import yaml


def validate_date_order(dates: dict) -> None:
    """Ensure train/valid/test windows are strictly ordered and non-overlapping."""
    keys = ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]
    missing = [k for k in keys if k not in dates]
    if missing:
        raise ValueError(f"Missing required dates: {missing}")

    parsed = {k: pd.Timestamp(dates[k]) for k in keys}
    for left, right in zip(keys, keys[1:]):
        if parsed[left] >= parsed[right]:
            raise ValueError(
                f"Invalid date order: {left}={parsed[left].date()} must be earlier than {right}={parsed[right].date()}"
            )


def _apply_model_mode(doc: dict, model_mode: str, sample_weight_half_life: int | None = None) -> None:
    """Adjust LightGBM hyperparameters for a given stage2 regime."""
    if model_mode == "default":
        return

    model = doc.setdefault("task", {}).setdefault("model", {}).setdefault("kwargs", {})
    if model_mode == "robust":
        # Based on stage2 analysis: positive IC but low IR / high drawdown.
        # Use stronger regularization and lower tree complexity to improve stability.
        model.update(
            {
                "learning_rate": 0.05,
                "num_leaves": 128,
                "max_depth": 6,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
                "lambda_l1": 500.0,
                "lambda_l2": 1000.0,
                "min_data_in_leaf": 64,
            }
        )
    else:
        raise ValueError(f"Unknown model_mode: {model_mode}")


def patch_yaml(
    template_path: str,
    output_path: str,
    dates: dict,
    model_mode: str = "default",
    data_start: str | None = None,
    sample_weight_half_life: int | None = None,
) -> None:
    validate_date_order(dates)

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = yaml.safe_load(content)

    # ──────────────────────────────────────────
    # 1. 更新 data_handler_config
    # ──────────────────────────────────────────
    dh = doc.get("data_handler_config", {})
    handler_start = data_start or dates["train_start"]
    # handler 的全局时间范围覆盖训练+验证+测试
    dh["start_time"]     = handler_start
    dh["end_time"]       = dates["test_end"]
    dh["fit_start_time"] = dates["train_start"]
    dh["fit_end_time"]   = dates["train_end"]

    _apply_model_mode(doc, model_mode, sample_weight_half_life=sample_weight_half_life)

    # ──────────────────────────────────────────
    # 2. 更新 dataset segments
    # ──────────────────────────────────────────
    segments = doc["task"]["dataset"]["kwargs"]["segments"]
    segments["train"] = [dates["train_start"], dates["train_end"]]
    segments["valid"] = [dates["valid_start"], dates["valid_end"]]
    segments["test"]  = [dates["test_start"],  dates["test_end"]]

    # 更新 handler 引用（内联 kwargs 方式兼容）
    hkw = doc["task"]["dataset"]["kwargs"]["handler"]
    if isinstance(hkw, dict) and "kwargs" in hkw:
        hkw["kwargs"].update(
            start_time=handler_start,
            end_time=dates["test_end"],
            fit_start_time=dates["train_start"],
            fit_end_time=dates["train_end"],
        )

    # ──────────────────────────────────────────
    # 3. 更新 port_analysis_config backtest 时间
    # ──────────────────────────────────────────
    pa = doc.get("port_analysis_config", {})
    strategy = pa.get("strategy", {})
    strategy_kwargs = strategy.get("kwargs", {})
    hold_num = int(os.getenv("HOLD_NUM", "5") or 5)
    cash_total = float(os.getenv("CASH_TOTAL", "10000") or 10000)
    strategy_kwargs.update(
        {
            "topk": hold_num,
            "n_drop": min(1, hold_num),
            "hold_thresh": 1,
            "only_tradable": True,
        }
    )
    bt = pa.get("backtest", {})
    bt["start_time"] = dates["test_start"]
    bt["end_time"]   = dates["test_end"]
    bt["account"] = cash_total

    # ──────────────────────────────────────────
    # 4. 写出
    # ──────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"✓ YAML 已写出: {output_path}")
    print(f"  train : {dates['train_start']} → {dates['train_end']}")
    print(f"  valid : {dates['valid_start']} → {dates['valid_end']}")
    print(f"  test  : {dates['test_start']}  → {dates['test_end']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template",    required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--train-start", required=True, dest="train_start")
    ap.add_argument("--train-end",   required=True, dest="train_end")
    ap.add_argument("--valid-start", required=True, dest="valid_start")
    ap.add_argument("--valid-end",   required=True, dest="valid_end")
    ap.add_argument("--test-start",  required=True, dest="test_start")
    ap.add_argument("--test-end",    required=True, dest="test_end")
    ap.add_argument("--model-mode", choices=["default", "robust"], default="default", dest="model_mode")
    ap.add_argument("--data-start", default=None, dest="data_start")
    ap.add_argument("--sample-weight-half-life", type=int, default=None, dest="sample_weight_half_life")
    args = ap.parse_args()

    sample_weight_half_life = args.sample_weight_half_life
    if sample_weight_half_life is None:
        env_half_life = os.getenv("SAMPLE_WEIGHT_HALF_LIFE", "").strip()
        sample_weight_half_life = int(env_half_life) if env_half_life else None

    if sample_weight_half_life is not None and sample_weight_half_life <= 0:
        raise ValueError("sample_weight_half_life must be positive")

    patch_yaml(
        args.template,
        args.output,
        {
            "train_start": args.train_start,
            "train_end":   args.train_end,
            "valid_start": args.valid_start,
            "valid_end":   args.valid_end,
            "test_start":  args.test_start,
            "test_end":    args.test_end,
        },
        model_mode=args.model_mode,
        data_start=getattr(args, "data_start", None),
        sample_weight_half_life=sample_weight_half_life,
    )


if __name__ == "__main__":
    main()
