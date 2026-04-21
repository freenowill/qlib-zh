#!/usr/bin/env python3
"""Small-cap stage2 wrapper.

Reuses the practice walk-forward engine, but swaps in CSI1000-specific model
templates and supports limiting folds through `STAGE2_MAX_FOLDS` for smoke runs.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_practice_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "practice" / "run_stage2_walk_forward.py"
    spec = importlib.util.spec_from_file_location("run_stage2_walk_forward_practice", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_practice_module()
    root = Path(__file__).resolve().parents[2]
    template_root = root / "scripts" / "small" / "templates"
    module.MODEL_SPECS = [
        {
            "name": "xgboost",
            "template": template_root / "workflow_config_xgboost_Alpha158_csi1000.yaml",
            "model_mode": "default",
        },
        {
            "name": "lightgbm",
            "template": template_root / "workflow_config_lightgbm_Alpha158_csi1000.yaml",
            "model_mode": "robust",
        },
    ]

    original_build_fold_dates = module._build_fold_dates

    def _build_fold_dates_limited(*args, **kwargs):
        folds = original_build_fold_dates(*args, **kwargs)
        raw_limit = str(os.environ.get("STAGE2_MAX_FOLDS", "0") or "0").strip()
        limit = int(raw_limit) if raw_limit else 0
        if limit > 0:
            return folds[-limit:]
        return folds

    module._build_fold_dates = _build_fold_dates_limited
    module.main()


if __name__ == "__main__":
    main()