#!/usr/bin/env python3
"""Stage2 runner for Alpha158 practice with explicit time-decay reweighting.

This script mirrors the Qlib qrun workflow but instantiates a custom
TimeDecayReweighter so we can tune the half-life without modifying the
installed qlib package.
"""
from __future__ import annotations

import argparse
import inspect
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qlib.config import C
from qlib.constant import REG_CN
from qlib.data.dataset import Dataset, TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import auto_filter_kwargs, fill_placeholder, flatten_dict, init_instance_by_config
from qlib.workflow import R

from time_decay_reweighter import TimeDecayReweighter

logger = get_module_logger("stage2_practice")


def _materialize_label_data(label_obj):
    if isinstance(label_obj, (pd.Series, pd.DataFrame)):
        return label_obj
    if isinstance(label_obj, TSDataSampler):
        values = []
        for idx in range(len(label_obj)):
            arr = np.asarray(label_obj[idx])
            if arr.ndim == 0:
                values.append(float(arr))
            elif arr.ndim == 1:
                values.append(float(arr[-1]))
            else:
                values.append(float(arr[-1, -1]))
        return pd.DataFrame({"label": values}, index=label_obj.get_index())
    raise TypeError(f"Unsupported label object type: {type(label_obj)!r}")


def _predict_for_segment(model: Model, dataset: Dataset, segment: str):
    try:
        signature = inspect.signature(model.predict)
        if "segment" in signature.parameters:
            return model.predict(dataset, segment=segment)
    except (TypeError, ValueError):
        pass

    original_segments = getattr(dataset, "segments", {}).copy()
    if segment not in original_segments:
        raise KeyError(f"Dataset segment not found: {segment}")

    patched_segments = original_segments.copy()
    patched_segments["test"] = original_segments[segment]
    try:
        dataset.config(segments=patched_segments)
        return model.predict(dataset)
    finally:
        dataset.config(segments=original_segments)


def _save_split_predictions(model: Model, dataset: Dataset) -> None:
    valid_pred = _predict_for_segment(model, dataset, "valid")
    valid_label = _materialize_label_data(dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L))
    test_pred = _predict_for_segment(model, dataset, "test")
    test_label = _materialize_label_data(dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L))

    R.save_objects(
        **{
            "valid_pred.pkl": valid_pred,
            "valid_label.pkl": valid_label,
            "test_pred_snapshot.pkl": test_pred,
            "test_label_snapshot.pkl": test_label,
        }
    )

def _coerce_scalar(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(ch in text.lower() for ch in [".", "e"]):
            return float(text)
        return int(text)
    except ValueError:
        return value


def _normalize_config_types(obj):
    if isinstance(obj, dict):
        return {k: _normalize_config_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_config_types(v) for v in obj]
    return _coerce_scalar(obj)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return _normalize_config_types(yaml.safe_load(f))


def _init_qlib(qlib_init: dict[str, Any], uri_folder: str) -> None:
    import qlib

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
    qlib.init(**qlib_init, exp_manager=exp_manager)


def _log_task_info(task_config: dict[str, Any]) -> None:
    R.log_params(**flatten_dict(task_config))
    R.save_objects(**{"task": task_config})
    R.set_tags(**{"hostname": os.uname().nodename})


def _load_warm_start_checkpoint(path: str | None):
    if not path:
        return None
    ckpt = Path(path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {ckpt}")
    with open(ckpt, "rb") as f:
        return pickle.load(f)


def _build_fit_kwargs(model: Model, warm_start_model, reweighter: TimeDecayReweighter) -> dict[str, Any]:
    fit_kwargs: dict[str, Any] = {"reweighter": reweighter}
    if warm_start_model is None:
        return fit_kwargs

    warm_inner_model = getattr(warm_start_model, "model", None)
    if warm_inner_model is None:
        return fit_kwargs

    model_name = type(model).__name__
    if model_name == "XGBModel":
        fit_kwargs["xgb_model"] = warm_inner_model
    elif model_name == "LGBModel":
        fit_kwargs["init_model"] = warm_inner_model
    else:
        logger.warning("Warm start is not implemented for model type: %s", model_name)
    return fit_kwargs


def _exe_task(task_config: dict[str, Any], reweighter: TimeDecayReweighter, warm_start_path: str | None = None) -> None:
    rec = R.get_recorder()
    model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
    dataset: Dataset = init_instance_by_config(task_config["dataset"], accept_types=Dataset)
    warm_start_model = _load_warm_start_checkpoint(warm_start_path)
    if warm_start_path:
        logger.info("Warm start checkpoint: %s", warm_start_path)
    else:
        logger.info("Warm start checkpoint: <none>")
    fit_kwargs = _build_fit_kwargs(model, warm_start_model, reweighter)

    auto_filter_kwargs(model.fit)(dataset, **fit_kwargs)
    R.save_objects(**{"params.pkl": model})
    dataset.config(dump_all=False, recursive=True)
    R.save_objects(**{"dataset": dataset})
    _save_split_predictions(model, dataset)

    placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
    task_config = fill_placeholder(task_config, placehorder_value)
    records = task_config.get("record", [])
    if isinstance(records, dict):
        records = [records]
    for record in records:
        r = init_instance_by_config(
            record,
            recorder=rec,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": dataset},
        )
        r.generate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Generated practice YAML")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--uri-folder", default="mlruns")
    ap.add_argument("--half-life", type=int, default=252)
    ap.add_argument("--floor", type=float, default=0.2)
    ap.add_argument("--warm-start", default=None, help="Previous fold params.pkl checkpoint for warm start")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    qlib_init = cfg.get("qlib_init", {})
    task = cfg.get("task", {})

    if args.half_life <= 0:
        raise ValueError("half-life must be positive")
    if not (0 < args.floor <= 1):
        raise ValueError("floor must be in (0, 1]")

    _init_qlib(qlib_init, args.uri_folder)

    reweighter = TimeDecayReweighter(half_life=args.half_life, floor=args.floor)

    with R.start(experiment_name=args.experiment_name):
        _log_task_info(task)
        _exe_task(task, reweighter=reweighter, warm_start_path=args.warm_start)


if __name__ == "__main__":
    main()
