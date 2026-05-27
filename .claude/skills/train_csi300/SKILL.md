---
name: train_csi300
description: CSI300 训练+预测流水线。使用 daily_pv_csi300.h5，提取 Alpha158(158) + csi_300.md(19 个独立有效因子)，执行 walk-forward LightGBM 训练和预测。
---

# /train_csi300

运行 CSI300 的完整 train+predict 流水线：
- **H5 数据**: `rdagent_workspace/factor_data_template/daily_pv_csi300.h5`
- **因子文件**: `tushare/csi_300.md`（19 个独立有效因子）
- **Alpha158**: 158 个价量因子（自动包含）
- **总特征数**: 158 + 19 = 177
- **模型**: LightGBM (walk-forward, 5 folds)

## 前置条件

- H5 文件: `rdagent_workspace/factor_data_template/daily_pv_csi300.h5`
  - 如果不存在，先运行 `/gen_tushare_h5 csi300` 生成
- Docker 镜像: `zhuhai123/qlib-rdagent:v1`（或其他通过 `DOCKER_IMAGE` 环境变量指定）

## 用法

```bash
# 默认实验名
/train_csi300

# 指定实验名
/train_csi300 <experiment_name>
```

内部执行:
```bash
PRACTICE_FACTOR_FILE=tushare/csi_300.md \
TARGET_MARKET=csi300 \
TARGET_BENCHMARK=SH000300 \
bash run_new_factor_practice ${exp_name:-csi300_train} --force-stage0
```

## 输出

- **IC 分析**: `DATA/analysis_outputs/<exp_name>/model_predict/factor_ic_summary_test.csv`
- **验证集 IC**: `factor_ic_summary_valid.csv`
- **Walk-forward 模型**: `model_predict/walk_forward/` 下各 fold 目录
- **回测报告**: `model_predict/full_backtest/`

## 因子说明

- **Alpha158** (158 个): 9 KBar + 4 Price + 145 Rolling — 纯价量因子, 定义见 `csi_300.md` Part 2
- **独立有效因子** (19 个): 从 42 个实践因子中筛选, 在 CSI300 walk-forward 中方向一致且 `|IR| >= 0.10`, 定义见 `csi_300.md` Part 1
