---
name: train_csi1000
description: CSI1000 训练+预测流水线。使用 daily_pv_csi1000.h5，默认提取 csi_1000.md(7 个独立有效因子)+Alpha158(158) = 165 特征；可选 practice_factor.md(42 个实践因子)+Alpha158(158) = 200 特征。执行 walk-forward LightGBM 训练和预测。
---

# /train_csi1000

运行 CSI1000 的完整 train+predict 流水线：
- **H5 数据**: `rdagent_workspace/factor_data_template/daily_pv_csi1000.h5`
- **模型**: LightGBM (walk-forward, 5 folds)

## 用法

```bash
# 默认: 7 个 CSI1000 有效因子 + Alpha158 (165 特征)
/train_csi1000

# 指定实验名 (7 + 158)
/train_csi1000 <experiment_name>

# 使用 42 个实践因子 + Alpha158 (200 特征)
/train_csi1000 <experiment_name> practice_factor.md
```

### 默认模式 (`csi_1000.md`, 165 特征)

```
Alpha158 (158 个价量因子) + 7 个 CSI1000 独立有效因子
```

内部执行:
```bash
PRACTICE_FACTOR_FILE=tushare/csi_1000.md \
TARGET_MARKET=csi1000 \
TARGET_BENCHMARK=SH000852 \
bash run_new_factor_practice ${exp_name:-csi1000_train} --force-stage0
```

### practice_factor.md 模式 (200 特征)

```
Alpha158 (158 个价量因子) + 42 个实践因子 (v3.0)
```

内部执行:
```bash
PRACTICE_FACTOR_FILE=tushare/practice_factor.md \
TARGET_MARKET=csi1000 \
TARGET_BENCHMARK=SH000852 \
bash run_new_factor_practice ${exp_name:-csi1000_train} --force-stage0
```

## 前置条件

- H5 文件: `rdagent_workspace/factor_data_template/daily_pv_csi1000.h5`
  - 如果不存在，先运行 `/gen_tushare_h5 csi1000` 生成
- Docker 镜像: `zhuhai123/qlib-rdagent:v1`（或其他通过 `DOCKER_IMAGE` 环境变量指定）

## 输出

- **IC 分析**: `DATA/analysis_outputs/<exp_name>/model_predict/factor_ic_summary_test.csv`
- **验证集 IC**: `factor_ic_summary_valid.csv`
- **Walk-forward 模型**: `model_predict/walk_forward/` 下各 fold 目录
- **回测报告**: `model_predict/full_backtest/`

## 因子说明

### 默认模式: `csi_1000.md`

| # | 因子名称 | 类型 | 方向 | Test RankIC | Test IR |
|---|----------|------|------|-------------|---------|
| 1 | Size | 市值/规模 | 负向 | -0.075 | -0.643 |
| 2 | avg_normalized_range_5d | 波动率 | 负向 | -0.069 | -0.354 |
| 3 | RealizedVolatility_20d | 波动率 | 负向 | -0.065 | -0.335 |
| 4 | MAX_20d | 行为金融 | 负向 | -0.064 | -0.407 |
| 5 | Turnover | 流动性 | 负向 | -0.064 | -0.370 |
| 6 | Amihud_20d | 非流动性 | 正向 | +0.066 | +0.503 |
| 7 | Sector_Relative_PB | 估值(行业中性) | 负向 | -0.041 | -0.536 |

+ Alpha158 (158 个价量因子: 9 KBar + 4 Price + 145 Rolling)

**总特征数: 165**

### practice_factor.md 模式

42 个实践因子 (v3.0) + Alpha158 (158 个价量因子)

**总特征数: 200**

详见 `tushare/practice_factor.md`。
