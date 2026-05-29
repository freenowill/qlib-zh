---
name: train_csi300
description: CSI300 训练+预测流水线。使用 daily_pv_csi300.h5，提取 Alpha158(158)+42 个增强因子=200 个特征，执行 walk-forward LightGBM 训练和预测。
---

# /train_csi300

运行 CSI300 的完整 train+predict 流水线：
- **H5 数据**: `rdagent_workspace/factor_data_template/daily_pv_csi300.h5`
- **因子文件**: `tushare/practice_factor.md`（42 个增强因子）+ Alpha158（158 个因子）= **200 个特征**
- **因子组成**: Alpha158 全量价量因子(158) + 42 个精选增强因子（覆盖波动/估值/质量/成长/行为/流动性/规模/现金流/非流动性/盈余质量等）
- **Walk-forward**: train=7y, valid=2y, test=3y, stride=1y (9 fold history)
- **回测策略**: buffered equal-weight top-K（缓冲带机制，降低换手率）
- **调仓频率**: 周频（每约 5 个交易日调仓一次）
- **持有数量**: 10 支
- **资金**: ¥50,000
- **模型**: LightGBM (robust mode)

> **设计理念**：Alpha158 提供完整的价量特征基础，`practice_factor.md` 的 42 个增强因子系统性地补充基本面、行为金融、非流动性、盈余质量等 Alpha158 缺失的维度。
> 联合使用 Alpha158(158) + practice_factor(42) = 200 个特征，覆盖更全面的 alpha 信号来源。
> 回测方式参考 `run_alpha158_practice`：使用 `buffered_equal_weight_topk` 策略（50% 缓冲带），**周频调仓**，支持 predict_only 增量预测模式。

## 前置条件

- H5 文件: `rdagent_workspace/factor_data_template/daily_pv_csi300.h5`
  - 如果不存在，先运行 `/gen_tushare_h5 csi300` 生成
- Docker 镜像: `zhuhai123/qlib-rdagent:v1`（或其他通过 `DOCKER_IMAGE` 环境变量指定）

## 用法

```bash
# 默认实验名（完整训练+回测）
/train_csi300

# 指定实验名
/train_csi300 <experiment_name>

# 仅预测（不训练，使用已有模型预测最新数据）
/train_csi300 <experiment_name> predict_only=True

# 指定预测日期
/train_csi300 <experiment_name> pred_date=2026-06-01
```

内部执行:
```bash
FULL_BACKTEST_STRATEGY=buffered_equal_weight_topk \
FULL_BACKTEST_BUFFER_PCT=0.5 \
FULL_BACKTEST_REBALANCE_FREQ=weekly \
HOLD_NUM=10 \
CASH_TOTAL=50000 \
PRACTICE_FACTOR_FILE=tushare/practice_factor.md \
TARGET_MARKET=csi300 \
TARGET_BENCHMARK=SH000300 \
bash run_new_factor_practice ${exp_name:-csi300_train} --force-stage0 new_factor_only=false
```

## 输出

- **IC 分析**: `DATA/analysis_outputs/<exp_name>/model_predict/factor_ic_summary_test.csv`
- **验证集 IC**: `factor_ic_summary_valid.csv`
- **Walk-forward 模型**: `model_predict/walk_forward/` 下各 fold 目录
- **回测报告**: `model_predict/full_backtest/`（含 `full_backtest_overview.html` 汇总页面）
- **回测策略**: `buffered_equal_weight_topk` — 等权持有 top-K 股票，跌出前 top-K×2 名后才卖出（50% 缓冲带），有效降低换手率。**周频调仓**: 每周最后一个交易日按最新信号调仓，其余交易日持仓不动。

## 因子说明

使用 **Alpha158（158 个价量因子）** + **practice_factor.md（42 个增强因子）** = **200 个特征**。

| 数据源 | 因子数 | 覆盖类别 |
|--------|--------|---------|
| **Alpha158 KBar** | 9 | 价格形态 (KMID, KLEN, KUP, KLOW, KSFT 等) |
| **Alpha158 Price** | 4 | 价格水平 (OPEN, HIGH, LOW, VWAP) |
| **Alpha158 Rolling** | 145 | 29 类 × [5,10,20,30,60] 窗口 (ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, CORR, SUMP 等) |
| **practice_factor 增强** | 42 | 波动/估值/质量/成长/反转/行为/流动性/规模/现金流/盈余质量/非流动性 |

### 42 个增强因子列表

| # | 因子名称 | 类别 |
|---|----------|------|
| 1 | RealizedVolatility_20d | 波动率 |
| 2 | volume_change_5d | 流动性 |
| 3 | trailing_PE_ratio | 估值 |
| 4 | obv_slope_10d | 量价 |
| 5 | sharpe_10d | 风险调整 |
| 6 | reversal_1d | 反转 |
| 7 | volume_weighted_momentum_5d | 量价 |
| 8 | Delta_roe | 质量 |
| 9 | earnings_yield | 估值 |
| 10 | Delta_net_profit_margin | 质量 |
| 11 | vwap_deviation_10d | 量价 |
| 12 | avg_normalized_range_5d | 波动率 |
| 13 | turnover_trend | 流动性 |
| 14 | vwap_deviation_5d | 量价 |
| 15 | reversal_2d | 反转 |
| 16 | PB_Ratio | 估值 |
| 17 | momentum_vol_adjusted_20 | 风险调整 |
| 18 | Sector_Relative_PB | 估值(中性) |
| 19 | Sector_Relative_PE | 估值(中性) |
| 20 | Sector_Relative_DividendYield | 估值(中性) |
| 21 | risk_adjusted_momentum_5d_20d | 风险调整 |
| 22 | Size | 规模 |
| 23 | Turnover | 流动性 |
| 24 | PriceToSales | 估值 |
| 25 | DividendYield | 股息 |
| 26 | Delta_OperatingCashFlowYield | 现金流 |
| 27 | Delta_AssetTurnover | 运营效率 |
| 28 | Delta_DebtToEquity | 财务杠杆 |
| 29 | SUE | 盈余质量 |
| 30 | AssetGrowth | 投资 |
| 31 | AccrualsRatio | 盈余质量 |
| 32 | DebtToEquity | 财务杠杆 |
| 33 | MAX_20d | 行为金融 |
| 34 | RevenueGrowth | 成长 |
| 35 | Amihud_20d | 非流动性 |
| 36 | FCF_Yield | 自由现金流 |
| 37 | ROA | 盈利能力 |
| 38 | Skewness_20d | 收益分布 |
| 39 | NetProfitGrowth | 成长 |
| 40 | EPS_Quality | 盈余质量 |
| 41 | TurnoverVol_20d | 流动性动态 |
| 42 | Reversal_3d | 反转 |

### 归一化策略

| 方法 | 适用因子 |
|------|---------|
| 截面 z-score | 全部 200 个因子（在 `build_features_from_h5.py` 中自动完成）|
| 行业中性化 | trailing_PE_ratio, PB_Ratio, DividendYield → Sector_Relative_* |
| 季度差分 | roe, netprofit_margin, DebtToEquity, AssetTurnover, OperatingCashFlowYield |
| 1%/99% 缩尾 | Delta_roe, Delta_net_profit_margin, SUE, AccrualsRatio, DebtToEquity, Amihud_20d, FCF_Yield, Skewness_20d, EPS_Quality |
