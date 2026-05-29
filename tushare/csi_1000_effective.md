# CSI1000 长期稳定有效因子

> 基于 `csi1000_5d_label_ic` 试验（2026-05-26）的 walk-forward IC 分析结果。
> 使用 CSI1000 成分股（1000 只），42 个实践因子，5 folds walk-forward（2015-2026 年）。
> 判定标准：`|mean_RankIC| ≥ 0.04` 且 `|IR_RankIC| ≥ 0.33`，在测试集和验证集中交叉验证一致。
> 更新：2026-05-26

---

## 入选因子总览

| # | 因子 | 方向 | 类型 | Test mean_RankIC | Test IR_RankIC | Valid mean_RankIC | Valid IR_RankIC |
|---|------|------|------|-----------------|----------------|------------------|----------------|
| 1 | Size | 负向 | 市值/规模 | -0.075 | -0.643 | -0.080 | -0.677 |
| 2 | avg_normalized_range_5d | 负向 | 波动率 | -0.069 | -0.354 | -0.070 | -0.371 |
| 3 | RealizedVolatility_20d | 负向 | 波动率 | -0.065 | -0.335 | -0.068 | -0.363 |
| 4 | MAX_20d | 负向 | 行为金融 | -0.064 | -0.407 | -0.065 | -0.436 |
| 5 | Turnover | 负向 | 流动性 | -0.064 | -0.370 | -0.067 | -0.402 |
| 6 | Amihud_20d | 正向 | 非流动性 | +0.066 | +0.503 | +0.075 | +0.573 |
| 7 | Sector_Relative_PB | 负向 | 估值(行业中性) | -0.041 | -0.536 | -0.040 | -0.511 |

**统计**: 7 个长期稳定有效因子（其中波动率类 2 个，流动性类 2 个，估值类 1 个，市值 1 个，行为金融 1 个）。
**对比 CSI300**: CSI1000 因子 IC 强度约为 CSI300 的 2 倍，小市值效应和流动性溢价在中小盘股中更为显著。

---

## 因子详情

### 1. Size

- **类型**：市值/规模因子 (Size Factor)
- **方向**：**负向**（小市值跑赢大市值）
- **描述**：总市值的自然对数。A 股小市值效应在中小盘股票中尤为显著——CSI1000 中 Size 因子 Rank IC 高达 -0.075，是 CSI300（-0.020）的 3 倍以上。在 1000 只成分股范围内，最小市值的股票仍享有显著的小市值溢价。
- **公式**：

  $$\text{Size}_t = \ln(\$total\_mv_t)$$

  其中 $\$total\_mv_t$ 为第 t 日总市值（元）。

- **数据来源**：H5 `$total_mv` → `daily_basic.csv` 的 `total_mv` 字段
- **CSI1000 表现**：Test RankIC = -0.075, IR = -0.643, pos_RankIC = 24.6%
- **参考**：practice_factor.md #22

---

### 2. avg_normalized_range_5d

- **类型**：波动率因子 (Volatility Factor)
- **方向**：**负向**（高振幅→低收益）
- **描述**：5 日平均归一化振幅。使用日内高低价差度量波动率（而非收益率标准差），与 RealizedVolatility_20d 互补。高振幅反映价格不确定性大，投资者要求更低的风险溢价。CSI1000 中该因子 IC 约 -0.07，远超 CSI300 的 -0.03。
- **公式**：

  $$ANR_t^{5} = \frac{1}{5}\sum_{i=0}^{4} \frac{high_{t-i} - low_{t-i}}{close_{t-i}}$$

- **数据来源**：H5 `$high`, `$low`, `$close` → `daily.csv` 的 `high`, `low`, `close` 字段
- **CSI1000 表现**：Test RankIC = -0.069, IR = -0.354, pos_RankIC = 37.8%
- **参考**：practice_factor.md #12

---

### 3. RealizedVolatility_20d

- **类型**：波动率因子 (Volatility Factor)
- **方向**：**负向**（高波动→低收益）
- **描述**：20 日年化已实现波动率，使用简单收益率的标准差乘以年化因子 √252。经典的低波动率异象（Low Volatility Anomaly）在 CSI1000 中显著成立——高波动股票在中小盘股中表现更差。IR 约 -0.34，中等稳定。
- **公式**：

  $$\sigma_t^{20} = \sqrt{\frac{1}{19} \sum_{i=0}^{19} (r_{t-i} - \bar{r}_{20})^2} \times \sqrt{252}$$

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **CSI1000 表现**：Test RankIC = -0.065, IR = -0.335, pos_RankIC = 37.3%
- **参考**：practice_factor.md #1

---

### 4. MAX_20d

- **类型**：行为金融因子 (MAX Effect / Lottery Preference)
- **方向**：**负向**（过去有极端正收益→后续反转）
- **描述**：过去 20 个交易日的最大日收益率（Bali et al. 2011）。行为金融学解释为彩票偏好（lottery preference）——中小盘股投资者更容易追捧有过极端正收益的"彩票型"股票，导致其后续表现不佳。MAX 效应在 CSI1000 中 IR 高达 -0.4，验证了该因子在小盘股中的强劲预测力。
- **公式**：

  $$MAX_t^{20} = \max(r_{t-19}, r_{t-18}, \dots, r_t)$$

  其中 $r_t = \dfrac{close_t}{close_{t-1}} - 1$。

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：Bali, Cakici & Whitelaw (2011), "Maxing out: Stocks as lotteries and the cross-section of expected returns"
- **CSI1000 表现**：Test RankIC = -0.064, IR = -0.407, pos_RankIC = 34.7%

---

### 5. Turnover

- **类型**：流动性因子 (Liquidity Factor)
- **方向**：**负向**（高换手→低收益）
- **描述**：日换手率 = 成交量 / 自由流通股本。高换手率通常反映散户主导的过度交易和信息不对称，预示后续收益走低。CSI1000 中的换手率效应强度（IC ≈ -0.065）约为 CSI300（IC ≈ -0.025）的 2.5 倍，说明流动性溢价在中小盘股中更为关键。
- **公式**：

  $$\text{Turnover}_t = \frac{\$volume_t}{\$free\_sh_t}$$

- **数据来源**：H5 `$volume`（`daily.csv` `vol`）, `$free_sh`（`daily_basic.csv` `free_share`）
- **CSI1000 表现**：Test RankIC = -0.064, IR = -0.370, pos_RankIC = 36.3%
- **参考**：practice_factor.md #23

---

### 6. Amihud_20d

- **类型**：非流动性因子 (Illiquidity Factor)
- **方向**：**正向**（高非流动性→高收益）
- **描述**：Amihud (2002) 非流动性度量，定义为过去 20 个交易日单位成交金额引起的价格变动（绝对值）的均值。该因子在 CSI1000 中表现极为突出——IC = +0.066，IR = +0.503，是全因子集中 IC 最高的正向因子。CSI1000 中大量中小盘股流动性不足，非流动性溢价极其显著。
- **公式**：

  $$Amihud_t = \frac{1}{20} \sum_{i=0}^{19} \frac{|r_{t-i}|}{\$amount_{t-i}} \times 10^8$$

- **数据来源**：
  - `$amount` → `daily.csv` 的 `amount` 字段（日成交金额）
  - `$close` → `daily.csv` 的 `close` 字段
- **参考**：Amihud, Y. (2002), "Illiquidity and stock returns: cross-section and time-series effects"
- **CSI1000 表现**：Test RankIC = +0.066, IR = +0.503, pos_RankIC = 69.5%
- **注意**：1%/99% 缩尾处理

---

### 7. Sector_Relative_PB

- **类型**：估值因子（行业中性化截面因子）
- **方向**：**负向**（行业调整后低 PB→高收益）
- **描述**：个股 PB 减去同一申万一级行业的 PB 截面中位数。消除行业系统性估值差异（如银行股 PB 普遍低于科技股）。该因子在 CSI1000 中 IR 高达 -0.536，是稳定性最高的因子之一——行业中性化处理有效消除了行业偏差，使得 PB 的相对高低在行业内可比。
- **公式**：

  $$\text{SectorRelPB}_{t, s} = \$pb_{t, s} - \text{median}_{p \in \text{industry}(s)}(\$pb_{t, p})$$

- **数据来源**：H5 `$pb` + `tushare/cn_data/sw_industry.csv`（申万 2021 版行业分类）
- **CSI1000 表现**：Test RankIC = -0.041, IR = -0.536, pos_RankIC = 28.8%
- **参考**：practice_factor.md #18

---

## 未入选因子的说明

以下因子未能通过筛选（在测试集或验证集中不满足 `|IR| ≥ 0.33` 且 `|IC| ≥ 0.04` 的双重标准）：

| 因子 | 最高 |IR| | 最高 |IC| | 落选原因 |
|------|-----|------|----------|------|------|
| PB_Ratio | 0.284 | 0.053 | IR 不足 0.3，说明方向一致性差。但其行业中性版本 Sector_Relative_PB 表现极佳 |
| TurnoverVol_20d | 0.410 | 0.032 | IC 偏低（< 0.04），信号强度不足以补偿其稳定性 |
| Sector_Relative_DividendYield | 0.315 | 0.023 | IC 偏低。股息率效应在大盘股中更强 |
| earnings_yield | 0.181 | 0.027 | 稳定性和信号强度均不足 |
| trailing_PE_ratio | 0.175 | 0.028 | 行业中性版本 Sector_Relative_PE 也未能提升 IR 和 IC |
| DividendYield | 0.199 | 0.026 | 同上 |
| PriceToSales | 0.250 | 0.037 | IR < 0.3 |
| momentum_vol_adjusted_20 | 0.202 | 0.030 | 动量效应在 CSI1000 中不显著 |

其余因子（reversal_1d/2d/3d, vwap_deviation, SUE, AssetGrowth, ROA, FCF_Yield 等）IC ≤ 0.02 或方向不稳定，不具备长期稳定预测能力。

---

## 使用建议

1. **Top 7 因子组合**: 以上 7 个因子在 CSI1000 中展现出长期稳定的预测力，可直接作为 AlphaExtra 模型的核心输入。建议给予 Amihud_20d、Size、Sector_Relative_PB 更高的权重（IR 最高）。
2. **因子互补性**: Size + Turnover + Amihud_20d 三者共同刻画了"小市值 + 低换手 + 高非流动性"这一 A 股中小盘经典 alpha 来源。MAX_20d + 波动率因子提供了独立于市值的风险信号。
3. **行业中性化**: Sector_Relative_PB 的表现远优于原始 PB_Ratio，建议对所有估值因子坚持行业中性化处理。
4. **与 CSI300 对比**: CSI1000 因子的 IC 绝对值约是 CSI300 的 2-3 倍——中小盘股的截面可预测性更高，因子投资策略在中小盘指数中更为有效。
