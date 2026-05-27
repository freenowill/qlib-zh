# 因子挖掘失败记录

> 记录所有被提出但未通过评估的因子，避免重复挖掘浪费 LLM 资源
> 数据源：`cn_extra_data`（58 个字段）
> LLM：DeepSeek-v4-pro / DeepSeek-v4-flash

## 失败因子列表

| 因子名称 | 类型 | 失败原因 | 日期 |
|----------|------|----------|------|
| Momentum_20d | 动量 | Duplicate of #1 MediumTermMomentum_20d | 2026-05-23 |
| Volatility_20d | 波动率 | Duplicate of #3 RealizedVolatility_20d | 2026-05-23 |
| Value_BP | 估值 | Duplicate of #29 book_to_price | 2026-05-23 |
| Profitability_EY | 估值 | Duplicate of #15 earnings_yield | 2026-05-23 |
| Quality_ROE | 质量 | Duplicate of #14 roe | 2026-05-23 |
| ShortTermReversal_5d | 反转 | Duplicate of #24 reversal_5d | 2026-05-23 |
| InversePB | 估值 | Duplicate of #29 book_to_price | 2026-05-23 |
| Momentum10 | 动量 | Duplicate of #17 momentum_10d | 2026-05-23 |
| Momentum20 | 动量 | Duplicate of #1 MediumTermMomentum_20d | 2026-05-23 |
| ShortTermReversal5 | 反转 | Duplicate of #24 reversal_5d | 2026-05-23 |
| mom_5d | 动量 | Duplicate of #9 momentum_5d | 2026-05-23 |

---

---

## 1. Momentum_20d

- **类型**：动量因子
- **描述**：20日价格动量，定义为过去20个交易日的累计简单收益率。
- **公式**：

  $$R_{t}^{20} = (P_t / P_{t-20}) - 1$$

- **失败原因**：Duplicate of existing factor #1 MediumTermMomentum_20d — same concept (20-day momentum), same window (20d), same formula.
- **日期**：2026-05-23
---
## 2. Volatility_20d

- **类型**：波动率因子
- **描述**：20日历史波动率，定义为过去20个交易日简单收益率的标准差。
- **公式**：

  $$\sigma_t^{20} = \sqrt{\frac{1}{19}\sum_{i=1}^{20}(r_{t-i+1} - \bar{r}_t)^2}$$

- **失败原因**：Duplicate of existing factor #3 RealizedVolatility_20d — same concept (20-day volatility), same window (20d). Minor formula difference (simple returns vs log returns, no annualization) does not constitute a new factor.
- **日期**：2026-05-23
---
## 3. Value_BP

- **类型**：估值因子
- **描述**：账面市值比（B/P），定义为每股净资产除以收盘价。
- **公式**：

  $$BP_t = \frac{bps_t}{P_t}$$

- **失败原因**：Duplicate of existing factor #29 book_to_price — same concept, same formula (bps/price = 1/PB).
- **日期**：2026-05-23
---
## 4. Profitability_EY

- **类型**：估值因子
- **描述**：盈利收益率（E/P），定义为每股收益除以收盘价。
- **公式**：

  $$EY_t = \frac{eps_t}{P_t}$$

- **失败原因**：Duplicate of existing factor #15 earnings_yield — same concept, same formula (eps/close).
- **日期**：2026-05-23
---
## 5. Quality_ROE

- **类型**：质量因子
- **描述**：净资产收益率（ROE），衡量公司运用股东权益创造利润的效率。
- **公式**：

  $$ROE_t = \$roe_t$$

- **失败原因**：Duplicate of existing factor #14 roe — same concept, same data source ($roe/$roe_yearly).
- **日期**：2026-05-23
---
## 6. ShortTermReversal_5d

- **类型**：反转因子
- **描述**：5日短期反转因子，定义为过去5个交易日的累计收益率。
- **公式**：

  $$R_{t}^{5} = \frac{P_t}{P_{t-5}} - 1$$

- **失败原因**：Duplicate of existing factor #24 reversal_5d — same concept, same window (5d), same formula (5-day return). Minor sign difference (positive vs negative) does not constitute a new factor.
- **日期**：2026-05-23
---

## 6. ShortTermReversal_5d

- **类型**：反转因子
- **描述**：过去5个交易日的累计收益率。
- **公式**：

  $$R_{t}^{5} = \frac{P_t}{P_{t-5}} - 1$$

- **失败原因**：Duplicate of existing factor #24 reversal_5d — same concept, same window (5d), same formula (5-day return). Minor sign difference (positive vs negative) does not constitute a new factor.
- **日期**：2026-05-23
---
## 7. InversePB

- **类型**：估值因子
- **描述**：市净率的倒数 (1/PB)，即账面市值比。
- **公式**：

  $$\text{InversePB}_t = \frac{1}{\text{PB}_t}$$

- **失败原因**：Duplicate of existing factor #29 book_to_price — same formula (1/PB = bps/price).
- **日期**：2026-05-23
---
## 8. Momentum10

- **类型**：动量因子
- **描述**：过去10个交易日的累计简单收益率。
- **公式**：

  $$\text{Momentum10}_t = \frac{P_t}{P_{t-10}} - 1$$

- **失败原因**：Duplicate of existing factor #17 momentum_10d — same window (10d), same formula.
- **日期**：2026-05-23
---
## 9. Momentum20

- **类型**：动量因子
- **描述**：过去20个交易日的累计简单收益率。
- **公式**：

  $$\text{Momentum20}_t = \frac{P_t}{P_{t-20}} - 1$$

- **失败原因**：Duplicate of existing factor #1 MediumTermMomentum_20d — same window (20d), same formula.
- **日期**：2026-05-23
---
## 10. ShortTermReversal5

- **类型**：反转因子
- **描述**：过去5个交易日的累计收益率。
- **公式**：

  $$R_{t}^{5} = \frac{P_t}{P_{t-5}} - 1$$

- **失败原因**：Duplicate of existing factor #24 reversal_5d — same concept, same window (5d), same formula. (Appeared again in round 2 with slightly different naming.)
- **日期**：2026-05-23
---
## 11. mom_5d

- **类型**：动量因子
- **描述**：5日价格动量，定义为过去5个交易日的累计简单收益率（使用复权收盘价）。
- **公式**：

  $$\text{Momentum}_{5d} = \frac{\text{adjclose}_t}{\text{adjclose}_{t-5}} - 1$$

- **失败原因**：Duplicate of existing factor #9 momentum_5d — same window (5d), same formula (cumulative return over 5 trading days).
- **日期**：2026-05-23
---

<!-- 失败因子条目格式：
## N. factor_name

- **类型**：<中文类别>
- **描述**：<一句话描述因子含义>
- **公式**：

  $$<LaTeX>$$

- **失败原因**：<final_decision: False，具体原因>
- **日期**：YYYY-MM-DD
---
-->
