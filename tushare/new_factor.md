# 因子挖掘结果 — 通过评估的因子

> 数据源：`cn_extra_data`（58 个字段：行情、估值、基本面、财务）
> LLM：DeepSeek-v4-pro
> 日期：2026-05-18

## 摘要

经多轮自动化因子挖掘，DeepSeek-v4-pro 基于 cn_extra_data 的 58 个字段共提出 20+ 因子，**8 个通过评估**。

| 因子名称 | 类型 | 数据维度 | 评估结果 |
|----------|------|----------|----------|
| MediumTermMomentum_20d | 动量因子 | 价格 | ✅ True |
| 20_day_reversal | 反转因子 | 价格 | ✅ True |
| RealizedVolatility_20d | 波动率因子 | 价格 | ✅ True |
| RSI_14d | 震荡因子 | 价格 | ✅ True |
| 5_day_volume_change | 流动性因子 | 成交量 | ✅ True |
| trailing_PE_ratio | 估值因子 | 估值 (pe_ttm) | ✅ True |
| obv_slope_10day | 量价因子 | 成交量+价格 | ✅ True |
| sharpe_10day | 风险调整因子 | 价格 | ✅ True |

---

## 1. MediumTermMomentum_20d

- **类型**：动量因子 (Momentum Factor)
- **描述**：20日收益率，反映中期动量效应；正值表示趋势走强。
- **公式**：

  $$M_t = \frac{close_t}{close_{t-20}} - 1$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $close_{t-20}$：20 个交易日前收盘价
- **评估反馈**：代码执行无错误，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），计算值与公式定义一致。

---

## 2. RealizedVolatility_20d

- **类型**：波动率因子 (Volatility Factor)
- **描述**：20日年化已实现波动率（收盘价对收盘价），衡量风险和活跃程度。
- **公式**：

  $$\sigma_t = \sqrt{ \frac{1}{19} \sum_{i=0}^{19} (r_{t-i} - \bar{r})^2 } \times \sqrt{252}$$

  其中 $r_t = \frac{close_t}{close_{t-1}} - 1$，$\bar{r} = \frac{1}{20} \sum_{i=0}^{19} r_{t-i}$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $r_t$：日收益率
  - 252：年化系数（交易日数）
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex、单列 float64），无异常值，实现与公式描述一致。

---

## 3. RSI_14d

- **类型**：震荡因子 (Oscillator Factor)
- **描述**：14日相对强弱指数 (RSI)，衡量超买/超卖状态；RSI > 70 表示超买，RSI < 30 表示超卖。
- **公式**：

  $$RSI_t = 100 - \frac{100}{1 + RS_t}$$

  $$RS_t = \frac{\frac{1}{14} \sum_{i=0}^{13} \max(\Delta close_{t-i}, 0)}{\frac{1}{14} \sum_{i=0}^{13} \max(-\Delta close_{t-i}, 0)}$$

  $$\Delta close_{t-i} = close_{t-i} - close_{t-i-1}$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $\Delta close_{t-i}$：第 t-i 日价格变动
  - 14：回溯周期（天）
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex、单列 float64），无异常值，实现符合规范。

---

## 4. 20_day_reversal

- **类型**：反转因子 (Reversal Factor)
- **描述**：20日反转效应，定义为 20 日价格收益率的负值，捕捉中期均值回复特性。
- **公式**：

  $$R_t = -\left(\frac{P_t}{P_{t-20}} - 1\right)$$

- **变量**：
  - $P_t$：第 t 日收盘价
  - $P_{t-20}$：20 个交易日前收盘价
- **评估反馈**：代码执行无错误，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），计算值与公式定义一致。

---

## 5. 5_day_volume_change

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：5日成交量变化率，衡量交易活跃度的短期变动，反映投资者关注度和流动性变化。
- **公式**：

  $$V_t = \frac{volume_t}{volume_{t-5}} - 1$$

- **变量**：
  - $volume_t$：第 t 日成交量
  - $volume_{t-5}$：5 个交易日前成交量
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex、单列 float64），无异常值，实现符合规范。

---

## 6. trailing_PE_ratio

- **类型**：估值因子 (Value Factor)
- **描述**：滚动市盈率（PE TTM），直接使用已披露的滚动十二个月盈利数据计算，是最经典的价值衡量指标。低 PE 通常意味着股票可能被低估。
- **公式**：

  $$Factor_t = PE\_TTM_t$$

- **变量**：
  - $PE\_TTM_t$：第 t 日滚动市盈率（来自 `$pe_ttm` 数据列）
- **数据来源**：cn_extra_data 估值数据中的 `$pe_ttm` 字段
- **评估反馈**：代码执行成功，正确读取 `$pe_ttm` 列并输出为因子值，格式符合规范（MultiIndex、单列 float64），允许部分股票缺失 PE 值（如亏损股无 PE）。

---

## 7. obv_slope_10day

- **类型**：量价因子 (Volume-Price Factor)
- **描述**：能量潮（OBV）10日斜率。OBV 是按每日涨跌方向累积的成交量；其斜率通过线性回归计算，反映成交量的趋势变化，捕捉"聪明钱"流向信号。OBV 上升表示量价配合上涨，OBV 下降表示量价配合下跌。
- **公式**：

  $$OBV_t = \sum_{i=1}^{t} V_i \cdot \text{sgn}(P_i - P_{i-1})$$

  $$\beta_t = \frac{\sum_{i=0}^{9} (i - 4.5)(OBV_{t-i} - \overline{OBV}_{t,10})}{\sum_{i=0}^{9} (i - 4.5)^2}$$

- **变量**：
  - $V_i$：第 i 日成交量
  - $P_i$：第 i 日收盘价
  - $\text{sgn}$：符号函数（涨+1、跌-1、平0）
  - $\overline{OBV}_{t,10}$：10日窗口内 OBV 均值
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex、单列 float64），无异常值，实现符合规范。

---

## 8. sharpe_10day

- **类型**：风险调整因子 (Risk-Adjusted Factor)
- **描述**：10日滚动 Sharpe 比率。用过去 10 个交易日的日均对数收益率除以其标准差，衡量单位风险下的收益效率。该因子融合了动量（分子）和波动率（分母）的信息，是经典的风险调整后收益指标。
- **公式**：

  $$S_t = \frac{\frac{1}{10}\sum_{i=0}^{9} r_{t-i}}{\sqrt{\frac{1}{9}\sum_{i=0}^{9}(r_{t-i} - \bar{r})^2}}$$

  $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

- **变量**：
  - $P_t$：第 t 日收盘价
  - $r_t$：日对数收益率
  - $\bar{r}$：10日内平均收益率
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex、单列 float64），无异常值，实现符合规范。

---

## 使用方法

以上因子可在 qlib 工作流中使用，通过 AlphaExtra handler（`qlib/contrib/data/handler_extra.py`）加载 cn_extra_data 数据源后，在因子表中引用。
