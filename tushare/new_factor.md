# 因子挖掘结果 — 通过评估的因子

> 数据源：`cn_extra_data`（58 个字段：行情、估值、基本面、财务）
> LLM：DeepSeek-v4-pro
> 日期：2026-05-20

## 摘要

经多轮自动化因子挖掘，DeepSeek-v4-pro 基于 cn_extra_data 的 58 个字段共提出 50+ 因子，**30 个通过评估**。

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
| momentum_5d | 动量因子 | 价格 | ✅ True |
| reversal_1d | 反转因子 | 价格 | ✅ True |
| volume_ratio_5d | 流动性因子 | 成交量 | ✅ True |
| intraday_volatility | 波动率因子 | 价格(高低) | ✅ True |
| volume_weighted_momentum_5d | 量价因子 | 成交量+价格 | ✅ True |
| roe | 质量因子 | 基本面 (roe_yearly) | ✅ True |
| earnings_yield | 估值因子 | 估值+价格 (eps) | ✅ True |
| net_profit_margin | 质量因子 | 基本面 (netprofit_margin) | ✅ True |
| momentum_10d | 动量因子 | 价格 | ✅ True |
| vwap_deviation_10d | 量价因子 | VWAP+价格 | ✅ True |
| avg_normalized_range_5d | 波动率因子 | 价格(高低) | ✅ True |
| turnover_trend | 流动性因子 | 换手率 | ✅ True |
| vwap_deviation_5d | 量价因子 | VWAP+价格 | ✅ True |
| reversal_2d | 反转因子 | 价格 | ✅ True |
| volume_ratio_5d_20d | 流动性因子 | 成交量 | ✅ True |
| reversal_5d | 反转因子 | 价格 | ✅ True |
| PB_Ratio | 估值因子 | 估值 (pb) | ✅ True |
| Momentum_Vol_Adjusted_20 | 风险调整因子 | 价格 | ✅ True |
| Sector_Relative_PB | 估值因子(截面) | 估值 (pb) | ✅ True |
| avg_volume_ratio_20d | 流动性因子 | 成交量 | ✅ True |
| book_to_price | 估值因子 | 估值 (pb) | ✅ True |
| risk_adjusted_momentum_5d_20d | 风险调整因子 | 价格 | ✅ True |

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

## 9. momentum_5d

- **类型**：动量因子 (Momentum Factor)
- **描述**：5日价格动量，定义为 5 日收益率。相比已有的 20 日动量因子，该因子捕捉更短期的趋势延续效应。
- **公式**：

  $$MOM_t = \frac{close_t}{close_{t-5}} - 1$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $close_{t-5}$：5 个交易日前收盘价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 10. reversal_1d

- **类型**：反转因子 (Reversal Factor)
- **描述**：1日反转因子，定义为负的 1 日收益率。捕捉隔日均值回复效应，相比已有的 20 日反转因子反映更短期的反转特性。
- **公式**：

  $$REV_t = -\left(\frac{close_t}{close_{t-1}} - 1\right)$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $close_{t-1}$：前一交易日收盘价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 11. volume_ratio_5d

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：5日量比，定义为当日成交量除以前 5 个交易日平均成交量。衡量当日成交活跃度相对于近期平均的水平，值 > 1 表示放量，值 < 1 表示缩量。
- **公式**：

  $$VR_t = \frac{volume_t}{\frac{1}{5} \sum_{i=1}^{5} volume_{t-i}}$$

- **变量**：
  - $volume_t$：第 t 日成交量
  - $volume_{t-i}$：第 t-i 日成交量（i=1,…,5）
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 12. intraday_volatility

- **类型**：波动率因子 (Volatility Factor)
- **描述**：日内振幅因子，定义为 (最高价 - 最低价) / 收盘价。衡量单日内的价格波动程度，相比已有的 20 日已实现波动率，该因子聚焦于日内波动特征。
- **公式**：

  $$IV_t = \frac{high_t - low_t}{close_t}$$

- **变量**：
  - $high_t$：第 t 日最高价
  - $low_t$：第 t 日最低价
  - $close_t$：第 t 日收盘价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 13. volume_weighted_momentum_5d

- **类型**：量价因子 (Volume-Price Factor)
- **描述**：成交量加权 5 日动量，以过去 5 个交易日的日收益率的成交量加权平均值作为因子值。成交量越大的交易日其收益率权重越高，相比简单移动平均更能反映"量价配合"的信号强度。
- **公式**：

  $$VWMOM_t = \frac{\sum_{i=1}^{5} volume_{t-i} \cdot R_{t-i}}{\sum_{i=1}^{5} volume_{t-i}}, \quad R_{t-i} = \frac{close_{t-i}}{close_{t-i-1}} - 1$$

- **变量**：
  - $volume_{t-i}$：第 t-i 日成交量
  - $close_{t-i}$：第 t-i 日收盘价
  - $R_{t-i}$：第 t-i 日日收益率
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 14. roe

- **类型**：质量因子 (Quality Factor)
- **描述**：净资产收益率 (ROE)，衡量公司运用股东权益创造利润的效率。ROE 越高表示公司盈利能力和资本运用效率越强，是价值投资中最核心的质量指标之一。
- **公式**：

  $$ROE_t = \$roe\_yearly_t$$

- **变量**：
  - $\$roe\_yearly_t$：第 t 日最新披露的年化 ROE 值（来自 `$roe_yearly` 字段）
- **数据来源**：cn_extra_data 财务数据中的 `$roe_yearly` 字段
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），允许部分股票缺失 ROE 值。

---

## 15. earnings_yield

- **类型**：估值因子 (Value Factor)
- **描述**：盈利收益率（E/P），定义为每股收益除以当前股价，即 PE 的倒数。相比 PE，盈利收益率在负盈利时更有意义（可得到负值），且可与债券收益率直接比较，常用于FED模型判断股债相对价值。
- **公式**：

  $$EY_t = \frac{\$eps_t}{close_t}$$

- **变量**：
  - $\$eps_t$：第 t 日最新披露的每股收益（来自 `$basic_eps` 字段）
  - $close_t$：第 t 日收盘价
- **数据来源**：cn_extra_data 财务数据中的 `$basic_eps` 字段 + 行情数据 `$close`
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），允许部分股票缺失 EPS 值。

---

## 16. net_profit_margin

- **类型**：质量因子 (Quality Factor) / 运营效率因子
- **描述**：净利润率，定义为净利润占营业收入的比例，反映公司在扣除所有成本费用后的最终盈利能力。净利率越高表示公司盈利质量和成本控制能力越强。
- **公式**：

  $$NPM_t = \$netprofit\_margin_t$$

- **变量**：
  - $\$netprofit\_margin_t$：第 t 日最新披露的净利润率（来自 `$netprofit_margin` 字段）
- **数据来源**：cn_extra_data 财务数据中的 `$netprofit_margin` 字段
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），允许部分股票缺失净利率值。

---

## 17. momentum_10d

- **类型**：动量因子 (Momentum Factor)
- **描述**：10日价格动量，填补 5 日和 20 日动量之间的窗口空白。捕捉中期趋势延续效应。
- **公式**：

  $$MOM_{10,t} = \frac{close_t}{close_{t-10}} - 1$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $close_{t-10}$：10 个交易日前收盘价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 18. vwap_deviation_10d

- **类型**：量价因子 (Volume-Price Factor) / 均值回复因子
- **描述**：10日 VWAP 偏离度，定义为收盘价相对于过去 10 日成交量加权均价 (VWAP) 的偏离比例。正值表示价格高于近期 VWAP（可能超买），负值表示低于 VWAP（可能超卖），捕捉均值回复机会。
- **公式**：

  $$VWAP\_dev_{10,t} = \frac{close_t}{\frac{1}{10}\sum_{i=0}^{9} vwap_{t-i}} - 1$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $vwap_{t-i}$：第 t-i 日成交量加权均价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 19. avg_normalized_range_5d

- **类型**：波动率因子 (Volatility Factor)
- **描述**：5日平均归一化振幅，定义为过去 5 个交易日 (最高价-最低价)/收盘价 的平均值。相比单日的 intraday_volatility，该因子通过平滑处理降低了日内噪音，更稳健地衡量近期波动水平。
- **公式**：

  $$ANR_{5,t} = \frac{1}{5}\sum_{i=0}^{4}\frac{high_{t-i} - low_{t-i}}{close_{t-i}}$$

- **变量**：
  - $high_{t-i}$：第 t-i 日最高价
  - $low_{t-i}$：第 t-i 日最低价
  - $close_{t-i}$：第 t-i 日收盘价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 20. turnover_trend

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：换手率趋势因子，定义为短期 (5日) 平均换手率与长期 (20日) 平均换手率的相对差异。正值表示近期交易活跃度上升（关注度提高），负值表示活跃度下降。相比基于成交量 (volume) 的因子，换手率已标准化为流通股本比例，跨股票可比性更强。
- **公式**：

  $$TO\_trend_t = \frac{\frac{1}{5}\sum_{i=0}^{4} turnover_{t-i} - \frac{1}{20}\sum_{i=0}^{19} turnover_{t-i}}{\frac{1}{20}\sum_{i=0}^{19} turnover_{t-i}}$$

- **变量**：
  - $turnover_{t-i}$：第 t-i 日换手率（来自 `$turnover` 或 `$turnover_f` 字段）
- **数据来源**：cn_extra_data 行情数据中的 `$turnover` / `$turnover_f` 字段
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 21. vwap_deviation_5d

- **类型**：量价因子 (Volume-Price Factor) / 均值回复因子
- **描述**：5日 VWAP 偏离度，与 10 日版本类似但使用更短窗口，对短期价格偏离更敏感，适合捕捉快速均值回复信号。
- **公式**：

  $$VWAP\_dev_{5,t} = \frac{close_t}{\frac{1}{5}\sum_{i=0}^{4} vwap_{t-i}} - 1$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $vwap_{t-i}$：第 t-i 日成交量加权均价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 22. reversal_2d

- **类型**：反转因子 (Reversal Factor)
- **描述**：2日反转因子，定义为负的 2 日收益率。填补 1 日反转和 20 日反转之间的窗口空白，捕捉短期过度反应后的均值回复。
- **公式**：

  $$REV_{2,t} = -\left(\frac{close_t}{close_{t-2}} - 1\right)$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $close_{t-2}$：2 个交易日前收盘价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 23. volume_ratio_5d_20d

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：短期/长期成交量比率，定义为 5 日平均成交量与 20 日平均成交量的比值。值 > 1 表示近期成交量相对放大（市场关注度上升），值 < 1 表示近期缩量。与单日量比 (volume_ratio_5d) 不同，该因子比较的是两个移动平均，信号更加平滑稳定。
- **公式**：

  $$VR\_5\_20_t = \frac{\frac{1}{5}\sum_{i=0}^{4} volume_{t-i}}{\frac{1}{20}\sum_{i=0}^{19} volume_{t-i}}$$

- **变量**：
  - $volume_{t-i}$：第 t-i 日成交量
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

## 24. reversal_5d

- **类型**：反转因子 (Reversal Factor)
- **描述**：5日反转因子，定义为负的 5 日收益率。填补 1/2/20 日反转之间的窗口空白，捕捉周度级别的均值回复效应。
- **公式**：

  $$REV_{5,t} = -\left(\frac{P_t}{P_{t-5}} - 1\right)$$

- **变量**：
  - $P_t$：第 t 日收盘价
  - $P_{t-5}$：5 个交易日前收盘价
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），无异常值。

---

---

## 25. PB_Ratio

- **类型**：估值因子 (Value Factor)
- **描述**：市净率 (PB)，直接使用日频 pb 数据列。PB 衡量公司市值相对于其净资产的倍数，是经典的价值衡量指标。低 PB 通常意味着股票可能被低估或被市场忽视。
- **公式**：

  $$PB_t = \$pb_t$$

- **变量**：
  - $\$pb_t$：第 t 日市净率（来自 `$pb` 字段）
- **数据来源**：cn_extra_data 估值数据中的 `$pb` 字段
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），与已有 PE 因子形成互补，覆盖估值维度的另一经典指标。

---

## 26. Momentum_Vol_Adjusted_20

- **类型**：风险调整因子 (Risk-Adjusted Factor) / 动量因子
- **描述**：20 日波动率调整动量，定义为 20 日价格收益率除以 20 日历史波动率。相比普通动量因子，该因子自动降低高波动期的敞口，提高风险调整后收益的稳定性。高值表示在低风险下实现了强动量，低值表示趋势弱或波动大。
- **公式**：

  $$\text{MomVol}_{t} = \frac{r_{t,20}}{\sigma_{t,20}}$$

  $$r_{t,20} = \frac{P_t - P_{t-20}}{P_{t-20}}, \quad \sigma_{t,20} = \sqrt{\frac{1}{20}\sum_{i=0}^{19}(r_{t-i} - \bar{r})^2}$$

- **变量**：
  - $P_t$：第 t 日收盘价
  - $r_{t,20}$：20 日简单收益率
  - $\sigma_{t,20}$：20 日历史波动率（日收益率标准差）
- **数据来源**：cn_extra_data 行情数据中的 `$close` 字段（计算衍生）
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），融合了动量和波动率两个维度的信息，提供风险调整后的趋势信号。

---

## 27. Sector_Relative_PB

- **类型**：估值因子 (Value Factor) / 截面相对估值
- **描述**：行业相对市净率，定义为个股 PB 减去同一板块（交易板）的截面中位数 PB。正值表示该股票 PB 高于同板块中位数（相对高估），负值表示低于中位数（相对低估）。相比绝对 PB，该因子消除了板块间系统性估值差异，更能捕捉个股层面的相对错误定价。
- **公式**：

  $$\text{SectorRelPB}_t = \$pb_t - \text{median}_{\text{sector}}(\$pb_t)$$

- **变量**：
  - $\$pb_t$：第 t 日市净率
  - $\text{median}_{\text{sector}}$：同板块（按交易代码前缀分组的截面）中位数 PB
- **数据来源**：cn_extra_data 估值数据中的 `$pb` 字段，截面分组基于交易代码前缀（近似板块分类）
- **评估反馈**：代码执行成功，经多轮迭代修复（float32→float64 类型转换、截面分组实现方式调整），最终输出格式正确（MultiIndex [datetime, instrument]、单列 float64），允许部分股票因板块内样本不足而产生缺失值。

---

---

## 28. avg_volume_ratio_20d

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：20 日平均成交量比率，定义为当日成交量除以过去 20 日平均成交量（含当日）。相比已有的 5 日量比（volume_ratio_5d），该因子使用更长的回溯窗口，信号更平滑，适合捕捉中长期成交量异常变化。
- **公式**：

  $$AVR20_t = \frac{V_t}{\frac{1}{20}\sum_{i=0}^{19} V_{t-i}}$$

- **变量**：
  - $V_t$：第 t 日成交量（来自 `$volume` 字段）
  - $V_{t-i}$：第 t-i 日成交量
- **数据来源**：cn_extra_data 行情数据中的 `$volume` 字段
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），与 volume_ratio_5d 形成不同窗口的互补。

---

## 29. book_to_price

- **类型**：估值因子 (Value Factor)
- **描述**：账面市值比 (B/P)，定义为市净率 (PB) 的倒数。相比 PB，B/P 在处理极低 PB 或负净资产股票时分布更稳定，且与 Fama-French HML 因子的构造方式一致。高 B/P 表示相对账面价值而言股价较低（价值股），低 B/P 表示成长股。
- **公式**：

  $$BP_t = \frac{1}{\$pb_t}$$

- **变量**：
  - $\$pb_t$：第 t 日市净率（来自 `$pb` 字段）
- **数据来源**：cn_extra_data 估值数据中的 `$pb` 字段
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），与 PB_Ratio 呈单调关系但分布特性不同（更接近正态），在回归模型中可能表现更优。

---

## 30. risk_adjusted_momentum_5d_20d

- **类型**：风险调整因子 (Risk-Adjusted Factor) / 动量因子
- **描述**：5 日风险调整动量，定义为 5 日价格动量除以年化 20 日已实现波动率。与已有的 Momentum_Vol_Adjusted_20（20 日动量/20 日波动率）不同，该因子使用更短的动量窗口（5 日），对短期趋势变化更敏感，同时仍用 20 日波动率做风险标准化。
- **公式**：

  $$\text{RAMom}_{5,20,t} = \frac{\frac{close_t}{close_{t-5}} - 1}{\sqrt{252} \cdot \text{std}\big(\ln(close_i/close_{i-1}), i=t-19,\dots,t\big)}$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $close_{t-5}$：5 个交易日前收盘价
  - 分子：5 日价格动量
  - 分母：年化 20 日已实现波动率（对数收益率标准差 × √252）
- **数据来源**：cn_extra_data 行情数据中的 `$close` 字段（计算衍生）
- **评估反馈**：代码执行成功，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），与 Momentum_Vol_Adjusted_20 形成短期/中期互补。

---

## 使用方法

以上因子可在 qlib 工作流中使用，通过 AlphaExtra handler（`qlib/contrib/data/handler_extra.py`）加载 cn_extra_data 数据源后，在因子表中引用。
