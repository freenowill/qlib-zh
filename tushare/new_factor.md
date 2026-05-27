# 因子挖掘结果 — 通过评估的因子

> 数据源：`cn_extra_data`（58 个字段：行情、估值、基本面、财务）
> LLM：DeepSeek-v4-pro
> 日期：2026-05-20
> 更新：2026-05-21（Alpha158 重叠分析 + 合并标记）

## Alpha158 重叠分析

Alpha158 是 qlib 内置的 158 个纯价量因子（KBar 9个 + Price 4个 + Rolling 145个），仅使用 open/high/low/close/vwap/volume。以下 30 个因子中，与 Alpha158 重叠的有 **9 个**，独立因子 **21 个**（含 7 个基本面/估值因子）。

### 重叠因子一览

| 序号 | new_factor 因子 | 重叠的 Alpha158 | 关系 | 建议 |
|------|----------------|-----------------|------|------|
| 1 | momentum_5d | **ROC5** | 公式等价：momentum_5d = 1/ROC5 − 1 | 保留其一即可 |
| 2 | momentum_10d | **ROC10** | 同上，10日窗口 | 保留其一即可 |
| 3 | MediumTermMomentum_20d | **ROC20** | 同上，20日窗口 | 保留其一即可 |
| 4 | reversal_5d | **ROC5** | reversal_5d = 1 − 1/ROC5 | 与 momentum_5d 共线 |
| 5 | reversal_20d | **ROC20** | reversal_20d = 1 − 1/ROC20 | 与 momentum_20d 共线 |
| 6 | RSI_14d | **SUMP10 / SUMP20** | 相同公式 (RSI = SUMP×100)，相邻窗口 | 互补（不同窗口） |
| 7 | intraday_volatility | **KLEN** | (high−low)/close vs (high−low)/open | 高度相似，可保其一 |
| 8 | avg_normalized_range_5d | **KLEN** (5日均值) | 间接相关 | 平滑版，略有差异 |
| 9 | volume_ratio_5d | **VMA5** | 倒数关系，VMA5 含当日 | 高度相似 |
| 10 | avg_volume_ratio_20d | **VMA20** | 严格倒数关系 | 保留其一即可 |

### 独立因子（32个，无 Alpha158 等效）

| 类别 | 因子 | 说明 |
|------|------|------|
| 反转 | reversal_1d, reversal_2d | Alpha158 无 1d/2d 窗口 ROC |
| 波动率 | RealizedVolatility_20d, avg_normalized_range_5d, Volatility_5d, Volatility_10d | Alpha158 STD 用价格水平，此用收益率 |
| 量价 | obv_slope_10day, volume_weighted_momentum_5d, vwap_deviation_5d, vwap_deviation_10d | OBV、VWAP 多日均值不在 Alpha158 |
| 流动性 | 5_day_volume_change, volume_ratio_5d_20d, turnover_trend, Liquidity_Turnover_5d | 换手率、量比不在 Alpha158 |
| 风险调整 | sharpe_10day, Momentum_Vol_Adjusted_20, risk_adjusted_momentum_5d_20d | Sharpe、波动率调整动量不在 Alpha158 |
| 市值/规模 | Size | Alpha158 无市值数据 |
| 估值 | trailing_PE_ratio, PB_Ratio, earnings_yield, book_to_price, Sector_Relative_PB, **PriceToSales** | **Alpha158 无基本面数据**，新增市销率 |
| 质量 | roe, net_profit_margin | **Alpha158 无财务数据** |
| 股息 | **DividendYield** | **新增**，Alpha158 无股息数据 |
| 现金流 | **OperatingCashFlowYield** | **新增**，Alpha158 无现金流数据 |
| 运营效率 | **AssetTurnover** | **新增**，Alpha158 无运营效率数据 |
| 财务杠杆 | **DebtToEquity** | **新增**，Alpha158 无财务杠杆数据 |

---

## 合并因子总览

以下按类别列出了与 Alpha158 合并后的所有 41 个因子，每个标注 `[+Alpha158]` 或 `[独立]`。

| # | 因子名称 | 类型 | 数据维度 | 评估 | Alpha158 |
|---|----------|------|----------|------|----------|
| 1 | MediumTermMomentum_20d | 动量 | 价格 | ✅ | `[+Alpha158 ROC20]` |
| 2 | 20_day_reversal | 反转 | 价格 | ✅ | `[+Alpha158 ROC20]` |
| 3 | RealizedVolatility_20d | 波动率 | 价格 | ✅ | `[独立]` |
| 4 | RSI_14d | 震荡 | 价格 | ✅ | `[+Alpha158 SUMP]` |
| 5 | 5_day_volume_change | 流动性 | 成交量 | ✅ | `[独立]` |
| 6 | trailing_PE_ratio | 估值 | pe_ttm | ✅ | `[独立]` |
| 7 | obv_slope_10day | 量价 | 成交量+价格 | ✅ | `[独立]` |
| 8 | sharpe_10day | 风险调整 | 价格 | ✅ | `[独立]` |
| 9 | momentum_5d | 动量 | 价格 | ✅ | `[+Alpha158 ROC5]` |
| 10 | reversal_1d | 反转 | 价格 | ✅ | `[独立]` |
| 11 | volume_ratio_5d | 流动性 | 成交量 | ✅ | `[+Alpha158 VMA5]` |
| 12 | intraday_volatility | 波动率 | 价格(高低) | ✅ | `[+Alpha158 KLEN]` |
| 13 | volume_weighted_momentum_5d | 量价 | 成交量+价格 | ✅ | `[独立]` |
| 14 | roe | 质量 | roe_yearly | ✅ | `[独立]` |
| 15 | earnings_yield | 估值 | eps+close | ✅ | `[独立]` |
| 16 | net_profit_margin | 质量 | netprofit_margin | ✅ | `[独立]` |
| 17 | momentum_10d | 动量 | 价格 | ✅ | `[+Alpha158 ROC10]` |
| 18 | vwap_deviation_10d | 量价 | VWAP+价格 | ✅ | `[独立]` |
| 19 | avg_normalized_range_5d | 波动率 | 价格(高低) | ✅ | `[独立]` |
| 20 | turnover_trend | 流动性 | 换手率 | ✅ | `[独立]` |
| 21 | vwap_deviation_5d | 量价 | VWAP+价格 | ✅ | `[独立]` |
| 22 | reversal_2d | 反转 | 价格 | ✅ | `[独立]` |
| 23 | volume_ratio_5d_20d | 流动性 | 成交量 | ✅ | `[独立]` |
| 24 | reversal_5d | 反转 | 价格 | ✅ | `[+Alpha158 ROC5]` |
| 25 | PB_Ratio | 估值 | pb | ✅ | `[独立]` |
| 26 | Momentum_Vol_Adjusted_20 | 风险调整 | 价格 | ✅ | `[独立]` |
| 27 | Sector_Relative_PB | 估值(截面) | pb | ✅ | `[独立]` |
| 28 | avg_volume_ratio_20d | 流动性 | 成交量 | ✅ | `[+Alpha158 VMA20]` |
| 29 | book_to_price | 估值 | pb | ✅ | `[独立]` |
| 30 | risk_adjusted_momentum_5d_20d | 风险调整 | 价格 | ✅ | `[独立]` |
| 31 | Liquidity_Turnover_5d | 流动性 | 换手率 | ✅ | `[独立]` |
| 32 | Size | 市值/规模 | total_mv | ✅ | `[独立]` |
| 33 | Turnover | 流动性 | 换手率 | ✅ | `[独立]` |
| 34 | Volatility_5d | 波动率 | 价格 | ✅ | `[独立]` |
| 35 | Volatility_10d | 波动率 | 价格 | ✅ | `[独立]` |
| 36 | avg_turnover_10d | 流动性 | 换手率 | ✅ | `[独立]` |
| 37 | PriceToSales | 估值 | ps_ttm | ✅ | `[独立]` |
| 38 | DividendYield | 股息 | dv_ratio | ✅ | `[独立]` |
| 39 | OperatingCashFlowYield | 现金流 | ocfps+close | ✅ | `[独立]` |
| 40 | AssetTurnover | 运营效率 | revenue+assets | ✅ | `[独立]` |
| 41 | DebtToEquity | 财务杠杆 | liab+equity | ✅ | `[独立]` |

**统计**: 41 个因子中，9 个与 Alpha158 重叠（均可在 Alpha158 中由等效公式替代），**32 个为独立因子**（Alpha158 无等效因子）。

---

## 因子详情

### 1. MediumTermMomentum_20d `[+Alpha158 ROC20]`

- **类型**：动量因子 (Momentum Factor)
- **描述**：20日收益率，反映中期动量效应；正值表示趋势走强。
- **与 Alpha158 关系**：`momentum_20d = close_t/close_{t-20} - 1 = 1/ROC20 - 1`。与 ROC20 提供完全相同的信息（单调变换）。若已使用 Alpha158 的 ROC20，可去除此因子。
- **公式**：

  $$M_t = \frac{close_t}{close_{t-20}} - 1$$

- **变量**：
  - $close_t$：第 t 日收盘价
  - $close_{t-20}$：20 个交易日前收盘价
- **评估反馈**：代码执行无错误，输出格式正确（MultiIndex [datetime, instrument]、单列 float64），计算值与公式定义一致。

---

### 2. 20_day_reversal `[+Alpha158 ROC20]`

- **类型**：反转因子 (Reversal Factor)
- **描述**：20日反转效应，定义为 20 日价格收益率的负值，捕捉中期均值回复特性。
- **与 Alpha158 关系**：`reversal_20d = 1 - close_t/close_{t-20} = 1 - 1/ROC20`。与 ROC20 和 momentum_20d 完全共线（linear dependency）。在包含 ROC20 的 Alpha158 体系中冗余。
- **公式**：

  $$R_t = -\left(\frac{P_t}{P_{t-20}} - 1\right)$$

---

### 3. RealizedVolatility_20d `[独立]`

- **类型**：波动率因子 (Volatility Factor)
- **描述**：20日年化已实现波动率（对数收益率标准差 × √252），衡量风险和活跃程度。
- **与 Alpha158 关系**：Alpha158 的 STD20 = Std($close, 20)/$close 是价格水平的变异系数，本因子用对数收益率标准差。**两者互补**——本因子捕捉收益率波动，STD20 捕捉价格水平波动。
- **公式**：

  $$\sigma_t = \sqrt{ \frac{1}{19} \sum_{i=0}^{19} (r_{t-i} - \bar{r})^2 } \times \sqrt{252}$$

  其中 $r_t = \frac{close_t}{close_{t-1}} - 1$，$\bar{r} = \frac{1}{20} \sum_{i=0}^{19} r_{t-i}$

---

### 4. RSI_14d `[+Alpha158 SUMP]`

- **类型**：震荡因子 (Oscillator Factor)
- **描述**：14日相对强弱指数 (RSI)，衡量超买/超卖状态；RSI > 70 表示超买，RSI < 30 表示超卖。
- **与 Alpha158 关系**：`RSI = SUMP × 100`。Alpha158 有 SUMP10 和 SUMP20，无 SUMP14。**公式相同、窗口不同**——14 日窗口是传统 RSI 标准参数，可与 Alpha158 的 SUMP10/20 互补。
- **公式**：

  $$RSI_t = 100 - \frac{100}{1 + RS_t}$$

  $$RS_t = \frac{\frac{1}{14} \sum_{i=0}^{13} \max(\Delta close_{t-i}, 0)}{\frac{1}{14} \sum_{i=0}^{13} \max(-\Delta close_{t-i}, 0)}$$

---

### 5. 5_day_volume_change `[独立]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：5日成交量变化率，衡量交易活跃度的短期变动。
- **与 Alpha158 关系**：Alpha158 **无成交量 ROC** 算子。VSUMP/VSUMN/VSUMD 统计成交量上涨/下跌天数的比例，不直接度量成交量变化幅度。**独立因子**。
- **公式**：

  $$V_t = \frac{volume_t}{volume_{t-5}} - 1$$

---

### 6. trailing_PE_ratio `[独立]`

- **类型**：估值因子 (Value Factor)
- **描述**：滚动市盈率（PE TTM），最经典的价值衡量指标。低 PE 通常意味着股票可能被低估。
- **与 Alpha158 关系**：Alpha158 **完全不含估值/基本面数据**，仅使用价量。**独立因子，与 Alpha158 互补**。
- **公式**：

  $$Factor_t = PE\_TTM_t$$

- **数据来源**：cn_extra_data 估值数据中的 `$pe_ttm` 字段

---

### 7. obv_slope_10day `[独立]`

- **类型**：量价因子 (Volume-Price Factor)
- **描述**：能量潮（OBV）10日斜率，反映成交量的趋势变化，捕捉"聪明钱"流向信号。
- **与 Alpha158 关系**：Alpha158 的 CORR/CORD 度量价量相关性，不计算 OBV 累积量。**独立因子**。
- **公式**：

  $$OBV_t = \sum_{i=1}^{t} V_i \cdot \text{sgn}(P_i - P_{i-1})$$

  $$\beta_t = \frac{\sum_{i=0}^{9} (i - 4.5)(OBV_{t-i} - \overline{OBV}_{t,10})}{\sum_{i=0}^{9} (i - 4.5)^2}$$

---

### 8. sharpe_10day `[独立]`

- **类型**：风险调整因子 (Risk-Adjusted Factor)
- **描述**：10日滚动 Sharpe 比率。用过去 10 个交易日的日均对数收益率除以其标准差。
- **与 Alpha158 关系**：Alpha158 **无 Sharpe 比率**。Alpha158 有 SUMP/SUMN/SUMD 度量涨跌比例，有 STD 度量价格波动，但不计算收益率/波动率比值。**独立因子**。
- **公式**：

  $$S_t = \frac{\frac{1}{10}\sum_{i=0}^{9} r_{t-i}}{\sqrt{\frac{1}{9}\sum_{i=0}^{9}(r_{t-i} - \bar{r})^2}}$$

  $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

---

### 9. momentum_5d `[+Alpha158 ROC5]`

- **类型**：动量因子 (Momentum Factor)
- **描述**：5日价格动量。
- **与 Alpha158 关系**：`momentum_5d = 1/ROC5 - 1`，与 ROC5 等价。**可被 ROC5 替代**。
- **公式**：

  $$MOM_t = \frac{close_t}{close_{t-5}} - 1$$

---

### 10. reversal_1d `[独立]`

- **类型**：反转因子 (Reversal Factor)
- **描述**：1日反转因子，捕捉隔日均值回复效应。
- **与 Alpha158 关系**：Alpha158 **无 1 日窗口 ROC**（ROC 最小窗口为 5）。CNTP/CNTN 仅统计涨跌天数，不度量幅度。**独立因子**。
- **公式**：

  $$REV_t = -\left(\frac{close_t}{close_{t-1}} - 1\right)$$

---

### 11. volume_ratio_5d `[+Alpha158 VMA5]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：5日量比，定义为当日成交量除以前 5 个交易日平均成交量。
- **与 Alpha158 关系**：`volume_ratio_5d = volume_t / mean(volume_{t-1..t-5})`，VMA5 = `mean(volume_{t-4..t}) / volume_t`（含当日）。**高度相似**，仅分母窗口偏移一日。
- **公式**：

  $$VR_t = \frac{volume_t}{\frac{1}{5} \sum_{i=1}^{5} volume_{t-i}}$$

---

### 12. intraday_volatility `[+Alpha158 KLEN]`

- **类型**：波动率因子 (Volatility Factor)
- **描述**：日内振幅因子 (high - low) / close。
- **与 Alpha158 关系**：KLEN = `(high-low)/open`。**概念相同、分母不同**（close vs open）。两者高度相关，可保留其一；若保留本因子，关注收盘价归一化的振幅在涨跌停时更有界。
- **公式**：

  $$IV_t = \frac{high_t - low_t}{close_t}$$

---

### 13. volume_weighted_momentum_5d `[独立]`

- **类型**：量价因子 (Volume-Price Factor)
- **描述**：成交量加权 5 日动量，成交量越大的交易日其收益率权重越高。
- **与 Alpha158 关系**：Alpha158 的 WVMA 度量量加权收益率的**波动率**，不直接度量加权收益率本身。**独立因子**。
- **公式**：

  $$VWMOM_t = \frac{\sum_{i=1}^{5} volume_{t-i} \cdot R_{t-i}}{\sum_{i=1}^{5} volume_{t-i}}, \quad R_{t-i} = \frac{close_{t-i}}{close_{t-i-1}} - 1$$

---

### 14. roe `[独立]`

- **类型**：质量因子 (Quality Factor)
- **描述**：净资产收益率 (ROE)，衡量公司运用股东权益创造利润的效率。
- **与 Alpha158 关系**：Alpha158 **不含财务数据**。**完全独立，与 Alpha158 互补**。
- **公式**：

  $$ROE_t = \$roe\_yearly_t$$

- **数据来源**：cn_extra_data 财务数据中的 `$roe_yearly` 字段

---

### 15. earnings_yield `[独立]`

- **类型**：估值因子 (Value Factor)
- **描述**：盈利收益率（E/P），PE 的倒数。可与债券收益率直接比较（FED模型）。
- **与 Alpha158 关系**：使用 `$eps`（基本面）+ `$close`（行情），Alpha158 无 `$eps`。**独立因子**。
- **公式**：

  $$EY_t = \frac{\$eps_t}{close_t}$$

---

### 16. net_profit_margin `[独立]`

- **类型**：质量因子 (Quality Factor)
- **描述**：净利润率，反映公司在扣除所有成本费用后的最终盈利能力。
- **与 Alpha158 关系**：Alpha158 不含财务数据。**独立因子**。
- **公式**：

  $$NPM_t = \$netprofit\_margin_t$$

---

### 17. momentum_10d `[+Alpha158 ROC10]`

- **类型**：动量因子 (Momentum Factor)
- **描述**：10日价格动量，填补 5 日和 20 日动量之间的窗口空白。
- **与 Alpha158 关系**：`momentum_10d = 1/ROC10 - 1`。**等价于 ROC10**。
- **公式**：

  $$MOM_{10,t} = \frac{close_t}{close_{t-10}} - 1$$

---

### 18. vwap_deviation_10d `[独立]`

- **类型**：量价因子 (Volume-Price Factor) / 均值回复因子
- **描述**：收盘价相对于过去 10 日 VWAP 均值的偏离比例。
- **与 Alpha158 关系**：Alpha158 的 VWAP0 = `$vwap/$close` 是**单日** VWAP 相对收盘价。本因子比较收盘价与**多日平均 VWAP**，捕捉中期均值回复。**独立因子**。
- **公式**：

  $$VWAP\_dev_{10,t} = \frac{close_t}{\frac{1}{10}\sum_{i=0}^{9} vwap_{t-i}} - 1$$

---

### 19. avg_normalized_range_5d `[独立]`

- **类型**：波动率因子 (Volatility Factor)
- **描述**：5日平均归一化振幅，定义为本因子通过平滑处理降低了日内噪音。
- **与 Alpha158 关系**：与 KLEN 的 5 日均值类似。Alpha158 无此因子，但可由 KLEN 的 5日 MA 近似。**弱重叠**。
- **公式**：

  $$ANR_{5,t} = \frac{1}{5}\sum_{i=0}^{4}\frac{high_{t-i} - low_{t-i}}{close_{t-i}}$$

---

### 20. turnover_trend `[独立]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：换手率趋势因子，短期 (5日) 与长期 (20日) 平均换手率的相对差异。
- **与 Alpha158 关系**：Alpha158 **不使用换手率**（turnover）。换手率已标准化为流通股本比例，跨股票可比性更强。**独立因子**。
- **公式**：

  $$TO\_trend_t = \frac{\frac{1}{5}\sum_{i=0}^{4} turnover_{t-i} - \frac{1}{20}\sum_{i=0}^{19} turnover_{t-i}}{\frac{1}{20}\sum_{i=0}^{19} turnover_{t-i}}$$

- **数据来源**：cn_extra_data 行情数据中的 `$turnover` / `$turnover_f` 字段

---

### 21. vwap_deviation_5d `[独立]`

- **类型**：量价因子
- **描述**：5日 VWAP 偏离度，对短期价格偏离更敏感。
- **与 Alpha158 关系**：同 vwap_deviation_10d，**独立因子**。
- **公式**：

  $$VWAP\_dev_{5,t} = \frac{close_t}{\frac{1}{5}\sum_{i=0}^{4} vwap_{t-i}} - 1$$

---

### 22. reversal_2d `[独立]`

- **类型**：反转因子 (Reversal Factor)
- **描述**：2日反转因子。填补 1 日反转和 20 日反转之间的窗口空白。
- **与 Alpha158 关系**：Alpha158 **无 2 日 ROC**。**独立因子**。
- **公式**：

  $$REV_{2,t} = -\left(\frac{close_t}{close_{t-2}} - 1\right)$$

---

### 23. volume_ratio_5d_20d `[独立]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：短期/长期成交量比率，5日均量 / 20日均量。信号更加平滑稳定。
- **与 Alpha158 关系**：相当于 VMA20/VMA5（两 VMA 之比），但 Alpha158 无 VMA 比值因子。**独立因子**。
- **公式**：

  $$VR\_5\_20_t = \frac{\frac{1}{5}\sum_{i=0}^{4} volume_{t-i}}{\frac{1}{20}\sum_{i=0}^{19} volume_{t-i}}$$

---

### 24. reversal_5d `[+Alpha158 ROC5]`

- **类型**：反转因子 (Reversal Factor)
- **描述**：5日反转因子，定义为负的 5 日收益率。
- **与 Alpha158 关系**：`reversal_5d = 1 - close_t/close_{t-5} = 1 - 1/ROC5`。与 ROC5 和 momentum_5d 共线。**冗余因子**。
- **公式**：

  $$REV_{5,t} = -\left(\frac{P_t}{P_{t-5}} - 1\right)$$

---

### 25. PB_Ratio `[独立]`

- **类型**：估值因子 (Value Factor)
- **描述**：市净率 (PB)，衡量公司市值相对于其净资产的倍数。
- **与 Alpha158 关系**：Alpha158 不含估值数据。**独立因子**。
- **公式**：

  $$PB_t = \$pb_t$$

---

### 26. Momentum_Vol_Adjusted_20 `[独立]`

- **类型**：风险调整因子 (Risk-Adjusted Factor) / 动量因子
- **描述**：20日波动率调整动量，定义为 20 日价格收益率除以 20 日历史波动率。
- **与 Alpha158 关系**：融合了动量（ROC20）和波动率两个维度的信息。Alpha158 无此复合因子。**独立因子**。
- **公式**：

  $$\text{MomVol}_{t} = \frac{r_{t,20}}{\sigma_{t,20}}$$

  $$r_{t,20} = \frac{P_t - P_{t-20}}{P_{t-20}}, \quad \sigma_{t,20} = \sqrt{\frac{1}{20}\sum_{i=0}^{19}(r_{t-i} - \bar{r})^2}$$

---

### 27. Sector_Relative_PB `[独立]`

- **类型**：估值因子 (Value Factor) / 截面相对估值
- **描述**：行业相对市净率，定义为个股 PB 减去同一板块的截面中位数 PB。消除板块间系统性估值差异。
- **与 Alpha158 关系**：截面因子需要**所有股票同时计算**，Alpha158 纯时间序列算子无法实现。**完全独立**。
- **公式**：

  $$\text{SectorRelPB}_t = \$pb_t - \text{median}_{\text{sector}}(\$pb_t)$$

---

### 28. avg_volume_ratio_20d `[+Alpha158 VMA20]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：20 日平均成交量比率 = volume_t / mean(volume, 20)。
- **与 Alpha158 关系**：`avg_volume_ratio_20d = 1 / VMA20`（严格倒数）。与 VMA20 提供相同信息。**可被 VMA20 替代**。
- **公式**：

  $$AVR20_t = \frac{V_t}{\frac{1}{20}\sum_{i=0}^{19} V_{t-i}}$$

---

### 29. book_to_price `[独立]`

- **类型**：估值因子 (Value Factor)
- **描述**：账面市值比 (B/P)，市净率 (PB) 的倒数。与 Fama-French HML 因子的构造方式一致。
- **与 Alpha158 关系**：使用 pb 数据，Alpha158 无估值数据。**独立因子**。
- **公式**：

  $$BP_t = \frac{1}{\$pb_t}$$

---

### 30. risk_adjusted_momentum_5d_20d `[独立]`

- **类型**：风险调整因子 (Risk-Adjusted Factor) / 动量因子
- **描述**：5 日风险调整动量 = (5日动量) / (年化 20 日已实现波动率)。对短期趋势变化更敏感。
- **与 Alpha158 关系**：Alpha158 无此复合因子。**独立因子**。
- **公式**：

  $$\text{RAMom}_{5,20,t} = \frac{\frac{close_t}{close_{t-5}} - 1}{\sqrt{252} \cdot \text{std}\big(\ln(close_i/close_{i-1}), i=t-19,\dots,t\big)}$$

---

### 31. Liquidity_Turnover_5d `[独立]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：5日平均换手率，反映短期交易活跃程度。换手率因子与 Alpha158 的成交量因子（VMA/VSTD）使用不同数据源（换手率 vs 成交量），提供互补信息。
- **与 Alpha158 关系**：Alpha158 的 VMA/VSTD 使用成交量（volume），本因子使用换手率（turnover）。换手率已按流通股本标准化，跨股票可比性更强。**独立因子**。
- **公式**：

  $$TO_t^{5} = \frac{1}{5}\sum_{i=1}^{5} turnover_{t-i+1}$$

- **变量**：
  - $turnover_t$：日换手率（`$turnover` 字段）
  - $5$：移动平均窗口
- **数据来源**：cn_extra_data 行情数据中的 `$turnover` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 32. Size `[独立]`

- **类型**：市值/规模因子 (Size Factor)
- **描述**：总市值的自然对数，衡量公司规模。小市值效应（Small-Cap Effect）是学术界最 robust 的异象之一。
- **与 Alpha158 关系**：Alpha158 不含市值数据。**完全独立，与 Alpha158 互补**。
- **公式**：

  $$\text{Size}_t = \ln(\text{TotalMV}_t)$$

- **变量**：
  - $\text{TotalMV}_t$：第 t 日总市值（`$total_mv` 字段）
- **数据来源**：cn_extra_data 估值数据中的 `$total_mv` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 33. Turnover `[独立]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：日换手率 = 成交量 / 自由流通股本。衡量股票交易活跃度的最直接指标。
- **与 Alpha158 关系**：Alpha158 使用成交量（volume）而非换手率。换手率已按流通股本标准化，跨股票可比性更强。**独立因子**。
- **公式**：

  $$\text{Turnover}_t = \frac{\text{Volume}_t}{\text{FreeShares}_t}$$

- **变量**：
  - $\text{Volume}_t$：第 t 日成交量（`$volume` 字段）
  - $\text{FreeShares}_t$：第 t 日自由流通股本（`$free_sh` 字段）
- **数据来源**：cn_extra_data 行情数据中的 `$volume`、`$free_sh` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 34. Volatility_5d `[独立]`

- **类型**：波动率因子 (Volatility Factor)
- **描述**：5日历史波动率，定义为过去5个交易日简单收益率的标准差。捕捉短期价格波动风险。
- **与 Alpha158 关系**：Alpha158 的 STD5 = Std($close,5)/$close 是价格水平的变异系数，本因子用收益率标准差。**两者互补**。
- **公式**：

  $$\sigma_t^{5} = \sqrt{\frac{1}{4}\sum_{i=1}^{5}(r_{t-i+1} - \bar{r}_t)^2}$$

  其中 $r_i = P_i/P_{i-1} - 1$，$\bar{r}$ 为5日收益率均值
- **数据来源**：cn_extra_data 行情数据中的 `$close` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 35. Volatility_10d `[独立]`

- **类型**：波动率因子 (Volatility Factor)
- **描述**：10日历史波动率，定义为过去10个交易日简单收益率的标准差。捕捉中期价格波动风险，填补 5 日和 20 日波动率之间的窗口空白。
- **与 Alpha158 关系**：Alpha158 的 STD10 = Std($close,10)/$close 是价格水平的变异系数，本因子用收益率标准差。**两者互补**。
- **公式**：

  $$\sigma_t^{10} = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(r_{t-i+1} - \bar{r}_t)^2}$$

  其中 $r_i = P_i/P_{i-1} - 1$，$\bar{r}$ 为10日收益率均值
- **数据来源**：cn_extra_data 行情数据中的 `$close` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 36. avg_turnover_10d `[独立]`

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：10日平均换手率，衡量交易活跃度的中期均值。与 Liquidity_Turnover_5d（5 日窗口）互补，提供不同时间尺度下的流动性视图。
- **与 Alpha158 关系**：Alpha158 使用成交量（volume）而非换手率（turnover），本因子使用 `$turnover` 字段，属于独立维度。
- **公式**：

  $$\text{AvgTurnover}_{10d} = \frac{1}{10} \sum_{i=t-9}^{t} \text{turnover}_i$$

- **变量**：
  - $turnover_i$：日换手率（`$turnover` 字段）
- **数据来源**：cn_extra_data 中的 `$turnover` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 37. PriceToSales `[独立]`

- **类型**：估值因子 (Value Factor)
- **描述**：市销率 (P/S)，定义为滚动 12 个月市销率。衡量公司市值相对于其销售收入的比例，常用于评估成长型公司的估值水平。
- **公式**：

  $$P/S_t = \$ps\_ttm_t$$

- **变量**：
  - $\$ps\_ttm_t$：第 t 日滚动 12 个月市销率
- **数据来源**：cn_extra_data 估值数据中的 `$ps_ttm` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 38. DividendYield `[独立]`

- **类型**：股息因子 (Dividend Factor)
- **描述**：股息率，定义为最近股息率。衡量股东通过现金分红获得的收益水平，是价值投资的重要参考指标。
- **公式**：

  $$\text{DividendYield}_t = \$dv\_ratio_t$$

- **变量**：
  - $\$dv\_ratio_t$：第 t 日股息率
- **数据来源**：cn_extra_data 中的 `$dv_ratio` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 39. OperatingCashFlowYield `[独立]`

- **类型**：现金流因子 (Cash Flow Factor)
- **描述**：经营现金流收益率，定义为每股经营现金流除以收盘价。衡量公司经营活动产生的现金流相对于其市值的比例，比净利润更难被操纵，是更可靠的价值指标。
- **公式**：

  $$\text{OCFYield}_t = \frac{\$ocfps_t}{\$close_t}$$

- **变量**：
  - $\$ocfps_t$：第 t 日每股经营现金流
  - $\$close_t$：第 t 日收盘价
- **数据来源**：cn_extra_data 中的 `$ocfps`、`$close` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 40. AssetTurnover `[独立]`

- **类型**：运营效率因子 (Operating Efficiency Factor)
- **描述**：资产周转率，定义为总收入除以总资产。衡量公司利用其资产产生收入的效率，周转率越高表示资产使用效率越好。
- **公式**：

  $$\text{AssetTurnover}_t = \frac{\$revenue_t}{\$total\_assets_t}$$

- **变量**：
  - $\$revenue_t$：第 t 日总收入
  - $\$total\_assets_t$：第 t 日总资产
- **数据来源**：cn_extra_data 中的 `$revenue`、`$total_assets` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

### 41. DebtToEquity `[独立]`

- **类型**：财务杠杆因子 (Financial Leverage Factor)
- **描述**：资产负债率（D/E），定义为总负债除以总权益。衡量公司财务杠杆水平和长期偿债风险，是信用分析和价值评估的核心指标。
- **公式**：

  $$\text{DebtToEquity}_t = \frac{\$total\_liab_t}{\$total\_equity_t}$$

- **变量**：
  - $\$total\_liab_t$：第 t 日总负债
  - $\$total\_equity_t$：第 t 日总权益
- **数据来源**：cn_extra_data 中的 `$total_liab`、`$total_equity` 字段
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]`

---

## Alpha158 完整因子列表（供参考）

Alpha158 = **9 KBar + 4 Price + 145 Rolling = 158 因子**，全部基于价量数据。

| 类别 | 因子 | 数量 |
|------|------|------|
| **KBar** | KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2 | 9 |
| **Price** | OPEN0, HIGH0, LOW0, VWAP0 | 4 |
| **Rolling @ [5,10,20,30,60]** | ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD | 145 (=29×5) |

## 使用建议

1. **纯价量模型**：Alpha158 (158) + 独立价量因子 (13) = **171 个价量因子**
2. **加入基本面**：Alpha158 (158) + 独立价量 (13) + 基本面/估值 (12) + 财务/杠杆/现金流 (7) = **190 个因子**
3. **精简去重**：Alpha158 (158) + 独立价量 (13) + 基本面/估值 (12) + 财务/杠杆/现金流 (7) − 重叠因子 (10) = **180 个去重因子**
4. Alpha158 通过 `Alpha158` handler 加载；独立因子通过 `AlphaExtra` handler（`qlib/contrib/data/handler_extra.py`）加载，两者可在 YAML 中 `handler` 列表合并使用
