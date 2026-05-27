# 实践因子列表

> 从 `new_factor.md` 精选，用于 `run_new_factor_practice --new_factor_only` 模式。
> 数据源：`daily_pv_all.h5`（58 个原始字段，由 `generate.py` 从 `extra_data` 构建）。
> 构建：`scripts/practice/build_features_from_h5.py` → qlib 二进制格式 → 截面 z-score 归一化。
> 更新：2026-05-24 | **版本：v3.0（v2.0 34 因子 + 新增 8 个，覆盖非流动性/FCF/偏度/盈利能力/反转）**

---

## v2.0 改进说明

基于对 33 个原版因子的系统分析，本版本做了以下改进：

### 剔除的冗余因子（5 个）

| 剔除因子 | 原因 | 替代 |
|---------|------|------|
| Volatility_5d | 与 Volatility_10d / RealizedVolatility_20d 高度共线（同是收益率标准差，仅窗口不同） | 保留 RealizedVolatility_20d + avg_normalized_range_5d |
| Volatility_10d | 同上，窗口介于 5d 与 20d 之间 | 保留 RealizedVolatility_20d |
| Liquidity_Turnover_5d | 与 Turnover / avg_turnover_10d 高度共线（换手率均值的不同窗口） | 保留 Turnover + turnover_trend |
| avg_turnover_10d | 同上 | 保留 Turnover + turnover_trend |
| volume_ratio_5d_20d | 与 turnover_trend 构造逻辑重复（量比 vs 换手率趋势，MA5/MA20 之比） | 保留 turnover_trend |

### 新增的高 alpha 因子（6 个）

| 新增因子 | 类型 | 数据来源 | A 股预期 IC |
|---------|------|---------|-----------|
| SUE | 盈余质量 | `$eps_yoy` | ★★★★★（A 股最强截面因子之一） |
| AssetGrowth | 投资 | `$assets_yoy` | ★★★★（投资效应 -IC） |
| AccrualsRatio | 盈余质量 | `$n_income`, `$ocf`, `$total_assets` | ★★★★（应计异象） |
| DebtToEquity | 财务杠杆 | `$total_liab`, `$total_equity` | ★★★（水平值+差分互补） |
| MAX_20d | 行为金融 | `$close` | ★★★★（最大日收益反转） |
| RevenueGrowth | 成长 | `$revenue_yoy` | ★★★（营收增速） |

### 改进汇总

| 维度 | v1.0（原版） | v2.0（改进版） |
|------|-------------|---------------|
| 因子总数 | 33 | 34 |
| 波动率类占比 | 12%（4/33） | 6%（2/34） |
| 流动性/换手类占比 | 18%（6/33） | 9%（3/34） |
| 新增盈余质量 | 无 | 2 个（SUE, Accruals） |
| 新增投资/成长 | 无 | 2 个（AssetGrowth, RevenueGrowth） |
| 新增行为金融 | 无 | 1 个（MAX_20d） |
| 杠杆:水平+差分 | 仅有 Delta | 水平 + Delta 双版本 |

---

## v3.0 改进说明

基于 v2.0 的 34 因子集，本版本重点弥补以下覆盖缺口：

### 新增的高 alpha 因子（8 个）

| 新增因子 | 类型 | 数据来源 | A 股预期 IC | 文献依据 |
|---------|------|---------|-----------|---------|
| Amihud_20d | 非流动性 | `$amount`, `$close` | ★★★★★ | Amihud (2002), A 股 top-5 因子 |
| FCF_Yield | 自由现金流 | `$fcf`, `$circ_mv` | ★★★★ | 比 OCF 更真实，高 FCF→低估 |
| ROA | 盈利能力 | `$roa_yearly` | ★★★★ | 与 ROE 互补，不受杠杆影响 |
| Skewness_20d | 收益分布 | `$close` | ★★★★ | 负偏度→崩盘风险溢价 |
| NetProfitGrowth | 成长 | `$netprofit_yoy` | ★★★★ | 与 RevenueGrowth 互补 |
| EPS_Quality | 盈余质量 | `$npta`, `$eps` | ★★★★ | 非经常性损益比例，Xie (2001) |
| TurnoverVol_20d | 流动性动态 | `$turnover` | ★★★ | 换手率离散度，Chordia et al. (2001) |
| Reversal_3d | 反转 | `$close` | ★★★★ | 填补 2d-5d 反转窗口空白 |

### 改进汇总

| 维度 | v2.0 | v3.0 |
|------|------|------|
| 因子总数 | 34 | 42 |
| FCF 覆盖 | 无 | 1 个（FCF_Yield） |
| 非流动性 | 无 | 1 个（Amihud_20d） |
| 收益率分布 | 仅 MAX_20d | +Skewness_20d（互补） |
| 盈利能力水平值 | 仅 Delta 差分 | +ROA 水平值 |
| 净利润增长 | 无 | +NetProfitGrowth |
| 盈余质量 | SUE, Accruals | +EPS_Quality（非经常性损益） |
| 换手率动态 | 无 | +TurnoverVol_20d |
| 反转窗口 | 1d, 2d | +3d（填补缺口） |

---

## 剔除说明

以下因子已从实践集中剔除（与 v1.0 相同）：

- **book_to_price** — 与 PB_Ratio 完全共线 (B/P = 1/PB)，保留 PB_Ratio
- **momentum_5d/10d/20d** — 与 Alpha158 ROC5/ROC10/ROC20 等价
- **reversal_5d/20d** — 与 ROC5/ROC20 + 反转等价，共线
- **rsi_14d** — 与 Alpha158 SUMP 等价
- **intraday_volatility** — 与 Alpha158 KLEN 等价
- **volume_ratio_5d** — 与 Alpha158 VMA5 等价
- **avg_volume_ratio_20d** — 与 Alpha158 VMA20 等价
- **20_day_reversal** — 与 ROC20 共线
- **MediumTermMomentum_20d** — 与 ROC20 等价

## 财务比率季度差分

以下因子使用季度差分（当期值 − 60 交易日前值）替代水平值：

| 差分因子 | 原始水平值 | 说明 |
|---------|-----------|------|
| Delta_roe | roe_yearly | ROE 季度变化 |
| Delta_net_profit_margin | netprofit_margin | 净利润率季度变化 |
| Delta_DebtToEquity | total_liab/total_equity | 杠杆率季度变化 |
| Delta_AssetTurnover | revenue/total_assets | 资产周转率季度变化 |
| Delta_OperatingCashFlowYield | ocfps/close | 现金流收益率季度变化 |

## 估值因子行业中性化

以下截面因子做行业中性化处理（个股值 − 同行业截面中位数）：

| 截面因子 | 源因子 | 行业分类 |
|---------|-------|---------|
| Sector_Relative_PB | PB_Ratio | 申万一级行业 (28 个大类) |
| Sector_Relative_PE | trailing_PE_ratio | 同上 |
| Sector_Relative_DividendYield | DividendYield | 同上 |

**行业分类来源**: 申万 2021 版一级行业 (SW2021)，通过 `tushare/fetch_sw_industry.py` 从 TuShare 获取。
申万未覆盖的股票（北交所等）自动回退为板块分类（主板/中小板/创业板/科创板）。
当同行业同日期有效同行股票 < 5 只时，排除自身计算中位数。

## 因子总览

| # | 因子名称 | 类型 | 数据维度 | 数据来源（H5 字段） | v3.0 |
|---|----------|------|----------|-------------------|------|
| 1 | RealizedVolatility_20d | 波动率 | 价格 | `$close` | ✓ |
| 2 | volume_change_5d | 流动性 | 成交量 | `$volume` | ✓ |
| 3 | trailing_PE_ratio | 估值 | pe_ttm | `$pe_ttm` | ✓ |
| 4 | obv_slope_10d | 量价 | 成交量+价格 | `$volume`, `$close` | ✓ |
| 5 | sharpe_10d | 风险调整 | 价格 | `$close` | ✓ |
| 6 | reversal_1d | 反转 | 价格 | `$close` | ✓ |
| 7 | volume_weighted_momentum_5d | 量价 | 成交量+价格 | `$volume`, `$close` | ✓ |
| 8 | Delta_roe | 质量 | roe_yearly | `$roe_yearly` | ✓ |
| 9 | earnings_yield | 估值 | eps+close | `$eps`, `$close` | ✓ |
| 10 | Delta_net_profit_margin | 质量 | netprofit_margin | `$netprofit_margin` | ✓ |
| 11 | vwap_deviation_10d | 量价 | VWAP+价格 | `$vwap`, `$close` | ✓ |
| 12 | avg_normalized_range_5d | 波动率 | 价格(高低) | `$high`, `$low`, `$close` | ✓ |
| 13 | turnover_trend | 流动性 | 换手率 | `$turnover` | ✓ |
| 14 | vwap_deviation_5d | 量价 | VWAP+价格 | `$vwap`, `$close` | ✓ |
| 15 | reversal_2d | 反转 | 价格 | `$close` | ✓ |
| 16 | PB_Ratio | 估值 | pb | `$pb` | ✓ |
| 17 | momentum_vol_adjusted_20 | 风险调整 | 价格 | `$close` | ✓ |
| 18 | Sector_Relative_PB | 估值(截面) | pb | `$pb` | ✓ |
| 19 | Sector_Relative_PE | 估值(截面) | pe_ttm | `$pe_ttm` | ✓ |
| 20 | Sector_Relative_DividendYield | 估值(截面) | dv_ratio | `$dv_ratio` | ✓ |
| 21 | risk_adjusted_momentum_5d_20d | 风险调整 | 价格 | `$close` | ✓ |
| 22 | Size | 市值/规模 | total_mv | `$total_mv` | ✓ |
| 23 | Turnover | 流动性 | 换手率 | `$volume`, `$free_sh` | ✓ |
| 24 | PriceToSales | 估值 | ps_ttm | `$ps_ttm` | ✓ |
| 25 | DividendYield | 股息 | dv_ratio | `$dv_ratio` | ✓ |
| 26 | Delta_OperatingCashFlowYield | 现金流 | ocfps+close | `$ocfps`, `$close` | ✓ |
| 27 | Delta_AssetTurnover | 运营效率 | revenue+assets | `$revenue`, `$total_assets` | ✓ |
| 28 | Delta_DebtToEquity | 财务杠杆 | liab+equity | `$total_liab`, `$total_equity` | ✓ |
| 29 | SUE | 盈余质量 | eps_yoy | `$eps_yoy` | v2.0 |
| 30 | AssetGrowth | 投资 | assets_yoy | `$assets_yoy` | v2.0 |
| 31 | AccrualsRatio | 盈余质量 | n_income+ocf+assets | `$n_income`, `$ocf`, `$total_assets` | v2.0 |
| 32 | DebtToEquity | 财务杠杆 | liab+equity | `$total_liab`, `$total_equity` | v2.0 |
| 33 | MAX_20d | 行为金融 | 价格 | `$close` | v2.0 |
| 34 | RevenueGrowth | 成长 | revenue_yoy | `$revenue_yoy` | v2.0 |
| 35 | Amihud_20d | 非流动性 | amount+close | `$amount`, `$close` | 新增 |
| 36 | FCF_Yield | 自由现金流 | fcf+circ_mv | `$fcf`, `$circ_mv` | 新增 |
| 37 | ROA | 盈利能力 | roa_yearly | `$roa_yearly` | 新增 |
| 38 | Skewness_20d | 收益分布 | 价格 | `$close` | 新增 |
| 39 | NetProfitGrowth | 成长 | netprofit_yoy | `$netprofit_yoy` | 新增 |
| 40 | EPS_Quality | 盈余质量 | npta+eps | `$npta`, `$eps` | 新增 |
| 41 | TurnoverVol_20d | 流动性动态 | 换手率 | `$turnover` | 新增 |
| 42 | Reversal_3d | 反转 | 价格 | `$close` | 新增 |

**统计**: 42 个实践因子（含 3 个行业中性化截面因子，5 个季度差分因子，v2.0 新增 6 个，v3.0 新增 8 个）。
**v3.0 变更**: 新增 8 个高 alpha 因子，覆盖非流动性、FCF、偏度、盈利能力水平值、净利润增长、EPS 质量、换手率离散度、3 日反转。

## 归一化策略

| 归一化方法 | 适用因子 |
|-----------|---------|
| 截面 z-score（每日期，跨股票） | 全部 42 个因子（在 `build_features_from_h5.py` 中自动完成） |
| 行业中性化 | trailing_PE_ratio → Sector_Relative_PE; PB_Ratio → Sector_Relative_PB; DividendYield → Sector_Relative_DividendYield |
| 季度差分 | roe, netprofit_margin, DebtToEquity, AssetTurnover, OperatingCashFlowYield |
| 1%/99% 缩尾 | Delta_roe, Delta_net_profit_margin, SUE, AccrualsRatio, DebtToEquity, Amihud_20d, FCF_Yield, Skewness_20d, EPS_Quality |

---

## 因子详情

### 1. RealizedVolatility_20d

- **类型**：波动率因子 (Volatility Factor)
- **描述**：20 日年化已实现波动率，使用简单收益率的标准差乘以年化因子 √252。衡量个股的中期价格波动风险。
- **公式**：

  $$\sigma_t^{20} = \sqrt{\frac{1}{19} \sum_{i=0}^{19} (r_{t-i} - \bar{r}_{20})^2} \times \sqrt{252}$$

  其中：
  - $r_t = \dfrac{close_t}{close_{t-1}} - 1$：第 t 日简单收益率
  - $\bar{r}_{20} = \dfrac{1}{20} \sum_{i=0}^{19} r_{t-i}$：20 日平均收益率
  - $close_t$：第 t 日复权收盘价

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：new_factor.md #3

---

### 2. volume_change_5d

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：5 日成交量变化率，捕捉交易活跃度的短期突变。正值表示成交量放大，负值表示萎缩。
- **公式**：

  $$VC_t^{5} = \frac{volume_t}{volume_{t-5}} - 1$$

  其中：
  - $volume_t$：第 t 日成交量（股数）
  - $volume_{t-5}$：5 个交易日前成交量

- **数据来源**：H5 `$volume` → `daily.csv` 的 `vol` 字段
- **参考**：new_factor.md #5

---

### 3. trailing_PE_ratio

- **类型**：估值因子 (Value Factor)
- **描述**：滚动市盈率（PE TTM），最经典的价值指标。低 PE 通常表示股票被低估，但需结合行业和成长性判断。
- **公式**：

  $$PE\_TTM_t = \$pe\_ttm_t$$

  其中 $\$pe\_ttm_t$ 为 TuShare 提供的滚动市盈率（基于过去 12 个月净利润）。

- **数据来源**：H5 `$pe_ttm` → `daily_basic.csv` 的 `pe_ttm` 字段
- **注意**：此因子在截面归一化**前**会被行业中性化处理生成 `Sector_Relative_PE`
- **参考**：new_factor.md #6

---

### 4. obv_slope_10d

- **类型**：量价因子 (Volume-Price Factor)
- **描述**：能量潮（On-Balance Volume, OBV）的 10 日线性回归斜率。OBV 累积"聪明钱"流向，斜率反映资金流入/流出的趋势强度。
- **公式**：

  $$OBV_t = \sum_{i=1}^{t} volume_i \cdot \text{sgn}(close_i - close_{i-1})$$

  $$\beta_t^{OBV,10} = \frac{\sum_{i=0}^{9} (i - 4.5)(OBV_{t-i} - \overline{OBV}_{10})}{\sum_{i=0}^{9} (i - 4.5)^2}$$

  其中：
  - $\text{sgn}(x) = 1$ 当 $x > 0$，$= -1$ 当 $x < 0$，$= 0$ 当 $x = 0$
  - $volume_i$：第 i 日成交量
  - $\overline{OBV}_{10}$：10 日 OBV 均值

- **数据来源**：H5 `$volume`（`daily.csv` `vol`）, `$close`（`daily.csv` `close`）
- **参考**：new_factor.md #7

---

### 5. sharpe_10d

- **类型**：风险调整因子 (Risk-Adjusted Factor)
- **描述**：10 日滚动 Sharpe 比率。衡量单位波动率所获得的超额收益，正值表示风险调整后收益为正。
- **公式**：

  $$S_t^{10} = \frac{\bar{r}_{10}}{\sigma_{10}(r)}$$

  其中：
  - $\bar{r}_{10} = \dfrac{1}{10} \sum_{i=0}^{9} r_{t-i}$
  - $\sigma_{10}(r) = \sqrt{\dfrac{1}{9} \sum_{i=0}^{9} (r_{t-i} - \bar{r}_{10})^2}$
  - $r_t = \dfrac{close_t}{close_{t-1}} - 1$

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：new_factor.md #8

---

### 6. reversal_1d

- **类型**：反转因子 (Reversal Factor)
- **描述**：1 日反转 = 昨日收益率的负值。捕捉隔夜均值回复效应（A 股次日反转效应显著）。
- **公式**：

  $$REV_t^{1} = -\left(\frac{close_t}{close_{t-1}} - 1\right)$$

  其中 $close_t$ 为第 t 日收盘价。

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：new_factor.md #10

---

### 7. volume_weighted_momentum_5d

- **类型**：量价因子 (Volume-Price Factor)
- **描述**：成交量加权 5 日动量。高成交量日的收益率获得更高权重，比简单动量更能反映"聪明钱"的真实方向。
- **公式**：

  $$VWMOM_t = \frac{\sum_{i=1}^{5} volume_{t-i} \cdot r_{t-i}}{\sum_{i=1}^{5} volume_{t-i}}$$

  其中：
  - $r_t = \dfrac{close_t}{close_{t-1}} - 1$
  - $volume_t$：第 t 日成交量
  - 要求至少 3 个有效数据点，否则为 NaN

- **数据来源**：H5 `$volume`（`daily.csv` `vol`）, `$close`（`daily.csv` `close`）
- **参考**：new_factor.md #13

---

### 8. Delta_roe

- **类型**：质量因子 (Quality Factor)
- **描述**：ROE（净资产收益率）的季度差分。ROE 水平值具有强行业/市值偏向，差分后信号更纯净，直接反映盈利能力的边际变化。
- **公式**：

  $$\Delta ROE_t = ROE\_yearly_t - ROE\_yearly_{t-60}$$

  其中：
  - $ROE\_yearly_t$：第 t 日最新 ROE（年报数据，使用 `$roe_yearly`）
  - $t-60$：约 60 个交易日前（~1 季度）

- **数据来源**：H5 `$roe_yearly` → `fina_indicator.csv` 的 `roe_yearly` 字段
- **注意**：1%/99% 缩尾处理
- **参考**：new_factor.md #14

---

### 9. earnings_yield

- **类型**：估值因子 (Value Factor)
- **描述**：盈利收益率（E/P 比率），PE 的倒数。可直接与债券收益率对比（Fed Model），高 E/P 表示股票可能被低估。
- **公式**：

  $$EY_t = \frac{\$eps_t}{\$close_t}$$

  其中：
  - $\$eps_t$：第 t 日最新每股收益（TTM）
  - $\$close_t$：第 t 日收盘价

- **数据来源**：H5 `$eps`（`fina_indicator.csv` `eps`）, `$close`（`daily.csv` `close`）
- **参考**：new_factor.md #15

---

### 10. Delta_net_profit_margin

- **类型**：质量因子 (Quality Factor)
- **描述**：净利润率的季度差分。反映公司盈利能力的边际变化，上升趋势通常被市场正面解读。
- **公式**：

  $$\Delta NPM_t = netprofit\_margin_t - netprofit\_margin_{t-60}$$

  其中：
  - $netprofit\_margin_t$：第 t 日最新净利润率（`$netprofit_margin`）
  - $t-60$：60 个交易日前

- **数据来源**：H5 `$netprofit_margin` → `fina_indicator.csv` 的 `netprofit_margin` 字段
- **注意**：1%/99% 缩尾处理
- **参考**：new_factor.md #16

---

### 11. vwap_deviation_10d

- **类型**：量价因子 / 均值回复因子
- **描述**：收盘价相对于过去 10 日 VWAP 均值的偏离比例。当收盘价显著高于平均 VWAP 时可能超买，反之超卖。
- **公式**：

  $$VWAP\_dev_t^{10} = \frac{close_t}{\frac{1}{10}\sum_{i=0}^{9} vwap_{t-i}} - 1$$

  其中：
  - $vwap_t$：第 t 日成交量加权平均价
  - $close_t$：第 t 日收盘价

- **数据来源**：H5 `$vwap`（`daily.csv` 计算）, `$close`（`daily.csv` `close`）
- **参考**：new_factor.md #18

---

### 12. avg_normalized_range_5d

- **类型**：波动率因子 (Volatility Factor)
- **描述**：5 日平均归一化振幅 = $\frac{high - low}{close}$ 的 5 日均值。使用日内高低价差（而非收益率）度量波动率，含有 Alpha158 KLEN 未包含的收盘价归一化信息。
- **公式**：

  $$ANR_t^{5} = \frac{1}{5}\sum_{i=0}^{4} \frac{high_{t-i} - low_{t-i}}{close_{t-i}}$$

  其中 $high_t$, $low_t$, $close_t$ 分别为第 t 日最高价、最低价、收盘价。

- **数据来源**：H5 `$high`, `$low`, `$close` → `daily.csv` 的 `high`, `low`, `close` 字段
- **参考**：new_factor.md #19

---

### 13. turnover_trend

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：换手率趋势 = (短期均换手率 − 长期均换手率) / 长期均换手率。正值表示换手率在上升（交投活跃度增加），负值表示下降。
- **公式**：

  $$TO\_trend_t = \frac{MA_5(turnover) - MA_{20}(turnover)}{MA_{20}(turnover)}$$

  其中：
  - $MA_k(turnover) = \dfrac{1}{k} \sum_{i=0}^{k-1} turnover_{t-i}$
  - $turnover_t$：第 t 日换手率（`$turnover`）

- **数据来源**：H5 `$turnover` → `daily_basic.csv` 的 `turnover_rate` 字段
- **参考**：new_factor.md #20

---

### 14. vwap_deviation_5d

- **类型**：量价因子
- **描述**：收盘价相对于过去 5 日 VWAP 均值的偏离比例。相对于 vwap_deviation_10d 对短期价格偏离更敏感。
- **公式**：

  $$VWAP\_dev_t^{5} = \frac{close_t}{\frac{1}{5}\sum_{i=0}^{4} vwap_{t-i}} - 1$$

- **数据来源**：H5 `$vwap`, `$close`
- **参考**：new_factor.md #21

---

### 15. reversal_2d

- **类型**：反转因子 (Reversal Factor)
- **描述**：2 日反转因子。填补 1 日反转与中长期反转之间的窗口空白，捕捉 2 日级别的均值回复。
- **公式**：

  $$REV_t^{2} = -\left(\frac{close_t}{close_{t-2}} - 1\right)$$

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：new_factor.md #22

---

### 16. PB_Ratio

- **类型**：估值因子 (Value Factor)
- **描述**：市净率 = 收盘价 / 每股净资产。经典价值因子，低 PB 通常表示股票被低估。
- **公式**：

  $$PB_t = \$pb_t$$

- **数据来源**：H5 `$pb` → `daily_basic.csv` 的 `pb` 字段
- **注意**：此因子在截面归一化**前**会被行业中性化处理生成 `Sector_Relative_PB`
- **参考**：new_factor.md #25

---

### 17. momentum_vol_adjusted_20

- **类型**：风险调整因子 (Risk-Adjusted Momentum)
- **描述**：20 日波动率调整动量 = 20 日简单收益率 / 20 日收益率标准差。融合了方向（动量）和风险（波动率）两个维度，高波动股票的动量被调低。
- **公式**：

  $$MomVol_t^{20} = \frac{r_t^{20}}{\sigma_t^{20}}$$

  其中：
  - $r_t^{20} = \dfrac{close_t}{close_{t-20}} - 1$
  - $\sigma_t^{20} = \sqrt{\dfrac{1}{19} \sum_{i=0}^{19} (r_{t-i} - \bar{r}_{20})^2}$

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：new_factor.md #26

---

### 18. Sector_Relative_PB

- **类型**：估值因子（行业中性化截面因子）
- **描述**：个股 PB 减去同一申万一级行业的 PB 截面中位数。消除不同行业间的系统性估值差异（如银行股 PB 普遍低于科技股），使 PB 在同一行业内具有可比性。
- **公式**：

  $$\text{SectorRelPB}_{t, s} = \$pb_{t, s} - \text{median}_{p \in \text{industry}(s)}(\$pb_{t, p})$$

  其中 $\text{industry}(s)$ 为股票 s 所属申万一级行业（如银行、医药生物等）。

- **数据来源**：H5 `$pb` + `tushare/cn_data/sw_industry.csv`（申万 2021 版行业分类）
- **注意**：此因子在全部股票计算完成后统一计算；申万未覆盖的股票回退板块分类
- **参考**：new_factor.md #27

---

### 19. Sector_Relative_PE

- **类型**：估值因子（行业中性化截面因子）
- **描述**：个股 PE_TTM 减去同一申万一级行业的 PE_TTM 截面中位数。逻辑同 Sector_Relative_PB。
- **公式**：

  $$\text{SectorRelPE}_{t, s} = \$pe\_ttm_{t, s} - \text{median}_{p \in \text{industry}(s)}(\$pe\_ttm_{t, p})$$

- **数据来源**：H5 `$pe_ttm` + `tushare/cn_data/sw_industry.csv`

---

### 20. Sector_Relative_DividendYield

- **类型**：估值因子（行业中性化截面因子）
- **描述**：个股股息率减去同一申万一级行业的股息率截面中位数。银行等高股息行业的个股不会因此因子而天然偏高。
- **公式**：

  $$\text{SectorRelDY}_{t, s} = \$dv\_ratio_{t, s} - \text{median}_{p \in \text{industry}(s)}(\$dv\_ratio_{t, p})$$

- **数据来源**：H5 `$dv_ratio` + `tushare/cn_data/sw_industry.csv`

---

### 21. risk_adjusted_momentum_5d_20d

- **类型**：风险调整因子
- **描述**：5 日风险调整动量 = 5 日动量 / 年化 20 日波动率。对短期趋势变化更敏感，但对高波动股票给予惩罚。
- **公式**：

  $$\text{RAMom}_t = \frac{\dfrac{close_t}{close_{t-5}} - 1}{\sigma_t^{20} \times \sqrt{252}}$$

  其中 $\sigma_t^{20}$ 为 20 日收益率标准差。

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：new_factor.md #30

---

### 22. Size

- **类型**：市值/规模因子 (Size Factor)
- **描述**：总市值的自然对数。A 股小市值效应显著，但线性 log(market cap) 可能不足以捕捉非线性的市值溢价。
- **公式**：

  $$\text{Size}_t = \ln(\$total\_mv_t)$$

  其中 $\$total\_mv_t$ 为第 t 日总市值（元）。

- **数据来源**：H5 `$total_mv` → `daily_basic.csv` 的 `total_mv` 字段
- **参考**：new_factor.md #32

---

### 23. Turnover

- **类型**：流动性因子 (Liquidity Factor)
- **描述**：日换手率 = 成交量 / 自由流通股本。已标准化为流通股本比例，跨股票可比性优于原始成交量。
- **公式**：

  $$\text{Turnover}_t = \frac{\$volume_t}{\$free\_sh_t}$$

  其中：
  - $\$volume_t$：第 t 日成交量（股数）
  - $\$free\_sh_t$：第 t 日自由流通股本

- **数据来源**：H5 `$volume`（`daily.csv` `vol`）, `$free_sh`（`daily_basic.csv` `free_share`）
- **参考**：new_factor.md #33

---

### 24. PriceToSales

- **类型**：估值因子 (Value Factor)
- **描述**：市销率（P/S 或 PS_TTM）。适用于评估成长型公司（尤其净利润为负时），弥补 PE 在亏损公司上的缺失。
- **公式**：

  $$PS\_TTM_t = \$ps\_ttm_t$$

- **数据来源**：H5 `$ps_ttm` → `daily_basic.csv` 的 `ps_ttm` 字段
- **参考**：new_factor.md #37

---

### 25. DividendYield

- **类型**：股息因子 (Dividend Factor)
- **描述**：股息率。衡量股东通过现金分红获得的收益，是价值投资的重要参考指标。
- **公式**：

  $$DY_t = \$dv\_ratio_t$$

- **数据来源**：H5 `$dv_ratio` → `daily_basic.csv` 的 `dv_ratio` 字段
- **注意**：此因子在截面归一化**前**会被行业中性化处理生成 `Sector_Relative_DividendYield`
- **参考**：new_factor.md #38

---

### 26. Delta_OperatingCashFlowYield

- **类型**：现金流因子 (Cash Flow Factor)
- **描述**：经营现金流收益率的季度差分。经营现金流比净利润更难操纵，差分后反映现金流质量的变化趋势。
- **公式**：

  $$\Delta OCFYield_t = \left(\frac{\$ocfps_t}{\$close_t}\right) - \left(\frac{\$ocfps_{t-60}}{\$close_{t-60}}\right)$$

  其中 $\$ocfps_t$ 为第 t 日每股经营现金流（TTM）。

- **数据来源**：H5 `$ocfps`（`fina_indicator.csv` `ocfps`）, `$close`（`daily.csv` `close`）
- **参考**：new_factor.md #39

---

### 27. Delta_AssetTurnover

- **类型**：运营效率因子 (Operating Efficiency Factor)
- **描述**：资产周转率的季度差分。资产周转率 = 营业收入 / 总资产，衡量单位资产创造收入的能力。差分值反映运营效率的变化方向。
- **公式**：

  $$\Delta AT_t = \left(\frac{\$revenue_t}{\$total\_assets_t}\right) - \left(\frac{\$revenue_{t-60}}{\$total\_assets_{t-60}}\right)$$

  其中：
  - $\$revenue_t$：第 t 日营业收入
  - $\$total\_assets_t$：第 t 日总资产

- **数据来源**：H5 `$revenue`（`income.csv` `total_revenue`）, `$total_assets`（`balancesheet.csv` `total_assets`）
- **参考**：new_factor.md #40

---

### 28. Delta_DebtToEquity

- **类型**：财务杠杆因子 (Financial Leverage Factor)
- **描述**：资产负债率的季度差分。负债率 = 总负债 / 总权益。差分值衡量公司杠杆水平的变化方向，负债率上升（正值）通常为风险信号。
- **公式**：

  $$\Delta D/E_t = \left(\frac{\$total\_liab_t}{\$total\_equity_t}\right) - \left(\frac{\$total\_liab_{t-60}}{\$total\_equity_{t-60}}\right)$$

  其中：
  - $\$total\_liab_t$：第 t 日总负债
  - $\$total\_equity_t$：第 t 日总权益

- **数据来源**：H5 `$total_liab`（`balancesheet.csv` `total_liab`）, `$total_equity`（`balancesheet.csv` `total_hldr_eqy_exc_min_int`）
- **参考**：new_factor.md #41

---

### 29. SUE 🆕

- **类型**：标准化意外盈余 (Standardized Unexpected Earnings, SUE)
- **描述**：EPS 同比增速除以其近期波动率。SUE 是横截面因子中 Rank IC 最高的因子之一，在 A 股市场尤其有效。当公司公布的 EPS 增长远超其近期波动水平时，市场会持续修正预期。
- **公式**：

  $$SUE_t = \frac{EPS\_YoY_t}{\sigma_{20}(EPS\_YoY)}$$

  其中：
  - $EPS\_YoY_t = \$eps\_yoy_t$：第 t 日基本每股收益同比增长率（%）
  - $\sigma_{20}(EPS\_YoY) = \sqrt{\dfrac{1}{19}\sum_{i=0}^{19}(EPS\_YoY_{t-i} - \overline{EPS\_YoY}_{20})^2}$：20 日滚动标准差
  - 当分母为 0 或缺失时，$SUE_t = 0$（无意外）

- **数据来源**：H5 `$eps_yoy` → `fina_indicator.csv` 的 `basic_eps_yoy` 字段（季度每股收益同比增速）
- **参考**：new_factor.md #14（eps_yoy）
- **注意**：1%/99% 缩尾处理

---

### 30. AssetGrowth 🆕

- **类型**：投资因子 (Investment Factor) / 总资产增长率
- **描述**：总资产同比增长率。Fama-French 五因子模型中的投资因子（CMA）的核心变量。高资产增长率（激进扩张）通常预示未来低收益（A 股负 IC）。
- **公式**：

  $$\text{AssetGrowth}_t = \$assets\_yoy_t$$

  其中 $\$assets\_yoy_t$ 为总资产同比增长率（小数形式，如 0.10 表示增长 10%）。

- **数据来源**：H5 `$assets_yoy` → `fina_indicator.csv` 的 `assets_yoy` 字段（总资产同比增速）

---

### 31. AccrualsRatio 🆕

- **类型**：应计比率 (Accruals Ratio) / 盈余质量因子
- **描述**：应计利润占总资产比例 = (净利润 − 经营现金流) / 总资产。Sloan (1996) 提出的应计异象：高应计公司（利润现金含量低）未来收益较低。应计部分比现金流部分更容易被操纵。
- **公式**：

  $$\text{Accruals}_t = \frac{\$n\_income_t - \$ocf_t}{\$total\_assets_t}$$

  其中：
  - $\$n\_income_t$：第 t 日净利润
  - $\$ocf_t$：第 t 日经营活动现金流净额
  - $\$total\_assets_t$：第 t 日总资产

- **数据来源**：
  - `$n_income` → `income.csv` 的 `n_income` 字段
  - `$ocf` → `cashflow.csv` 的 `n_cashflow_act` 字段
  - `$total_assets` → `balancesheet.csv` 的 `total_assets` 字段
- **注意**：1%/99% 缩尾处理

---

### 32. DebtToEquity 🆕

- **类型**：财务杠杆因子 (Financial Leverage Factor, 水平值版本)
- **描述**：资产负债率水平值 = 总负债 / 总权益。与 Delta_DebtToEquity（差分版本）互补使用：水平值衡量当前的杠杆负担，差分值衡量变化方向。两者在 LightGBM 中可被同时使用。
- **公式**：

  $$D/E_t = \frac{\$total\_liab_t}{\$total\_equity_t}$$

- **数据来源**：H5 `$total_liab`（`balancesheet.csv` `total_liab`）, `$total_equity`（`balancesheet.csv` `total_hldr_eqy_exc_min_int`）

---

### 33. MAX_20d 🆕

- **类型**：最大日收益因子 (MAX Effect, Bali et al. 2011)
- **描述**：过去 20 个交易日的最大日收益率。行为金融学解释为彩票偏好（lottery preference）——投资者追捧有过极端正收益的股票，导致其后续表现不佳。A 股市场 MAX 效应显著为负。
- **公式**：

  $$MAX_t^{20} = \max(r_{t-19}, r_{t-18}, \dots, r_t)$$

  其中 $r_t = \dfrac{close_t}{close_{t-1}} - 1$。

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：Bali, Cakici & Whitelaw (2011), "Maxing out: Stocks as lotteries and the cross-section of expected returns"

---

### 34. RevenueGrowth 🆕

- **类型**：成长因子 (Growth Factor)
- **描述**：营业收入同比增长率。衡量公司业务扩张速度，高营收增长通常被市场正面定价，但需结合利润率综合判断。
- **公式**：

  $$\text{RevenueGrowth}_t = \$revenue\_yoy_t$$

  其中 $\$revenue\_yoy_t$ 为营业收入同比增长率（小数形式，如 0.15 表示增长 15%）。

- **数据来源**：H5 `$revenue_yoy` → `fina_indicator.csv` 的 `or_yoy` 字段（营业收入同比增速）

---

### 35. Amihud_20d 🆕

- **类型**：非流动性因子 (Illiquidity Factor)
- **描述**：Amihud (2002) 非流动性度量，定义为过去 20 个交易日单位成交金额引起的价格变动（绝对值）的均值。非流动性越高，股票的流动性溢价越大（预期收益越高）。A 股市场因零售投资者主导，此因子 Rank IC 极高。
- **公式**：

  $$Amihud_t = \frac{1}{20} \sum_{i=0}^{19} \frac{|r_{t-i}|}{\$amount_{t-i}} \times 10^8$$

  其中：
  - $r_t = \dfrac{close_t}{close_{t-1}} - 1$：第 t 日简单收益率
  - $\$amount_t$：第 t 日成交金额（元）
  - $\times 10^8$：标量缩放因子，使值域落入合理范围

- **数据来源**：
  - `$amount` → `daily.csv` 的 `amount` 字段（日成交金额）
  - `$close` → `daily.csv` 的 `close` 字段
- **参考**：Amihud, Y. (2002), "Illiquidity and stock returns: cross-section and time-series effects"
- **注意**：1%/99% 缩尾处理

---

### 36. FCF_Yield 🆕

- **类型**：自由现金流因子 (Free Cash Flow Factor)
- **描述**：自由现金流收益率 = 自由现金流 / 流通市值。FCF = 经营现金流 − 资本支出，代表真正可分配给股东的现金。高 FCF Yield → 股票被低估。比基于净利润的 Earnings Yield 更难被会计操纵。
- **公式**：

  $$FCFYield_t = \frac{\$fcf_t}{\$circ\_mv_t}$$

  其中：
  - $\$fcf_t$：第 t 日自由现金流（净利润 + 折旧 − 资本支出 − 营运资本变动）
  - $\$circ\_mv_t$：第 t 日流通市值

- **数据来源**：
  - `$fcf` → `cashflow.csv` 的 `free_cashflow` 字段
  - `$circ_mv` → `daily_basic.csv` 的 `circ_mv` 字段
- **注意**：1%/99% 缩尾处理；FCF 可为负值（资本支出超过经营现金流时）

---

### 37. ROA 🆕

- **类型**：盈利能力因子 (Profitability Factor)
- **描述**：年化总资产收益率（ROA）。衡量单位资产创造利润的能力。与 ROE 互补——ROE 受杠杆影响（高杠杆可以抬高 ROE），ROA 不受杠杆影响，是更纯粹的盈利能力度量。
- **公式**：

  $$ROA_t = \$roa\_yearly_t / 100$$

  其中 $\$roa\_yearly_t$ 为年化 ROA（百分比，如 5.2 表示 5.2%），除以 100 转换为小数形式。

- **数据来源**：H5 `$roa_yearly` → `fina_indicator.csv` 的 `roa_yearly` 字段

---

### 38. Skewness_20d 🆕

- **类型**：收益分布因子 (Return Distribution Factor)
- **描述**：过去 20 个交易日收益率的偏度。负偏度表示收益率分布左偏（有极端负收益），蕴含崩盘风险溢价（更高预期回报）。与 MAX_20d 互补——MAX 捕获正极端收益的彩票效应，Skewness 捕获分布非对称性。
- **公式**：

  $$Skew_t = \frac{\frac{1}{20}\sum_{i=0}^{19}(r_{t-i} - \bar{r}_{20})^3}{\left(\frac{1}{19}\sum_{i=0}^{19}(r_{t-i} - \bar{r}_{20})^2\right)^{3/2}}$$

  其中：
  - $r_t = \dfrac{close_t}{close_{t-1}} - 1$
  - $\bar{r}_{20} = \dfrac{1}{20}\sum_{i=0}^{19} r_{t-i}$

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：Conrad, Dittmar & Ghysels (2013), "Ex Ante Skewness and Expected Stock Returns"
- **注意**：1%/99% 缩尾处理；最少需要 10 个有效收益率

---

### 39. NetProfitGrowth 🆕

- **类型**：成长因子 (Growth Factor)
- **描述**：净利润同比增长率。与 RevenueGrowth 互补——营收增长可能伴随利润率下降（增收不增利），净利润增长才是"真增长"。A 股市场对利润增速的定价比营收增速更强。
- **公式**：

  $$NPGrowth_t = \$netprofit\_yoy_t / 100$$

  其中 $\$netprofit\_yoy_t$ 为净利润同比增长率（百分比，如 15.3 表示增长 15.3%）。

- **数据来源**：H5 `$netprofit_yoy` → `fina_indicator.csv` 的 `netprofit_yoy` 字段

---

### 40. EPS_Quality 🆕

- **类型**：盈余质量因子 (Earnings Quality Factor)
- **描述**：经常性损益占 EPS 的比例。非经常性损益（如资产出售、政府补贴）占比越高，EPS 质量越差。Xie (2001) 发现市场高估非经常性收益的持续性，导致高非经常性占比的股票后续表现不佳。
- **公式**：

  $$EPSQ_t = 1 - \frac{|\$npta_t|}{\max(|\$eps_t|, 0.01)}$$

  其中：
  - $\$npta_t$：第 t 日非经常性损益（可正可负）
  - $\$eps_t$：第 t 日基本每股收益
  - $\max(|\$eps_t|, 0.01)$：防止分母为 0（若 |eps| < 0.01 则用 0.01）
  - 值域：$[0, 1]$，1 表示全部来自经常性损益（最高质量）

- **数据来源**：
  - `$npta` → `fina_indicator.csv` 的 `npta` 字段
  - `$eps` → `fina_indicator.csv` 的 `eps` 字段
- **参考**：Xie, H. (2001), "The Mispricing of Abnormal Accruals"
- **注意**：1%/99% 缩尾处理

---

### 41. TurnoverVol_20d 🆕

- **类型**：流动性动态因子 (Liquidity Dynamics Factor)
- **描述**：20 日换手率变异系数 = 换手率标准差 / 换手率均值。Chordia, Subrahmanyam & Anshuman (2001) 发现换手率的波动率（而非换手率水平）是独立的负向预测信号——交易活跃度越不稳定，信息不对称程度越高，预期收益越低。
- **公式**：

  $$TOVol_t = \frac{\sigma_{20}(turnover)}{\mu_{20}(turnover)}$$

  其中：
  - $\sigma_{20}(turnover)$：20 日换手率标准差
  - $\mu_{20}(turnover)$：20 日换手率均值

- **数据来源**：H5 `$turnover` → `daily_basic.csv` 的 `turnover_rate` 字段
- **参考**：Chordia, Subrahmanyam & Anshuman (2001), "Trading activity and expected stock returns"

---

### 42. Reversal_3d 🆕

- **类型**：反转因子 (Reversal Factor)
- **描述**：3 日反转 = 过去 3 日收益率的负值。填补 1d/2d 反转（已被 reversal_1d/reversal_2d 覆盖）与 5d 窗口之间的空白。A 股因 T+1 制度和散户追涨杀跌特征，3-5 日反转效应强于 1-2 日。
- **公式**：

  $$Rev3_t = -\left(\frac{close_t}{close_{t-3}} - 1\right)$$

  其中 $close_t$ 为第 t 日收盘价。

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段

---

## Alpha158 完整因子列表（供参考）

Alpha158 = **9 KBar + 4 Price + 145 Rolling = 158 因子**，全部基于价量数据。

| 类别 | 因子 | 数量 |
|------|------|------|
| **KBar** | KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2 | 9 |
| **Price** | OPEN0, HIGH0, LOW0, VWAP0 | 4 |
| **Rolling @ [5,10,20,30,60]** | ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD | 145 (=29×5) |

## 使用建议

1. **纯实践因子模型**：以上 42 个因子（`--new-factor-only` 模式）。适用于快速验证新因子集的表现。
2. **与 Alpha158 联合**：Alpha158 (158) + 以上 42 个因子 = 200 个因子。Alpha158 通过 `Alpha158` handler 加载，独立因子通过 `AlphaExtra` handler 的 `direct: true` 模式加载。
3. **v3.0 相对 v2.0 的优势**：新增 FCF 覆盖、Amihud 非流动性、收益率偏度、ROA 水平值、净利润增长、EPS 质量、换手率离散度、3 日反转。因子集从 34 增到 42，填补了所有主要覆盖缺口。
