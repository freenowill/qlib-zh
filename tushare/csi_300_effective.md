# CSI300 长期稳定有效因子

> 数据源：`rdagent_workspace/factor_data_template/daily_pv_csi300.h5`（58 个原始字段）
> 标签：5 日 forward 收益，CSZScoreNorm 横截面归一化
> 模型：LightGBM, CSI300, walk-forward 5 folds (2021→2026), 共 1,210 个交易日
> 分析方法：跨折叠 IC 聚合（5 folds 均值），筛选标准：方向一致 + avg|IR| ≥ 0.10
> 分析日期：2026-05-26

## 筛选标准

从 `practice_factor.md` 的 42 个因子中，基于 CSI300 walk-forward 测试结果筛选：

| 标准 | 说明 |
|------|------|
| 方向一致性 | Valid/Test 集的 RankIC 符号相同 |
| 稳定性 | avg\|IR\| (IC 均值/标准差) ≥ 0.10 |
| 有效性 | 通过以上两条的因子共 **19 个**（42 个中的 45%）|

---

## 因子总览

| # | 因子名称 | 类型 | avg\|IC\| | avg\|IR\| | Valid Pos% | Test Pos% |
|---|----------|------|-----------|-----------|-----------|----------|
| 1 | Amihud_20d | 非流动性 | 2.88% | **0.226** | 59.2% | 57.5% |
| 2 | avg_normalized_range_5d | 波动率 | **3.38%** | **0.162** | 42.9% | 43.6% |
| 3 | PB_Ratio | 估值 | **3.57%** | 0.152 | 44.9% | 43.3% |
| 4 | DividendYield | 股息 | 3.20% | 0.138 | 54.7% | 55.1% |
| 5 | earnings_yield | 估值 | 3.08% | 0.128 | 53.6% | 54.9% |
| 6 | MAX_20d | 行为金融 | 2.70% | 0.146 | 42.8% | 43.5% |
| 7 | Turnover | 流动性 | 2.82% | 0.141 | 43.8% | 44.6% |
| 8 | RealizedVolatility_20d | 波动率 | 2.86% | 0.131 | 43.0% | 43.6% |
| 9 | trailing_PE_ratio | 估值 | 2.83% | 0.116 | 46.8% | 45.3% |
| 10 | Sector_Relative_DividendYield | 估值(截面) | 1.83% | **0.189** | 57.4% | 59.1% |
| 11 | AccrualsRatio | 盈余质量 | 1.67% | **0.164** | 45.1% | 44.8% |
| 12 | SUE | 盈余质量 | 1.38% | **0.155** | 43.3% | 45.2% |
| 13 | Size | 市值/规模 | 1.93% | 0.130 | 45.9% | 45.1% |
| 14 | AssetGrowth | 投资/增长 | 1.44% | 0.136 | 43.6% | 46.0% |
| 15 | EPS_Quality | 盈余质量 | 1.70% | 0.104 | 53.1% | 52.1% |
| 16 | RevenueGrowth | 成长 | 1.32% | 0.109 | 43.7% | 45.9% |
| 17 | Sector_Relative_PB | 估值(截面) | 1.15% | 0.124 | 46.0% | 45.2% |
| 18 | Delta_OperatingCashFlowYield | 现金流 | 1.33% | 0.101 | 53.7% | 53.4% |
| 19 | Sector_Relative_PE | 估值(截面) | 1.04% | 0.116 | 45.8% | 44.1% |

---

## 因子详情

### 1. Amihud_20d

- **类型**：非流动性因子 (Illiquidity Factor)
- **IC 方向**：正值（高非流动性 → 高预期收益）
- **稳定性**：★★★★★（avg\|IR\| = 0.226，42 个因子中最高）
- **描述**：Amihud (2002) 非流动性度量，定义为过去 20 个交易日单位成交金额引起的价格变动（绝对值）的均值。高非流动性对应流动性溢价，在 CSI300 中效果极其稳定。
- **公式**：

  $$Amihud_t = \frac{1}{20} \sum_{i=0}^{19} \frac{|r_{t-i}|}{\$amount_{t-i}} \times 10^8$$

  其中：
  - $r_t = \dfrac{close_t}{close_{t-1}} - 1$：第 t 日简单收益率
  - $\$amount_t$：第 t 日成交金额（元）
  - $\times 10^8$：标量缩放因子

- **数据来源**：H5 `$amount`（`daily.csv` `amount`）, `$close`（`daily.csv` `close`）
- **参考**：practice_factor.md #35, Amihud (2002)

---

### 2. avg_normalized_range_5d

- **类型**：波动率因子 (Volatility Factor)
- **IC 方向**：负值（低波动 → 高预期收益，低波动异象）
- **稳定性**：★★★★★（avg\|IR\| = 0.162）
- **描述**：5 日平均归一化振幅 = (high − low) / close 的 5 日均值。使用日内高低价差度量波动率，低波动股票在 CSI300 中持续跑赢。
- **公式**：

  $$ANR_t^{5} = \frac{1}{5}\sum_{i=0}^{4} \frac{high_{t-i} - low_{t-i}}{close_{t-i}}$$

- **数据来源**：H5 `$high`, `$low`, `$close` → `daily.csv` 的 `high`, `low`, `close` 字段
- **参考**：practice_factor.md #12

---

### 3. PB_Ratio

- **类型**：估值因子 (Value Factor)
- **IC 方向**：负值（低 PB → 高预期收益，价值效应）
- **稳定性**：★★★★★（avg\|IC\| = 3.57%，42 个因子中绝对值最高）
- **描述**：市净率 = 收盘价 / 每股净资产。CSI300 中最强的估值因子，低 PB 股票系统性跑赢。
- **公式**：

  $$PB_t = \$pb_t$$

- **数据来源**：H5 `$pb` → `daily_basic.csv` 的 `pb` 字段
- **注意**：此因子也会被行业中性化处理生成 `Sector_Relative_PB`
- **参考**：practice_factor.md #16

---

### 4. DividendYield

- **类型**：股息因子 (Dividend Factor)
- **IC 方向**：正值（高股息 → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.138）
- **描述**：股息率，衡量股东通过现金分红获得的收益。在 CSI300（大市值蓝筹为主）中高股息策略效果显著。
- **公式**：

  $$DY_t = \$dv\_ratio_t$$

- **数据来源**：H5 `$dv_ratio` → `daily_basic.csv` 的 `dv_ratio` 字段
- **注意**：此因子也会被行业中性化处理生成 `Sector_Relative_DividendYield`
- **参考**：practice_factor.md #25

---

### 5. earnings_yield

- **类型**：估值因子 (Value Factor)
- **IC 方向**：正值（高 E/P → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.128）
- **描述**：盈利收益率（E/P 比率），PE 的倒数。与 PB 互补验证价值效应，高盈利收益率在 CSI300 中稳定有效。
- **公式**：

  $$EY_t = \frac{\$eps_t}{\$close_t}$$

- **数据来源**：H5 `$eps`（`fina_indicator.csv` `eps`）, `$close`（`daily.csv` `close`）
- **参考**：practice_factor.md #9

---

### 6. MAX_20d

- **类型**：行为金融因子 (MAX Effect, Bali et al. 2011)
- **IC 方向**：负值（低最大日收益 → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.146）
- **描述**：过去 20 个交易日的最大日收益率。行为金融学解释为彩票偏好（lottery preference）——投资者追捧有过极端正收益的股票，导致其后续表现不佳。
- **公式**：

  $$MAX_t^{20} = \max(r_{t-19}, r_{t-18}, \dots, r_t)$$

  其中 $r_t = \dfrac{close_t}{close_{t-1}} - 1$。

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：practice_factor.md #33

---

### 7. Turnover

- **类型**：流动性因子 (Liquidity Factor)
- **IC 方向**：负值（低换手 → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.141）
- **描述**：日换手率 = 成交量 / 自由流通股本。低换手率在 CSI300 中持续跑赢，高换手常伴随投机交易。
- **公式**：

  $$\text{Turnover}_t = \frac{\$volume_t}{\$free\_sh_t}$$

- **数据来源**：H5 `$volume`（`daily.csv` `vol`）, `$free_sh`（`daily_basic.csv` `free_share`）
- **参考**：practice_factor.md #23

---

### 8. RealizedVolatility_20d

- **类型**：波动率因子 (Volatility Factor)
- **IC 方向**：负值（低波动 → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.131）
- **描述**：20 日年化已实现波动率，使用简单收益率的标准差乘以年化因子 √252。低波动异象在 CSI300 中非常稳定。
- **公式**：

  $$\sigma_t^{20} = \sqrt{\frac{1}{19} \sum_{i=0}^{19} (r_{t-i} - \bar{r}_{20})^2} \times \sqrt{252}$$

- **数据来源**：H5 `$close` → `daily.csv` 的 `close` 字段
- **参考**：practice_factor.md #1

---

### 9. trailing_PE_ratio

- **类型**：估值因子 (Value Factor)
- **IC 方向**：负值（低 PE → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.116）
- **描述**：滚动市盈率（PE TTM），经典价值指标。低 PE 股票系统性跑赢高 PE 股票。
- **公式**：

  $$PE\_TTM_t = \$pe\_ttm_t$$

- **数据来源**：H5 `$pe_ttm` → `daily_basic.csv` 的 `pe_ttm` 字段
- **注意**：此因子也会被行业中性化处理生成 `Sector_Relative_PE`
- **参考**：practice_factor.md #3

---

### 10. Sector_Relative_DividendYield

- **类型**：估值因子（行业中性化截面因子）
- **IC 方向**：正值（行业内高股息 → 高预期收益）
- **稳定性**：★★★★★（avg\|IR\| = 0.189，行业中性化后 IR 大幅提升）
- **描述**：个股股息率减去同一申万一级行业的股息率截面中位数。行业中性化后的股息率因子 IC 稳定性显著优于原始因子（IR: 0.189 vs 0.138）。
- **公式**：

  $$\text{SectorRelDY}_{t, s} = \$dv\_ratio_{t, s} - \text{median}_{p \in \text{industry}(s)}(\$dv\_ratio_{t, p})$$

- **数据来源**：H5 `$dv_ratio` + `tushare/cn_data/sw_industry.csv`
- **参考**：practice_factor.md #20

---

### 11. AccrualsRatio

- **类型**：盈余质量因子 (Accruals Ratio, Sloan 1996)
- **IC 方向**：负值（低应计 → 高预期收益）
- **稳定性**：★★★★★（avg\|IR\| = 0.164）
- **描述**：应计利润占总资产比例 = (净利润 − 经营现金流) / 总资产。高应计公司（利润现金含量低）未来收益较低，应计部分比现金流部分更容易被操纵。
- **公式**：

  $$\text{Accruals}_t = \frac{\$n\_income_t - \$ocf_t}{\$total\_assets_t}$$

- **数据来源**：
  - `$n_income` → `income.csv` 的 `n_income` 字段
  - `$ocf` → `cashflow.csv` 的 `n_cashflow_act` 字段
  - `$total_assets` → `balancesheet.csv` 的 `total_assets` 字段
- **参考**：practice_factor.md #31

---

### 12. SUE

- **类型**：标准化意外盈余 (Standardized Unexpected Earnings)
- **IC 方向**：负值（低 SUE → 高预期收益）⚠️
- **稳定性**：★★★★★（avg\|IR\| = 0.155）
- **描述**：EPS 同比增速除以其近期波动率。在 CSI300 中方向为负——利好可能已被提前 Price In，反转效应占主导。尽管方向与文献预期相反，但 IR 极高，信号稳定。
- **公式**：

  $$SUE_t = \frac{EPS\_YoY_t}{\sigma_{20}(EPS\_YoY)}$$

- **数据来源**：H5 `$eps_yoy` → `fina_indicator.csv` 的 `basic_eps_yoy` 字段
- **注意**：方向为负（与 A 股全市场文献相反），建议与 EPS_Quality 联合使用
- **参考**：practice_factor.md #29

---

### 13. Size

- **类型**：市值/规模因子 (Size Factor)
- **IC 方向**：负值（小市值 → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.130）
- **描述**：总市值的自然对数。即使在 CSI300（大盘股）内部，小市值效应仍然有效。
- **公式**：

  $$\text{Size}_t = \ln(\$total\_mv_t)$$

- **数据来源**：H5 `$total_mv` → `daily_basic.csv` 的 `total_mv` 字段
- **参考**：practice_factor.md #22

---

### 14. AssetGrowth

- **类型**：投资因子 (Investment Factor) / 总资产增长率
- **IC 方向**：负值（低资产增长 → 高预期收益，投资效应）
- **稳定性**：★★★★（avg\|IR\| = 0.136）
- **描述**：总资产同比增长率。Fama-French 五因子模型中的投资因子（CMA）核心变量。激进扩张的公司未来收益较低。
- **公式**：

  $$\text{AssetGrowth}_t = \$assets\_yoy_t$$

- **数据来源**：H5 `$assets_yoy` → `fina_indicator.csv` 的 `assets_yoy` 字段
- **参考**：practice_factor.md #30

---

### 15. EPS_Quality

- **类型**：盈余质量因子 (Earnings Quality Factor)
- **IC 方向**：正值（高盈余质量 → 高预期收益）
- **稳定性**：★★★（avg\|IR\| = 0.104）
- **描述**：经常性损益占 EPS 的比例。非经常性损益（如资产出售、政府补贴）占比越高，EPS 质量越差。
- **公式**：

  $$EPSQ_t = 1 - \frac{|\$npta_t|}{\max(|\$eps_t|, 0.01)}$$

- **数据来源**：
  - `$npta` → `fina_indicator.csv` 的 `npta` 字段
  - `$eps` → `fina_indicator.csv` 的 `eps` 字段
- **参考**：practice_factor.md #40

---

### 16. RevenueGrowth

- **类型**：成长因子 (Growth Factor)
- **IC 方向**：负值（低营收增长 → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.109）
- **描述**：营业收入同比增长率。在 CSI300 中方向为负——高增长预期已被定价，低增长公司存在预期差修复机会。
- **公式**：

  $$\text{RevenueGrowth}_t = \$revenue\_yoy_t$$

- **数据来源**：H5 `$revenue_yoy` → `fina_indicator.csv` 的 `or_yoy` 字段
- **参考**：practice_factor.md #34

---

### 17. Sector_Relative_PB

- **类型**：估值因子（行业中性化截面因子）
- **IC 方向**：负值（行业内低 PB → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.124）
- **描述**：个股 PB 减去同一申万一级行业的 PB 截面中位数。行业中性化后 PB 的选股能力更纯粹。
- **公式**：

  $$\text{SectorRelPB}_{t, s} = \$pb_{t, s} - \text{median}_{p \in \text{industry}(s)}(\$pb_{t, p})$$

- **数据来源**：H5 `$pb` + `tushare/cn_data/sw_industry.csv`
- **参考**：practice_factor.md #18

---

### 18. Delta_OperatingCashFlowYield

- **类型**：现金流因子 (Cash Flow Factor)
- **IC 方向**：正值（OCF 收益率提升 → 高预期收益）
- **稳定性**：★★★（avg\|IR\| = 0.101）
- **描述**：经营现金流收益率的季度差分。经营现金流比净利润更难操纵，差分后反映现金流质量的变化趋势。
- **公式**：

  $$\Delta OCFYield_t = \left(\frac{\$ocfps_t}{\$close_t}\right) - \left(\frac{\$ocfps_{t-60}}{\$close_{t-60}}\right)$$

- **数据来源**：H5 `$ocfps`（`fina_indicator.csv` `ocfps`）, `$close`（`daily.csv` `close`）
- **参考**：practice_factor.md #26

---

### 19. Sector_Relative_PE

- **类型**：估值因子（行业中性化截面因子）
- **IC 方向**：负值（行业内低 PE → 高预期收益）
- **稳定性**：★★★★（avg\|IR\| = 0.116）
- **描述**：个股 PE_TTM 减去同一申万一级行业的 PE_TTM 截面中位数。行业中性化 PE，逻辑同 Sector_Relative_PB。
- **公式**：

  $$\text{SectorRelPE}_{t, s} = \$pe\_ttm_{t, s} - \text{median}_{p \in \text{industry}(s)}(\$pe\_ttm_{t, p})$$

- **数据来源**：H5 `$pe_ttm` + `tushare/cn_data/sw_industry.csv`
- **参考**：practice_factor.md #19

---

## 因子类别有效性总结

| 类别 | 有效数量 | 平均\|IC\| | 平均\|IR\| | 因子 |
|------|----------|-----------|-----------|------|
| **估值** | 5 | 2.52% | 0.139 | PB_Ratio, earnings_yield, trailing_PE_ratio, Sector_Relative_PB, Sector_Relative_PE |
| **波动率/风险** | 3 | 2.98% | 0.146 | RealizedVolatility_20d, avg_normalized_range_5d, MAX_20d |
| **流动性** | 2 | 2.85% | 0.183 | Amihud_20d, Turnover |
| **股息** | 2 | 2.52% | 0.163 | DividendYield, Sector_Relative_DividendYield |
| **盈余质量** | 3 | 1.58% | 0.141 | AccrualsRatio, SUE, EPS_Quality |
| **增长** | 2 | 1.38% | 0.123 | AssetGrowth, RevenueGrowth |
| **市值** | 1 | 1.93% | 0.130 | Size |
| **现金流** | 1 | 1.33% | 0.101 | Delta_OperatingCashFlowYield |

## 未通过筛选的因子（23 个）

以下 `practice_factor.md` 中的因子在 CSI300 上未达到稳定有效标准（方向不一致或 avg\|IR\| < 0.10）：

| 因子 | 原因 |
|------|------|
| PriceToSales, DebtToEquity, ROA | avg\|IR\| < 0.10（弱有效） |
| volume_weighted_momentum_5d, TurnoverVol_20d | avg\|IR\| < 0.10 |
| Skewness_20d, FCF_Yield | avg\|IR\| < 0.10 |
| turnover_trend, vwap_deviation_10d, vwap_deviation_5d | avg\|IR\| < 0.10 |
| reversal_1d, reversal_2d, Reversal_3d | avg\|IR\| < 0.05（反转在 CSI300 中很弱） |
| volume_change_5d, sharpe_10d, obv_slope_10d | avg\|IR\| < 0.05 |
| Delta_roe, Delta_net_profit_margin, Delta_DebtToEquity, Delta_AssetTurnover | avg\|IR\| < 0.07 |
| **NetProfitGrowth, momentum_vol_adjusted_20, risk_adjusted_momentum_5d_20d** | **方向不一致（Valid/Test 符号反转）** |

## 组合建议

基于 CSI300 有效因子，推荐多因子等权组合配置：

| 成分 | 权重 | 因子 |
|------|------|------|
| 价值 | 30% | PB_Ratio, earnings_yield, trailing_PE_ratio, Sector_Relative_PB, Sector_Relative_PE |
| 低波动 | 25% | RealizedVolatility_20d, avg_normalized_range_5d, MAX_20d |
| 流动性 | 15% | Amihud_20d, Turnover |
| 质量/盈余 | 15% | AccrualsRatio, EPS_Quality, SUE(负向) |
| 股息 | 10% | DividendYield, Sector_Relative_DividendYield |
| 现金流 | 5% | Delta_OperatingCashFlowYield |
