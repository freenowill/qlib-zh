# Alpha158 Practice 量化交易系统

## 目标

- 严格时间因果
- Walk-forward 滚动训练
- 回测与实盘同一信号口径
- 输出 Top-5 周频调仓组合

## Stage 设计

### Stage 1 数据准备

- 输入：`~/.qlib/qlib_data/cn_data`
- 动作：执行 `scripts/check_data_health.py check_data`
- 输出：日志与数据健康状态

### Stage 2 滚动训练

- 模型：`LightGBM + Alpha158`
- 切分：
	- `train = [t-5y, t-2y]`
	- `valid = [t-2y, t]`
	- `test = [t, t+3m]`
- 输出：`model_predict/scores.csv`、`walk_forward_summary.csv`、`report_of_backtest.txt`

### Stage 3 初筛

- 股票池：沪深300
- 过滤：ST、停牌、涨跌停、上市不足60天、成交额后20%、股价高于50
- 输出：`first_screen/first_screen.csv`

### Stage 4 风险评估

- 数据：`baostock + 公开字段`
- 标签：财务 `优/良/中/差`，估值 `高/中/低`，风险 `高/中/低`
- 输出：`risk_eval/risk_eval.csv`

### Stage 5 组合构建

- 过滤：财务差 / 高风险 / 高估值
- 排序：模型分数为主，辅以 `ICIR`、收益、回撤、Sharpe、月胜率
- 输出：`second_screen/second_screen.csv`

### Stage 6 实盘信号

- 输出：
	- `final_result/result.csv`
	- `final_result/result_update.csv`
- 替换规则：排名落入后50%则替换

## 防未来函数原则

1. 训练、验证、测试严格按时间切分
2. `test` 仅用于冻结模型后的预测
3. Stage 3-6 只读取最新 `scores.csv`
4. 调仓信号统一使用下一交易日执行口径
