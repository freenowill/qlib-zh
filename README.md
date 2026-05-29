# qlib-zh

基于 **Qlib**（微软 AI 量化框架）与 **RDAgent**（自动因子挖掘）的中国 A 股量化投资平台。集成 TuShare 数据管道、多市场 walk-forward LightGBM 训练、以及 Claude Code 驱动的自动化技能。

> **下游应用**: 因子模型选出的 Top-K 股票，可接入 [stock-fish](https://github.com/freenowill/stock-fish) 进行舆情分析与股价推演，形成「因子选股 → 舆情验证 → 股价推演」的完整决策链。

## 快速开始

```bash
# 拉取 Docker 镜像
docker pull zhuhai123/qlib-rdagent:v1

# 下载 qlib 数据到 ~/.qlib/qlib_data/
# https://github.com/chenditc/investment_data/releases

# 初始化数据管道（TuShare 原始数据 → qlib 格式）
python tushare/get_tushare_data.py
python tushare/check_health.py
```

## 主要工作流

| 命令 | 用途 | 市场 | 因子 |
|------|------|------|------|
| `bash run_new_factor_practice <exp>` | AlphaExtra walk-forward 训练+回测 | 可切换 CSI300/CSI1000/all | Alpha158 + 自定义因子（H5 预计算） |
| `bash run_alpha158_practice <exp>` | Alpha158 6-stage 流水线 | CSI300 | Alpha158 (158) |
| `bash run_alpha158_small <exp>` | 小盘风格实盘选股 | CSI1000 | Alpha158 (158) |
| `/train_csi300` (Claude Code) | CSI300 训练+预测 | CSI300 | Alpha158(158) + practice_factor(42) = 200 |
| `/train_csi1000` (Claude Code) | CSI1000 训练+预测 | CSI1000 | Alpha158(158) + 7/42 有效因子 = 165/200 |
| `/factor-mining` (Claude Code) | 自动因子挖掘 | 全市场 | rdagent fin_factor + DeepSeek |

### 运行示例

```bash
# AlphaExtra: 全流程（H5 → 特征构建 → 数据健康 → walk-forward）
bash run_new_factor_practice my_exp

# AlphaExtra: 仅预测（不训练）
bash run_new_factor_practice my_exp predict_only=True pred_date=2026-05-30

# AlphaExtra: CSI1000 + 指定因子文件
PRACTICE_FACTOR_FILE=tushare/csi_1000.md TARGET_MARKET=csi1000 \
  bash run_new_factor_practice csi1000_exp

# Alpha158 6-stage: 指定历史窗口
WALK_FORWARD_HISTORY_YEARS=10 WALK_FORWARD_START_DATE=2020-01-01 \
  bash run_alpha158_practice my_exp stage=1 end_stage=6
```

## 项目结构

| 目录 | 说明 |
|------|------|
| `qlib/` | Qlib 核心库（上游 fork，有少量 patch） |
| `scripts/` | 自定义脚本：数据采集、stage runner、因子构建 |
| `tushare/` | TuShare 数据管道：API 拉取、健康检查、因子注册 |
| `rdagent_workspace/` | RDAgent 运行时数据（HDF5、模板） |
| `.claude/` | Claude Code 技能与权限配置 |
| `DATA/` | 实验输出（IC 分析、模型、回测报告） |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API key（factor-mining 使用） | 必填 |
| `DOCKER_IMAGE` | Docker 镜像 | `zhuhai123/qlib-rdagent:v1` |
| `TARGET_MARKET` | 股票池 | `all` |
| `TARGET_BENCHMARK` | 基准指数 | `SH000300` |
| `PRACTICE_FACTOR_FILE` | 自定义因子文件 | `tushare/practice_factor.md` |
| `HOLD_NUM` | 持仓数 | `5` |
| `CASH_TOTAL` | 资金 | `30000` |
| `WALK_FORWARD_HISTORY_YEARS` | 历史年数 | `9` |
| `H5_FILE` | HDF5 数据源 | `daily_pv_all.h5` |

## 数据说明

- **qlib 官方数据**（`~/.qlib/qlib_data/cn_data`）仅支持 Alpha158 等标准价量因子
- **AlphaExtra 及自定义因子**（`run_new_factor_practice`、`/train_csi300`、`/train_csi1000`）需要 TuShare 额外数据支持，通过 `tushare/` 管道生成 H5 特征文件

## 安全提醒

- **所有提交到 GitHub 的代码不得含有 API key、token 等私密信息**
- API key 通过环境变量传入（如 `DEEPSEEK_API_KEY`），容器内通过 `.env` 文件
- 仓库中的 `.env` 仅含占位符，不包含真实密钥

## 详细文档

完整的架构说明、数据管道细节、RDAgent 内部机制和已知问题，见 [CLAUDE.md](CLAUDE.md)。
