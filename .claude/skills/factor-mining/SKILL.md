---
name: factor-mining
description: Automated factor mining pipeline using rdagent fin_factor + DeepSeek LLM. Discovers, validates, and records non-duplicate quantitative factors across diverse categories (momentum, value, quality, growth, volatility, liquidity, leverage, cash flow, size, dividend).
---

# Factor Mining Skill

Automated end-to-end factor discovery pipeline. Reads existing factors from `tushare/new_factor.md` (passed) and `tushare/fail_new_factor.md` (failed), runs rd-agent fin_factor with DeepSeek-v4-pro to discover new factors, filters by evaluation quality, appends unique validated factors, and records failures to prevent re-mining.

## Target Factor Categories

| Category | Chinese Label | Alpha158 覆盖 | Examples | Key cn_extra_data fields |
|----------|---------------|---------------|----------|--------------------------|
| Value | 估值 | 无 | PE, PB, PS, earnings yield | `$pe_ttm`, `$pb`, `$ps_ttm`, `$pe` |
| Quality | 质量 | 无 | ROE, ROA, profit margin, npta | `$roe_yearly`, `$roa_yearly`, `$netprofit_margin`, `$npta` |
| Growth | 成长 | 无 | EPS growth, revenue growth, BPS growth | `$eps_yoy`, `$revenue_yoy`, `$bps_yoy`, `$netprofit_yoy`, `$assets_yoy` |
| Momentum/Reversal | 动量/反转 | ROC5-60, 已覆盖基础 | price momentum, reversal | `$close`, `$adjclose`, `$vwap` |
| Low Volatility | 低波动 | STD, 部分覆盖 | realized vol, variance | `$close` (computed) |
| Liquidity | 流动性 | VMA, VSTD, 已覆盖基础 | turnover, volume ratio, amount | `$turnover`, `$turnover_f`, `$volume`, `$vol_ratio`, `$amount` |
| Financial Leverage | 财务杠杆 | 无 | debt/equity, debt/assets | `$debt_to_assets`, `$liab_to_eqty`, `$assets_to_eqt` |
| Cash Flow | 现金流 | 无 | OCF, FCF, CFPS, OCF/profit | `$ocf`, `$fcf`, `$icf`, `$cfps`, `$ocfps`, `$ocf_to_profit`, `$ocf_to_assets` |
| Market Cap/Size | 市值/规模 | 无 | total MV, circ MV, shares | `$total_mv`, `$circ_mv`, `$total_sh`, `$float_sh`, `$free_sh` |
| Dividend | 股息 | 无 | dividend yield, dv TTM | `$dv_ratio`, `$dv_ttm` |
| Operating Efficiency | 运营效率 | 无 | op/revenue, revenue PS | `$op_to_revenue`, `$revenue_ps`, `$operate_profit` |
| Volume-Price | 量价 | CORR, CORD, 部分覆盖 | OBV, VWAP-based, MFI | `$volume`, `$close`, `$vwap`, `$amount` |
| Risk-Adjusted | 风险调整 | 无 | Sharpe, Sortino, Information ratio | `$close` (computed) |

> **Alpha158 覆盖说明**: 动量/反转 (ROC, RSV, RANK)、波动率 (STD, QTLU/D)、流动性 (VMA, VSTD)、量价 (CORR, CORD, WVMA) 等 Alpha158 已有基础覆盖。Factor mining 优先挖掘 **Alpha158 无法覆盖的维度**：估值、质量、成长、财务杠杆、现金流、市值/规模、股息、运营效率、风险调整。

## Prerequisites

Before each run, verify:

1. **Docker disk space**: Run `docker system prune -af` if disk is near full (>20GB free needed)
2. **DeepSeek API key**: Set `DEEPSEEK_API_KEY` environment variable: `export DEEPSEEK_API_KEY="sk-..."`. The skill reads this variable; never hardcode keys.
3. **DeepSeek proxy status**: The proxy must be running on `0.0.0.0:18080`. Check with `lsof -i :18080`. If not running, start it (see Step 2).
4. **Linux Cython modules**: `qlib/data/_libs/rolling.cpython-310-x86_64-linux-gnu.so` must exist. If missing, compile (see Step 3).
5. **HDF5 source data**: `rdagent_workspace/factor_data_template/daily_pv_all.h5` and `daily_pv_debug.h5` must exist. Generated directly from `tushare/extra_data/` CSV files (5525 stocks, 58 fields). Run `python generate.py` if missing.

## Step-by-Step Workflow

### Step 1: Read existing factors (both passed AND failed) and determine target categories

Read **both** `tushare/new_factor.md` (passed factors) and `tushare/fail_new_factor.md` (failed factors) to build a comprehensive dedup set. The goal is to avoid wasting LLM resources on factors that have already been tried — whether they passed or failed.

```bash
# Extract all passed factor names from new_factor.md (strip `[+Alpha158]` / `[独立]` tags)
grep -E '^### [0-9]+\. ' tushare/new_factor.md | \
  sed -E 's/^### [0-9]+\. //; s/ `?\[\+Alpha158[^]]*\]`?//g; s/ `?\[独立\]`?//g'

# Extract all failed factor names from fail_new_factor.md
grep -E '^## [0-9]+\. ' tushare/fail_new_factor.md 2>/dev/null | sed 's/^## [0-9]*\. //'

# Extract all known factor names from previous run logs
grep -ohP 'factor_name: \K[^\n]+' factors_results_round*.txt 2>/dev/null | sort -u
```

Build a **dedup registry** that covers BOTH passed and failed factors:
- Factor names (case-insensitive, normalize `_`/`-`/space to single separator)
- Factor formulations (normalize whitespace, remove LaTeX formatting, compare semantically)
- Concept fingerprints (e.g., "20-day price change ratio" → momentum family, window=20)
- **IMPORTANT**: A factor that already failed is just as important to avoid as one that already passed. Both waste LLM resources if re-proposed.

**Determine which categories are uncovered.** Cross-reference existing factors (passed + failed) against the target categories table above. Prioritize categories with zero coverage.

### Step 2: Start DeepSeek API proxy (if not running)

```bash
# Check if already running
lsof -i :18080 | grep LISTEN && echo "Proxy already running" || {
  pkill -f deepseek_proxy.py 2>/dev/null; sleep 1

  cat > /tmp/deepseek_proxy.py << 'PYEOF'
"""Forward proxy: 0.0.0.0:18080 -> api.deepseek.com"""
import http.server, json, ssl, urllib.request, sys
PORT=18080; TARGET="https://api.deepseek.com"
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(s):
        l=int(s.headers.get('Content-Length',0)); b=s.rfile.read(l) if l else b''
        u=TARGET+(s.path or"/v1/chat/completions")
        r=urllib.request.Request(u,data=b,method='POST')
        r.add_header('Content-Type','application/json')
        a=s.headers.get('Authorization','')
        if a: r.add_header('Authorization',a)
        try:
            p=urllib.request.urlopen(r,timeout=300,context=ssl.create_default_context())
            s.send_response(p.status)
            for k,v in p.getheaders():
                if k.lower() not in('transfer-encoding','connection'): s.send_header(k,v)
            s.end_headers(); s.wfile.write(p.read())
        except Exception as e:
            b=json.dumps({"error":str(e)}).encode(); s.send_response(502)
            s.send_header('Content-Length',len(b)); s.end_headers(); s.wfile.write(b)
    def log_message(s,f,*a): print(f"[proxy {a[0]}]",file=sys.stderr)
http.server.HTTPServer(('0.0.0.0',PORT),H).serve_forever()
PYEOF

  python3 /tmp/deepseek_proxy.py &
  sleep 2
  lsof -i :18080 | grep LISTEN && echo "Proxy started on :18080" || echo "ERROR: Proxy failed to start"
}
```

### Step 3: Ensure Cython modules exist

```bash
# Check if Linux .so files exist
if [ ! -f qlib/data/_libs/rolling.cpython-310-x86_64-linux-gnu.so ]; then
  echo "Compiling Cython extensions in Docker..."
  docker run --rm \
    -v "$(pwd):/repo" -w /repo \
    zhuhai123/qlib-rdagent:v1 \
    bash -c "pip install cython numpy -q && python -c \"
from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy
ext=[Extension('qlib.data._libs.rolling',['qlib/data/_libs/rolling.pyx'],language='c++',include_dirs=[numpy.get_include()]),
     Extension('qlib.data._libs.expanding',['qlib/data/_libs/expanding.pyx'],language='c++',include_dirs=[numpy.get_include()])]
setup(ext_modules=cythonize(ext,language_level='3'),script_args=['build_ext','--inplace'])
\""
fi
```

### Step 4: Regenerate HDF5 source data

Regenerate HDF5 from the latest `tushare/extra_data/` CSV files (58 fields, no qlib dependency). The `generate.py` reads CSVs directly, aligns to a common calendar, and builds `daily_pv_all.h5` + `daily_pv_debug.h5`.

```bash
# Verify extra_data exists
if [ ! -d tushare/extra_data ]; then
  echo "ERROR: extra_data not found. Run: bash tushare/check_health.py first, or manually create extra_data directory"
  exit 1
fi

cd rdagent_workspace/factor_data_template
python generate.py
cd -
```

### Step 5: Run fin_factor with targeted exploration

The base command. Adjust `--loop-n` and `--step-n` based on how many new categories you need:

```bash
HOST_PWD="$(pwd)"
HOST_IP=$(ifconfig en0 2>/dev/null | grep 'inet ' | awk '{print $2}' | head -1)
TIMESTAMP=$(date +%Y%m%d_%H%M)

docker run --rm \
  --dns 8.8.8.8 --dns 114.114.114.114 \
  -e PYTHONPATH="$HOST_PWD" \
  -e DOCKER_HOST=unix:///var/run/docker.sock \
  -e OPENAI_API_KEY="${DEEPSEEK_API_KEY:?err:请先 export DEEPSEEK_API_KEY="sk-..."}" \
  -e CHAT_MODEL='openai/deepseek-v4-flash' \
  -e OPENAI_API_BASE="http://${HOST_IP}:18080/v1" \
  -e CONDA_DEFAULT_ENV=qlib_env \
  -e RDAGENT_MAX_ROUNDS=15 \
  -e RDAGENT_RETRY_WAIT_SECONDS=30 \
  -v "$HOST_PWD:$HOST_PWD" \
  -v "$HOME/.qlib:/root/.qlib" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w "$HOST_PWD" \
  zhuhai123/qlib-rdagent:v1 \
  rdagent fin_factor --step-n 5 --loop-n 2 \
  2>&1 | tee "factors_results_${TIMESTAMP}.txt"
```

**Existing factor avoidance:** `sitecustomize.py` (loaded via `PYTHONPATH`) automatically injects factors from `tushare/new_factor.md` and `tushare/fail_new_factor.md` into the LLM prompt at runtime, instructing the model not to re-propose them. A hard dedup filter in `convert_response()` also catches any duplicates that slip through. No additional configuration needed.

**Parameter guide:**
| Scenario | `--loop-n` | `--step-n` | Expected time |
|----------|-----------|-----------|---------------|
| Quick test / 1-2 categories | 1 | 5 | ~15 min |
| Normal run / 3-5 categories | 2 | 5-10 | ~30-45 min |
| Full sweep / all remaining | 3 | 10-15 | ~1-1.5 hours |

**Targeting specific categories:** The LLM proposes factors based on available data columns. rdagent's feedback loop naturally diversifies across loops. For best coverage of missing categories, run with higher `--loop-n` (3-5) to give more exploration chances.

### Step 6: Parse results — extract BOTH passing and failing factors

After fin_factor completes, extract all factors and their evaluation results from the output file:

```bash
OUTPUT_FILE="factors_results_YYYYMMDD_HHMM.txt"  # replace with actual

# 1. Get all unique factor names proposed
grep -oP 'factor_name: \K\S+' "$OUTPUT_FILE" | sort -u

# 2. Get evaluation results - find all "Final decisions:" lines
grep -oP 'Final decisions: \[.*?\] True count: \d+' "$OUTPUT_FILE"

# 3. Extract full factor details for ALL factors (both passing and failing)
# Each factor appears as a block with: factor_name, factor_description, factor_formulation, variables
grep -A3 'factor_name: ' "$OUTPUT_FILE" | grep -v '^--$'

# 4. For each factor, determine if it passed (final_decision: True in its LAST evaluation)
# Factors that appear with final_decision: False and never later appear with True → FAILED
# Factors that appear with final_decision: True → PASSED
```

**How to read evaluation results:**
- The output contains blocks like:
  ```
  factor_name: FactorName
  factor_description: [Category] Description
  factor_formulation: LaTeX/plain formula
  variables: {'var1': 'desc1', ...}
  ```
- Each block is followed by execution feedback and a JSON `{"final_decision": true/false, ...}`
- `final_decision: True` means the code ran correctly and output format is valid
- `final_decision: False` means the factor failed (code error, wrong output format, or invalid values)
- `Final decisions: [True, False, True, ...] True count: N` means N out of M factors passed in that batch
- A factor may appear multiple times across loops; its **last** decision is what matters

**Track failures carefully.** Factors with `final_decision: False` must be recorded in `tushare/fail_new_factor.md` so they are not re-proposed in future runs.

### Step 7: Filter and deduplicate

**Quality criteria for accepting a factor (ALL must pass):**

1. **final_decision == True** — code executed without error, output format is correct (MultiIndex [datetime, instrument], single float64 column)
2. **Not a duplicate** of any existing factor in `tushare/new_factor.md` OR `tushare/fail_new_factor.md`

**Dedup rules (apply in order, check against BOTH new_factor.md AND fail_new_factor.md):**
0. **Strip tags before comparing**: Remove `[+Alpha158 XXX]` and `[独立]` tags from factor names in new_factor.md; compare the bare name only
1. Exact name match (case-insensitive, after normalizing `_`/`-`)
2. Same formula with different variable naming → DUPLICATE
3. Same underlying concept + same window → DUPLICATE (e.g., `momentum_20d` = `MediumTermMomentum_20d` = `20d_return`)
4. Same concept + different window → DIFFERENT factor (e.g., `momentum_5d` ≠ `momentum_20d`)
5. Different data source for same concept → DIFFERENT (e.g., PE_ttm based ≠ PB based, even if both are value)
6. If uncertain, compare the actual `factor_formulation` field — normalize whitespace and compare
7. **Alpha158 overlap check**: Reference the "Alpha158 重叠分析" table in new_factor.md. If the proposed factor is conceptually equivalent to an existing Alpha158 factor (e.g., momentum_Nd ↔ ROC_N, RSI_Nd ↔ SUMP_N, intraday range ↔ KLEN, volume_ratio ↔ VMA), mark as DUPLICATE — Alpha158 already provides the same signal.

**How to check for IC/Rank IC:** The rdagent evaluator's `final_decision: True` already validates:
- Code execution success
- Output format correctness (MultiIndex, float64, single column)
- Factor values are finite and reasonable

IC/Rank IC values are computed by the downstream Qlib model training (Stage2 walk-forward), not by fin_factor itself. The `final_decision: True` is the primary quality gate. If individual IC values appear in the output, prefer factors with IC > 0.02 or Rank IC > 0.02, but do NOT reject a passing factor solely for missing IC values — the IC is context-dependent.

### Step 8: Append qualified factors to new_factor.md

For each qualifying factor, append to `tushare/new_factor.md` in this format:

```markdown
### N. factor_name [TAG]

- **类型**：<Chinese category label from the table above>
- **描述**：<One-line Chinese+English description of what the factor measures and how to interpret it>
- **公式**：

  $$<LaTeX formula>$$

- **变量**：
  - $var_1$：<description>
  - $var_2$：<description>
- **数据来源**：cn_extra_data <specific fields used>
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
- **Alpha158**：`[独立]` 或 `[+Alpha158 XXX]` — 根据是否与 Alpha158 已有因子重叠选择
---
```

**Rules for appending to new_factor.md:**
- Increment the section number (`N`) from the last existing factor
- **TAG**: Use `[独立]` if the factor has no Alpha158 equivalent; use `[+Alpha158 XXX]` (e.g., `[+Alpha158 ROC5]`) if it overlaps with an Alpha158 factor — check new_factor.md's "Alpha158 重叠分析" table
- Add the new entry before the `## Alpha158 完整因子列表` section
- Update the summary table at the top of the file: add a row for each new factor with the TAG

### Step 9: Record failed factors in fail_new_factor.md

For each factor that received `final_decision: False` in its last evaluation, append to `tushare/fail_new_factor.md` in this format:

```markdown
## N. factor_name

- **类型**：<Chinese category label>
- **描述**：<One-line description of what the factor measures>
- **公式**：

  $$<LaTeX formula>$$

- **失败原因**：final_decision: False — <code error / output format incorrect / invalid values / etc.>
- **日期**：YYYY-MM-DD
---
```

**Rules for appending to fail_new_factor.md:**
- Increment the section number (`N`) from the last existing failed factor
- Update the summary table at the top: add a row with factor name, type, failure reason, and date
- Only record a failed factor ONCE — if it already appears in `fail_new_factor.md`, do not duplicate; update the existing entry's date and reason if they changed
- A factor that previously failed but now passes should be REMOVED from `fail_new_factor.md` and added to `new_factor.md`
- **Before appending, check**: is this failed factor actually a duplicate (by name/formula/concept) of an existing failed factor? If so, skip or update the existing entry

### Step 10: Cleanup

```bash
# Stop the proxy (or leave it running for the next mining session)
# pkill -f deepseek_proxy.py 2>/dev/null

# Free Docker disk space
docker system prune -af
```

## Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Proxy not running | SSL/timeout errors in Docker | Start proxy: `python3 /tmp/deepseek_proxy.py &` |
| Cython ModuleNotFoundError | `No module named 'qlib.data._libs.rolling'` | Recompile `.so` inside Docker (Step 3) |
| OOM (exit 137) | Docker container killed mid-run | Reduce `--step-n` or `--loop-n`; use smaller batches |
| Disk full | Docker daemon error / exit 1 | `docker system prune -af` |
| Embedding API 404 | DeepSeek has no embedding endpoint | Already stubbed in `sitecustomize.py` |
| Same factors every run | LLM re-proposes similar factors | Increase `--loop-n` to 3-5 for more exploration diversity |
| Same FAILED factors re-proposed | LLM keeps trying already-failed factors | Ensure `fail_new_factor.md` is up to date; add explicit "DO NOT propose" instructions in the prompt |
| All factors rejected | `True count: 0` in final decisions | Check if Cython modules are compiled; verify proxy connectivity |
| Proxy port conflict | `Address already in use` | `lsof -i :18080` and kill existing process |
| HDF5 data missing | `daily_pv_all.h5 not found` | Run `cd rdagent_workspace/factor_data_template && python generate.py` (reads from extra_data CSV, no qlib needed) |

## Progress Tracking

Update this table after each successful run. Mark categories with factors. Failed factors (tracked in `fail_new_factor.md`) also count toward category coverage — a category is "DONE" when it has at least one passing factor.

| Category | Alpha158 基础 | new_factor 独立因子 | Status |
|----------|---------------|---------------------|--------|
| 动量/反转 | ROC, RSV, RANK (25) | reversal_1d/2d, momentum_vol_adjusted_20 等 | DONE |
| 波动率 | STD, QTLU/D, WVMA (25) | RealizedVolatility_20d, avg_normalized_range_5d, Volatility_5d, Volatility_10d | DONE |
| 震荡 | SUMP/SUMN/SUMD (15) | RSI_14d | DONE |
| 流动性 | VMA, VSTD, VSUMP/M/D (25) | 5_day_volume_change, turnover_trend, volume_ratio_5d_20d, Liquidity_Turnover_5d, Turnover | DONE |
| 估值 | 无 | trailing_PE_ratio, PB_Ratio, earnings_yield, book_to_price, Sector_Relative_PB, **PriceToSales** | DONE |
| 量价 | CORR, CORD (10) | obv_slope_10day, volume_weighted_momentum_5d, vwap_deviation_5d/10d | DONE |
| 风险调整 | 无 | sharpe_10day, Momentum_Vol_Adjusted_20, risk_adjusted_momentum_5d_20d | DONE |
| 质量 | 无 | roe, net_profit_margin | DONE |
| 成长 | 无 | — | TODO |
| 财务杠杆 | 无 | **DebtToEquity** | **DONE** |
| 现金流 | 无 | **OperatingCashFlowYield** | **DONE** |
| 市值/规模 | 无 | Size | DONE |
| 股息 | 无 | **DividendYield** | **DONE** |
| 运营效率 | 无 | net_profit_margin, **AssetTurnover** | DONE |

## Related Files

| File | Purpose |
|------|---------|
| `tushare/new_factor.md` | All factors that **passed** evaluation — used in production |
| `tushare/fail_new_factor.md` | All factors that were **proposed but failed** — used for dedup only |
| `factors_results_*.txt` | Raw rdagent output logs from each mining run |
