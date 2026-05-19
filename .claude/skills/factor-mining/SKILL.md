---
name: factor-mining
description: Automated factor mining pipeline using rdagent fin_factor + DeepSeek LLM. Discovers, validates, and records non-duplicate quantitative factors across diverse categories (momentum, value, quality, growth, volatility, liquidity, leverage, cash flow, size, dividend).
---

# Factor Mining Skill

Automated end-to-end factor discovery pipeline. Reads existing factors from `tushare/new_factor.md`, runs rd-agent fin_factor with DeepSeek-v4-pro to discover new factors, filters by IC/Rank IC quality, and appends unique validated factors.

## Target Factor Categories

| Category | Chinese Label | Examples | Key cn_extra_data fields |
|----------|---------------|----------|--------------------------|
| Value | 估值 | PE, PB, PS, earnings yield | `$pe_ttm`, `$pb`, `$ps_ttm`, `$pe` |
| Quality | 质量 | ROE, ROA, profit margin, npta | `$roe_yearly`, `$roa_yearly`, `$netprofit_margin`, `$npta` |
| Growth | 成长 | EPS growth, revenue growth, BPS growth | `$eps_yoy`, `$revenue_yoy`, `$bps_yoy`, `$netprofit_yoy`, `$assets_yoy` |
| Momentum/Reversal | 动量/反转 | price momentum, reversal | `$close`, `$adjclose`, `$vwap` |
| Low Volatility | 低波动 | realized vol, variance | `$close` (computed) |
| Liquidity | 流动性 | turnover, volume ratio, amount | `$turnover`, `$turnover_f`, `$volume`, `$vol_ratio`, `$amount` |
| Financial Leverage | 财务杠杆 | debt/equity, debt/assets | `$debt_to_assets`, `$liab_to_eqty`, `$assets_to_eqt` |
| Cash Flow | 现金流 | OCF, FCF, CFPS, OCF/profit | `$ocf`, `$fcf`, `$icf`, `$cfps`, `$ocfps`, `$ocf_to_profit`, `$ocf_to_assets` |
| Market Cap/Size | 市值/规模 | total MV, circ MV, shares | `$total_mv`, `$circ_mv`, `$total_sh`, `$float_sh`, `$free_sh` |
| Dividend | 股息 | dividend yield, dv TTM | `$dv_ratio`, `$dv_ttm` |
| Operating Efficiency | 运营效率 | op/revenue, revenue PS | `$op_to_revenue`, `$revenue_ps`, `$operate_profit` |
| Volume-Price | 量价 | OBV, VWAP-based, MFI | `$volume`, `$close`, `$vwap`, `$amount` |
| Risk-Adjusted | 风险调整 | Sharpe, Sortino, Information ratio | `$close` (computed) |

## Prerequisites

Before each run, verify:

1. **Docker disk space**: Run `docker system prune -af` if disk is near full (>20GB free needed)
2. **DeepSeek proxy status**: The proxy must be running on `0.0.0.0:18080`. Check with `lsof -i :18080`. If not running, start it (see Step 2).
3. **Linux Cython modules**: `qlib/data/_libs/rolling.cpython-310-x86_64-linux-gnu.so` must exist. If missing, compile (see Step 3).
4. **HDF5 source data**: `rdagent_workspace/factor_data_template/daily_pv_all.h5` and `daily_pv_debug.h5` must exist. Run `python rdagent_workspace/factor_data_template/generate.py` if missing.

## Step-by-Step Workflow

### Step 1: Read existing factors and determine target categories

Read `tushare/new_factor.md` to extract all existing factor names, types, and formulations. Build a dedup set:

```bash
# Extract all factor names from new_factor.md
grep -E '^## [0-9]+\. ' tushare/new_factor.md | sed 's/^## [0-9]*\. //'

# Extract all known factor names from previous run logs
grep -ohP 'factor_name: \K[^\n]+' factors_results_round*.txt 2>/dev/null | sort -u
```

Build a **dedup registry** with:
- Factor names (case-insensitive, normalize `_`/`-`/space to single separator)
- Factor formulations (normalize whitespace, remove LaTeX formatting, compare semantically)
- Concept fingerprints (e.g., "20-day price change ratio" → momentum family, window=20)

**Determine which categories are uncovered.** Cross-reference existing factors against the target categories table above. Prioritize categories with zero coverage.

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

### Step 4: Ensure HDF5 source data exists

```bash
cd rdagent_workspace/factor_data_template
if [ ! -f daily_pv_all.h5 ] || [ ! -f daily_pv_debug.h5 ]; then
  python generate.py
fi
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
  -e OPENAI_API_KEY='sk-c697147e8ba0406c8ae76839f21d2048' \
  -e CHAT_MODEL='openai/deepseek-v4-pro' \
  -e OPENAI_API_BASE="http://${HOST_IP}:18080/v1" \
  -e CONDA_DEFAULT_ENV=qlib_env \
  -e RDAGENT_MAX_ROUNDS=15 \
  -e RDAGENT_RETRY_WAIT_SECONDS=30 \
  -v "$HOST_PWD:$HOST_PWD" \
  -v "$HOME/.qlib:/root/.qlib" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w "$HOST_PWD" \
  zhuhai123/qlib-rdagent:v1 \
  rdagent fin_factor --step-n 20 --loop-n 3 \
  2>&1 | tee "factors_results_${TIMESTAMP}.txt"
```

**Parameter guide:**
| Scenario | `--loop-n` | `--step-n` | Expected time |
|----------|-----------|-----------|---------------|
| Quick test / 1-2 categories | 1 | 10 | ~30 min |
| Normal run / 3-5 categories | 2-3 | 15-20 | ~60-90 min |
| Full sweep / all remaining | 3-5 | 20-30 | ~2-3 hours |

**Targeting specific categories:** The LLM proposes factors based on available data columns. rdagent's feedback loop naturally diversifies across loops. For best coverage of missing categories, run with higher `--loop-n` (3-5) to give more exploration chances.

### Step 6: Parse results and extract passing factors

After fin_factor completes, extract all passing factors from the output file:

```bash
OUTPUT_FILE="factors_results_YYYYMMDD_HHMM.txt"  # replace with actual

# 1. Get all unique factor names proposed
grep -oP 'factor_name: \K\S+' "$OUTPUT_FILE" | sort -u

# 2. Get evaluation results - find all "Final decisions:" lines
grep -oP 'Final decisions: \[.*?\] True count: \d+' "$OUTPUT_FILE"

# 3. Extract full factor details for each passing factor
# Each factor appears as a block with: factor_name, factor_description, factor_formulation, variables
grep -A3 'factor_name: ' "$OUTPUT_FILE" | grep -v '^--$'
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
- `Final decisions: [True, False, True, ...] True count: N` means N out of M factors passed in that batch

### Step 7: Filter and deduplicate

**Quality criteria for accepting a factor (ALL must pass):**

1. **final_decision == True** — code executed without error, output format is correct (MultiIndex [datetime, instrument], single float64 column)
2. **Not a duplicate** of any existing factor in `tushare/new_factor.md`

**Dedup rules (apply in order):**
1. Exact name match (case-insensitive, after normalizing `_`/`-`)
2. Same formula with different variable naming → DUPLICATE
3. Same underlying concept + same window → DUPLICATE (e.g., `momentum_20d` = `MediumTermMomentum_20d` = `20d_return`)
4. Same concept + different window → DIFFERENT factor (e.g., `momentum_5d` ≠ `momentum_20d`)
5. Different data source for same concept → DIFFERENT (e.g., PE_ttm based ≠ PB based, even if both are value)
6. If uncertain, compare the actual `factor_formulation` field — normalize whitespace and compare

**How to check for IC/Rank IC:** The rdagent evaluator's `final_decision: True` already validates:
- Code execution success
- Output format correctness (MultiIndex, float64, single column)
- Factor values are finite and reasonable

IC/Rank IC values are computed by the downstream Qlib model training (Stage2 walk-forward), not by fin_factor itself. The `final_decision: True` is the primary quality gate. If individual IC values appear in the output, prefer factors with IC > 0.02 or Rank IC > 0.02, but do NOT reject a passing factor solely for missing IC values — the IC is context-dependent.

### Step 8: Append qualified factors to new_factor.md

For each qualifying factor, append to `tushare/new_factor.md` in this format:

```markdown
## N. factor_name

- **类型**：<Chinese category label from the table above>
- **描述**：<One-line Chinese+English description of what the factor measures and how to interpret it>
- **公式**：

  $$<LaTeX formula>$$

- **变量**：
  - $var_1$：<description>
  - $var_2$：<description>
- **数据来源**：cn_extra_data <specific fields used>
- **评估反馈**：Code execution successful, output format correct (MultiIndex [datetime, instrument], single float64 column), no anomalies in factor values.
---
```

**Rules for appending:**
- Increment the section number (`N`) from the last existing factor
- Add the new entry before the `## 使用方法` section
- Update the summary table at the top of the file: add a row for each new factor
- Update the "经多轮自动化因子挖掘" description if the count changes
- Keep the LaTeX clean and well-formatted
- Link variables to their cn_extra_data field names (e.g., `$pe_ttm`)

### Step 9: Cleanup

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
| All factors rejected | `True count: 0` in final decisions | Check if Cython modules are compiled; verify proxy connectivity |
| Proxy port conflict | `Address already in use` | `lsof -i :18080` and kill existing process |
| HDF5 data missing | `daily_pv_all.h5 not found` | Run `python rdagent_workspace/factor_data_template/generate.py` |

## Progress Tracking

Update this table after each successful run. Mark categories with factors:

| Category | Status | Factors discovered |
|----------|--------|--------------------|
| 动量/反转 | DONE | MediumTermMomentum_20d, 20_day_reversal |
| 波动率 | DONE | RealizedVolatility_20d |
| 震荡 | DONE | RSI_14d |
| 流动性 | DONE | 5_day_volume_change |
| 估值 | DONE | trailing_PE_ratio |
| 量价 | DONE | obv_slope_10day |
| 风险调整 | DONE | sharpe_10day |
| 质量 | TODO | ROE, ROA, profit margin factors needed |
| 成长 | TODO | EPS/revenue/BPS growth factors needed |
| 财务杠杆 | TODO | debt/assets, liability/equity factors needed |
| 现金流 | TODO | OCF, FCF, OCF/profit factors needed |
| 市值/规模 | TODO | total_mv, circ_mv factors needed |
| 股息 | TODO | dv_ratio, dv_ttm factors needed |
| 运营效率 | TODO | op/revenue, revenue_ps factors needed |
