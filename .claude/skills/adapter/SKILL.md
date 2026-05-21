---
name: adapter
description: Sync new_factor.md changes to the downstream pipeline. When new factors are added, updates process_extra_data.py (computation), YAML template (training), and stage1 health check (validation) so run_explore_data.sh and run_new_factor_practice use all factors.
---

# /adapter — Pipeline Adapter Skill

When `tushare/new_factor.md` gets new factors (from `/factor-mining` or manual), this skill adapts the downstream pipeline: factor computation, training config, and health checks.

## Files modified by this skill

| File | Role |
|------|------|
| `tushare/process_extra_data.py` | Factor computation (`NEW_FACTOR_RAW_INPUTS`, `_INTERMEDIATE_RAW`, `compute_new_factor_factors()`, `FACTOR_CLASSIFICATION`, header) |
| `examples/benchmarks/LightGBM/workflow_config_lightgbm_AlphaExtra.yaml` | Training feature list (`factor_config.direct`) |
| `scripts/practice/stage1_data_health_extra.py` | Validation feature set (`EXPECTED_FEATURES`) |
| `tushare/run_explore_data.sh` | Header comment (factor count) |

---

## Step 1: Parse new_factor.md

Extract all factors and their data dependencies:

```bash
# All factor names
grep -E '^## [0-9]+\. ' tushare/new_factor.md | sed 's/^## [0-9]*\. //'

# All data source references ($field_name)
grep -oP '\$\w+' tushare/new_factor.md | sort -u
```

For each factor, determine:
- **bin filename**: snake_case, e.g. `PB_Ratio` → `pb_ratio`, `trailing_PE_ratio` → `pe_ttm`
- **type**: `direct` (reads a raw field), `derived` (computed from raw fields), or `cross-sectional` (needs all stocks on same date)
- **raw dependencies**: which CSV columns it needs (trace `$field` references in the formula)

### Normalization rules for bin filenames
- Lowercase, underscores
- Common mappings: `roe` → `roe_yearly`, `net_profit_margin` → `netprofit_margin`, `trailing_PE_ratio` → `pe_ttm`, `MediumTermMomentum_20d` → `momentum_20d`
- When in doubt, match the formula's actual data column name

---

## Step 2: Diff — what's missing?

### 2a. Factors already in process_extra_data.py

```bash
# Derived factors (computed in compute_new_factor_factors)
grep -oP 'new_factors\["[^"]+"\]' tushare/process_extra_data.py | sort -u

# Direct factors (listed in comment at end of compute_new_factor_factors)
grep -A8 '已在 full_convert_csv_to_bin' tushare/process_extra_data.py
```

### 2b. Factors already in YAML

```bash
sed -n '/factor_config:/,/port_analysis_config:/p' examples/benchmarks/LightGBM/workflow_config_lightgbm_AlphaExtra.yaml | grep '^        - '
```

### 2c. Factors already in stage1

```bash
python3 -c "
import re
t = open('scripts/practice/stage1_data_health_extra.py').read()
m = re.search(r'EXPECTED_FEATURES = frozenset\(\{(.+?)\}\)', t, re.DOTALL)
if m: print(m.group(1))
"
```

### 2d. Compute the gap

```
missing = new_factor.md factors - process_extra_data factors
```

---

## Step 3: Implement missing factors in process_extra_data.py

### 3a. Assess raw data dependencies

For each missing factor, check if its raw fields exist in the CSV extraction dicts (lines 357-392):

- `EXTRA_DAILY_FEATURES` — daily_basic.csv columns (pe, pe_ttm, pb, ps, turnover, total_mv, etc.)
- `FUNDAMENTAL_FEATURES` — fina_indicator.csv columns (eps, roe_yearly, netprofit_margin, bps, etc.)
- `FINANCIAL_STATEMENT_SOURCES` — income/balancesheet/cashflow CSV columns

If a raw field is in one of these dicts but **not in `NEW_FACTOR_RAW_INPUTS`** (line 633), add it. `NEW_FACTOR_RAW_INPUTS` controls which fields are written to bin files.

### 3b. Audit the `_INTERMEDIATE_RAW` cleanup list (line 1089)

Fields in `_INTERMEDIATE_RAW` are deleted after factor computation. If a new factor needs a raw field as a **direct** pass-through (not just as a compute intermediate), remove that field from the cleanup list.

### 3c. Add derived factor code

In `compute_new_factor_factors()` (starts line 714), append new blocks. Use these patterns:

**Price momentum (N-day):**
```python
# ---- Factor N: factor_name ----
momN = np.full(n, np.nan, dtype=np.float64)
for i in range(N, n):
    if close[i] > 0 and close[i - N] > 0:
        momN[i] = close[i] / close[i - N] - 1.0
new_factors["factor_name"] = momN
```

**Volume ratio (N-day average):**
```python
if volume is not None:
    vrN = np.full(n, np.nan, dtype=np.float64)
    for i in range(N-1, n):
        seg = volume[i - N + 1 : i + 1]
        valid = seg[~np.isnan(seg) & (seg > 0)]
        if len(valid) >= 3:
            vrN[i] = volume[i] / np.mean(valid)
    new_factors["factor_name"] = vrN
```

**Risk-adjusted (momentum / volatility):**
```python
ramom = np.full(n, np.nan, dtype=np.float64)
for i in range(max(N_vol, N_mom), n):
    if np.isnan(mom[i]):
        continue
    seg = ret[i - (N_vol - 1) : i + 1]
    valid = seg[~np.isnan(seg)]
    if len(valid) >= 5:
        vol = np.std(valid, ddof=1)
        if vol > 0:
            ramom[i] = mom[i] / (vol * np.sqrt(252))
new_factors["factor_name"] = ramom
```

**Simple arithmetic (e.g., 1/x):**
```python
if pb is not None:
    bp = safe_div(np.ones(n, dtype=np.float64), pb)
    new_factors["book_to_price"] = bp
```

**Direct pass-through:** No compute code. Just ensure the raw field is in `NEW_FACTOR_RAW_INPUTS` and NOT in `_INTERMEDIATE_RAW`. Add a comment at the end of `compute_new_factor_factors()` listing it.

### 3d. Cross-sectional factors

Factors that compare stocks on the same date (e.g., sector-relative PB) need a post-processing pass because `process_symbol()` processes one stock at a time.

Add a function after `compute_new_factor_factors()`:

```python
def _stock_sector(symbol: str) -> str:
    code = symbol[2:]
    if symbol.startswith("SH"):
        return "star" if code.startswith("688") else "sh_main"
    elif symbol.startswith("SZ"):
        if code.startswith("30"): return "chinext"
        if code.startswith("002") or code.startswith("003"): return "sme"
        return "sz_main"
    return "other"

def compute_cross_sectional_factors(feat_root: Path, calendar: list[str]) -> list[str]:
    """Compute cross-sectional factors across all stocks at each date.
    Call AFTER all stocks have been individually processed.

    Returns list of factor names written.
    """
    stock_dirs = sorted(d for d in (feat_root).iterdir() if d.is_dir())
    if not stock_dirs:
        return []

    # Read pb and sector for all stocks
    stock_pb = {}   # symbol -> np.array
    for d in stock_dirs:
        sym = d.name
        fpath = d / "pb.day.bin"
        if not fpath.exists():
            continue
        data = np.fromfile(str(fpath), dtype="<f4")[1:].astype(np.float64)
        stock_pb[sym] = data

    if not stock_pb:
        return []

    n = len(next(iter(stock_pb.values())))
    sectors = {sym: _stock_sector(sym) for sym in stock_pb}

    for sym, pb_arr in stock_pb.items():
        sector = sectors[sym]
        peers = [s for s, sec in sectors.items() if sec == sector and s != sym]
        rel_pb = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            peer_vals = []
            for p in peers:
                if i < len(stock_pb[p]) and not np.isnan(stock_pb[p][i]):
                    peer_vals.append(stock_pb[p][i])
            if peer_vals and not np.isnan(pb_arr[i]):
                rel_pb[i] = pb_arr[i] - np.median(peer_vals)
        # Write bin
        any_bin = next((feat_root / sym).glob("*.day.bin"), None)
        start_idx = int(np.fromfile(str(any_bin), dtype="<f4", count=1)[0]) if any_bin else 0
        write_bin(feat_root / sym / "sector_relative_pb.day.bin", start_idx, rel_pb.astype(np.float32))

    return ["sector_relative_pb"]
```

Call this from the script's `main()` after the parallel processing loop completes.

### 3e. Update FACTOR_CLASSIFICATION (lines 1095-1115)

Add new factor names to the appropriate category lists:

```python
FACTOR_CLASSIFICATION["估值"] = [f for f in derived if f in (
    "earnings_yield", "book_to_price",
)]
FACTOR_CLASSIFICATION["风险调整"] = [f for f in derived if f in (
    "sharpe_10d", "momentum_vol_adjusted_20", "risk_adjusted_momentum_5d_20d",
)]
```

Also add new direct factors to the last category block:
```python
FACTOR_CLASSIFICATION["基本面/财务"] = [f for f in features
    if f in ("pe_ttm", "pb", "roe_yearly", "netprofit_margin")]
```

### 3f. Update header comments

- `process_extra_data.py` line 8: update "24 个因子公式" → new count
- `run_explore_data.sh` line 9: update "(N 个因子)" → new count

---

## Step 4: Update YAML template

Edit `examples/benchmarks/LightGBM/workflow_config_lightgbm_AlphaExtra.yaml`, section `factor_config.direct`:

Add each new factor's bin filename (without `.day.bin`). Keep alphabetical order:

```yaml
factor_config:
    direct:
        - avg_normalized_range_5d
        - book_to_price
        - earnings_yield
        - intraday_volatility
        - momentum_10d
        - momentum_20d
        - momentum_5d
        - momentum_vol_adjusted_20
        - netprofit_margin
        - obv_slope_10d
        - pb_ratio
        - pe_ttm
        - realized_volatility_20d
        - reversal_1d
        - reversal_2d
        - reversal_20d
        - reversal_5d
        - risk_adjusted_momentum_5d_20d
        - roe_yearly
        - rsi_14d
        - sector_relative_pb
        - sharpe_10d
        - turnover_trend
        - volume_change_5d
        - volume_ratio_5d
        - volume_ratio_5d_20d
        - volume_weighted_momentum_5d
        - vwap_deviation_10d
        - vwap_deviation_5d
```

The YAML list is the single source of truth for model training — every factor in the pipeline must be listed here.

---

## Step 5: Update stage1 health check

Edit `scripts/practice/stage1_data_health_extra.py`, the `EXPECTED_FEATURES` frozenset (line 29):

Add each new factor name. Keep `close` first (needed for label expression).

```python
EXPECTED_FEATURES = frozenset({
    "close", "pe_ttm", "roe_yearly", "netprofit_margin",
    "momentum_5d", "momentum_10d", "momentum_20d",
    "reversal_1d", "reversal_2d", "reversal_5d", "reversal_20d",
    "realized_volatility_20d", "intraday_volatility",
    "avg_normalized_range_5d", "rsi_14d",
    "volume_change_5d", "volume_ratio_5d", "volume_ratio_5d_20d",
    "turnover_trend", "obv_slope_10d", "volume_weighted_momentum_5d",
    "vwap_deviation_5d", "vwap_deviation_10d",
    "sharpe_10d", "earnings_yield",
    # NEW factors go here
})
```

---

## Step 6: Verify consistency

```bash
# Factor count should match across all sources
echo "=== new_factor.md ==="
grep -cE '^## [0-9]+\. ' tushare/new_factor.md

echo "=== process_extra_data.py derived ==="
grep -oP 'new_factors\["[^"]+"\]' tushare/process_extra_data.py | sort -u | wc -l

echo "=== YAML direct list ==="
python3 -c "
import yaml
with open('examples/benchmarks/LightGBM/workflow_config_lightgbm_AlphaExtra.yaml') as f:
    cfg = yaml.safe_load(f)
print(len(cfg['data_handler_config']['factor_config']['direct']))
"

echo "=== stage1 EXPECTED_FEATURES (minus close) ==="
python3 -c "
import re
t = open('scripts/practice/stage1_data_health_extra.py').read()
m = re.search(r'EXPECTED_FEATURES = frozenset\(\{(.+?)\}\)', t, re.DOTALL)
if m:
    items = [x.strip().strip('\"') for x in m.group(1).split(',')]
    print(len([x for x in items if x != 'close']))
"

# Cross-reference: every YAML factor must exist as a bin in process_extra_data output
```

---

## Step 7: Regenerate data

```bash
cd tushare
# Resume from existing data, only re-process stale stocks
./run_explore_data.sh --resume
```

Then validate with a stage1-only run:

```bash
bash run_new_factor_practice _adapter_check stage=1 end_stage=1
```

If stage1 passes, run a full stage2 training to confirm model can load all features:

```bash
bash run_new_factor_practice _adapter_check stage=2 end_stage=2
```

---

## Real-world example: current state (2026-05-21)

All 30 factors from new_factor.md are now implemented in the pipeline. Last sync: `/adapter` executed on 2026-05-21.

If new factors are added (e.g., #31+), re-run this skill to detect and implement the gap.

---

## Edge cases

**Factor removed from new_factor.md**: Rare. If it happens, remove from YAML and stage1, keep the computation code (harmless to have extra bin files).

**Factor renamed in new_factor.md**: Treat as remove old + add new. Update YAML and stage1 accordingly.

**New factor needs a CSV field not yet in EXTRA_DAILY_FEATURES**: Add the `csv_column: alias` mapping to the dict, then add the alias to `NEW_FACTOR_RAW_INPUTS`.

**Cross-sectional factor with many fields**: The `compute_cross_sectional_factors` pattern can be extended to read multiple fields per stock. Keep memory in check — process in batches if > 5000 stocks.
