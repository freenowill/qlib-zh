---
name: gen_tushare_h5
description: Generate HDF5 data files from tushare/extra_data for a given index (CSI300/CSI1000/etc). Produces daily_pv_{index}.h5 with 58 raw fields, used by build_features_from_h5.py and fin_factor.
---

# /gen_tushare_h5 — Generate HDF5 by Index

Reads `tushare/extra_data/{SYMBOL}/*.csv`, filters by index constituents, extracts 58 raw fields, and outputs an HDF5 file for downstream factor computation.

## Usage

```bash
# CSI300 (default)
python3 scripts/generate_h5.py

# CSI1000
python3 scripts/generate_h5.py csi1000
python3 scripts/generate_h5.py csi1000 --output /tmp/daily_pv_csi1000.h5

# CSI500 / CSI800 / all stocks
python3 scripts/generate_h5.py csi500
python3 scripts/generate_h5.py all
```

## Output

`rdagent_workspace/factor_data_template/daily_pv_{index}.h5`

MultiIndex DataFrame: (datetime, instrument) × 58 columns (market + valuation + fundamental + financial).

## Files modified

| File | Role |
|------|------|
| `scripts/generate_h5.py` | Main extraction script (parallel, calendar building, merge) |
| `tushare/cn_data/instruments/{index}.txt` | Index constituent definitions (read-only) |
| `tushare/extra_data/{SYMBOL}/*.csv` | Raw TuShare CSV data (read-only) |

## Edge cases

- **Stock in index but no extra_data dir**: reported in output, silently skipped
- **Stock has partial CSV files**: missing fields become NaN in H5
- **Stock suspended/trading halt**: no daily.csv data for that period → NaN

## Verification

```bash
python3 -c "
import pandas as pd
h5 = pd.HDFStore('rdagent_workspace/factor_data_template/daily_pv_csi300.h5', 'r')
print(f'Shape: {h5.get_storer(\"data\").shape}')
cols = h5.get_storer('data').non_index_axes[0][1]
print(f'Columns: {len(cols)}')
insts = h5.get_storer('data').non_index_axes[1][1]
print(f'Instruments: {len(insts)}')
h5.close()
"
```
