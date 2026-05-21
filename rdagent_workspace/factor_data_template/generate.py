"""Generate daily_pv_all.h5 and daily_pv_debug.h5 from cn_extra_data.

Reads ALL 58 features (market, valuation, fundamental, financial),
outputs HDF5 files expected by rdagent fin_factor pipeline.

Usage (inside Docker or with qlib installed):
    cd rdagent_workspace/factor_data_template && python generate.py
"""
import os
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# ── Provider: cn_extra_data (58 features) ──────────────────────────
_provider = os.path.expanduser("~/.qlib/qlib_data/cn_extra_data")
if not Path(_provider).exists():
    _provider = os.path.expanduser("~/.qlib/qlib_data/cn_extra_data_improve")
print(f"Provider: {_provider}")
qlib.init(provider_uri=_provider, region=REG_CN)

# ── All 58 fields ──────────────────────────────────────────────────
ALL_FIELDS = [
    # Market data (10)
    "adjclose", "amount", "change", "close", "factor",
    "high", "low", "open", "volume", "vwap",
    # Valuation (15)
    "pe", "pe_ttm", "pb", "ps", "ps_ttm",
    "dv_ratio", "dv_ttm", "turnover", "turnover_f",
    "vol_ratio", "total_mv", "circ_mv", "total_sh", "float_sh", "free_sh",
    # Fundamental / fina_indicator (20)
    "eps", "dt_eps", "bps", "ocfps", "cfps", "revenue_ps",
    "undist_ps", "roe", "roe_yearly", "roa_yearly", "npta",
    "netprofit_margin", "debt_to_assets", "assets_to_eqt",
    "eps_yoy", "netprofit_yoy", "roe_yoy", "bps_yoy",
    "assets_yoy", "revenue_yoy",
    # Financial statements (13)
    "revenue", "n_income", "operate_profit",
    "total_assets", "total_liab", "total_equity",
    "ocf", "icf", "fcf",
    "liab_to_eqty", "op_to_revenue", "ocf_to_profit", "ocf_to_assets",
]

fields = [f"${f}" for f in ALL_FIELDS]
print(f"Fields: {len(fields)}")

# ── Instruments ────────────────────────────────────────────────────
all_instruments = D.list_instruments(
    D.instruments(market="all"), freq="day", as_list=True
)
print(f"Instruments: {len(all_instruments)}")

# ══════════════════════════════════════════════════════════════════════
# Full dataset: batch-process to avoid OOM
# ══════════════════════════════════════════════════════════════════════
BATCH_SIZE = int(os.environ.get("GENERATE_BATCH_SIZE", "400"))
batches = [
    all_instruments[i : i + BATCH_SIZE]
    for i in range(0, len(all_instruments), BATCH_SIZE)
]
print(f"Batches: {len(batches)} (batch_size={BATCH_SIZE})")

parts = []
for i, batch in enumerate(batches):
    print(f"  Batch {i+1}/{len(batches)}: {len(batch)} instruments ...", end=" ", flush=True)
    part = (
        D.features(batch, fields, freq="day")
        .swaplevel()
        .sort_index()
    )
    print(f"shape={part.shape}")
    parts.append(part)

print("Concatenating ...", end=" ", flush=True)
data = pd.concat(parts)
print(f"full shape={data.shape}")
data.to_hdf("./daily_pv_all.h5", key="data")
del parts, data
print("daily_pv_all.h5 written")

# ══════════════════════════════════════════════════════════════════════
# Debug dataset: CSI 300 subset, target date range
# ══════════════════════════════════════════════════════════════════════
_debug_start = os.environ.get("GENERATE_DEBUG_START", "2020-01-01")
_debug_end = os.environ.get("GENERATE_DEBUG_END", "2024-12-31")

try:
    csi300 = D.list_instruments(
        D.instruments(market="csi300"), freq="day", as_list=True
    )
except Exception:
    csi300 = all_instruments[:300]

print(f"Debug instruments: {len(csi300)}  range=[{_debug_start}, {_debug_end}]")
debug_data = (
    D.features(csi300, fields, start_time=_debug_start, end_time=_debug_end, freq="day")
    .swaplevel()
    .sort_index()
)
print(f"Debug shape={debug_data.shape}")
debug_data.to_hdf("./daily_pv_debug.h5", key="data")
print("daily_pv_debug.h5 written")

print("Done.")
