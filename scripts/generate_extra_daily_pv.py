"""Generate daily_pv_all.h5 and daily_pv_debug.h5 from cn_extra_data.

Reads ALL features (58 fields: market, valuation, fundamental, financial),
outputs HDF5 files expected by rdagent fin_factor pipeline.
"""
import os
import sys

import qlib

# Use cn_extra_data which has 58 features (vs 10 in default cn_data)
provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_extra_data")
qlib.init(provider_uri=provider_uri)

from qlib.data import D

# All features available in cn_extra_data
ALL_FIELDS = [
    # Market data
    "adjclose", "amount", "change", "close", "factor",
    "high", "low", "open", "volume", "vwap",
    # Valuation
    "pe", "pe_ttm", "pb", "ps", "ps_ttm",
    "dv_ratio", "dv_ttm", "turnover", "turnover_f",
    "vol_ratio", "total_mv", "circ_mv", "total_sh", "float_sh", "free_sh",
    # Fundamental (fina_indicator)
    "eps", "dt_eps", "bps", "ocfps", "cfps", "revenue_ps",
    "undist_ps", "roe", "roe_yearly", "roa_yearly", "npta",
    "netprofit_margin", "debt_to_assets", "assets_to_eqt",
    "eps_yoy", "netprofit_yoy", "roe_yoy", "bps_yoy",
    "assets_yoy", "revenue_yoy",
    # Financial statements
    "revenue", "n_income", "operate_profit",
    "total_assets", "total_liab", "total_equity",
    "ocf", "icf", "fcf",
    "liab_to_eqty", "op_to_revenue", "ocf_to_profit", "ocf_to_assets",
]

fields = [f"${f}" for f in ALL_FIELDS]
print(f"Reading {len(fields)} fields from {provider_uri}")

instruments = D.instruments()
print(f"Instruments: {len(instruments)}")

# Full dataset
data = (
    D.features(instruments, fields, freq="day")
    .swaplevel()
    .sort_index()
)
print(f"Full data shape: {data.shape}")
data.to_hdf("./daily_pv_all.h5", key="data")
print("daily_pv_all.h5 written")

# Debug dataset (subset: first 100 instruments, 2018-2020)
try:
    data_debug = (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2020-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    instruments_subset = data_debug.reset_index()["instrument"].unique()[:100]
    data_debug = (
        data_debug.swaplevel()
        .loc[instruments_subset]
        .swaplevel()
        .sort_index()
    )
    print(f"Debug data shape: {data_debug.shape}")
    data_debug.to_hdf("./daily_pv_debug.h5", key="data")
    print("daily_pv_debug.h5 written")
except Exception as e:
    print(f"Debug data fallback (subset creation issue): {e}")
    # Fallback: use head
    subset = data.iloc[:100000]
    subset.to_hdf("./daily_pv_debug.h5", key="data")
    print(f"Debug data (fallback) shape: {subset.shape}")

print("Done.")
