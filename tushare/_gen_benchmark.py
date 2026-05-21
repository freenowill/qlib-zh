import numpy as np
from pathlib import Path

IMPROVE_DIR = Path("/workspace/tushare/cn_extra_data_improve")

with open(IMPROVE_DIR / "calendars" / "day.txt") as f:
    calendar = [line.strip() for line in f if line.strip()]
cal_compact = [d.replace("-", "") for d in calendar]

import sxsc_tushare as sx
sx.set_token("4cbb80cf41ae83b53f9bc431a502c328565e53938bce7cadce52bc2a")
api = sx.get_api(env="prd")
df = api.query("index_daily", ts_code="000300.SH", start_date="20100101", end_date="20260520")
print(f"Pulled {len(df)} days")

df["trade_date"] = df["trade_date"].astype(str)
close_map = dict(zip(df["trade_date"], df["close"].astype(float)))

feat_dir = IMPROVE_DIR / "features" / "sh000300"
feat_dir.mkdir(parents=True, exist_ok=True)

result = np.full(len(cal_compact), np.nan, dtype=np.float32)
for i, d in enumerate(cal_compact):
    v = close_map.get(d)
    if v is not None and v > 0:
        result[i] = np.float32(v)

any_bin = next((IMPROVE_DIR / "features" / "sh600000").glob("*.day.bin"))
start_idx = int(np.fromfile(str(any_bin), dtype="<f4", count=1)[0])

header = np.array([float(start_idx)], dtype="<f4")
np.concatenate([header, result.astype("<f4")]).tofile(str(feat_dir / "close.day.bin"))
print(f"close.bin: {np.sum(~np.isnan(result))} non-null / {len(calendar)} days")

inst_file = IMPROVE_DIR / "instruments" / "all.txt"
lines = inst_file.read_text().splitlines()
if not any(l.startswith("SH000300") for l in lines):
    with open(inst_file, "a") as f:
        f.write(f"SH000300\t{calendar[0]}\t{calendar[-1]}\n")
    print("Added SH000300 to all.txt")
print("Done")
