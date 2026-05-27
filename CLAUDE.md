# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a Chinese-market quantitative investment platform built on **Qlib** (Microsoft's AI-oriented quant framework) integrated with **RDAgent** for automated factor mining. The codebase is a fork/customization of upstream Qlib with TuShare-based data pipelines, a 6-stage alpha research workflow, and Docker-based rdagent execution.

## Repo architecture (what's ours vs upstream)

| Layer | Location | Role |
|-------|----------|------|
| **Upstream Qlib** | `qlib/` | Core library: data layer, model backends, backtest engine, workflow runner. Forked from Microsoft qlib; do not refactor casually. |
| **Our scripts & pipelines** | `scripts/` | Data collection (PIT, yahoo), rdagent wrappers, stage runners, analysis tools. This is where most custom work lives. |
| **Our data pipelines** | `tushare/` | TuShare data fetching/processing → qlib binary format. 58 base fields + derived factors computed by `build_features_from_h5.py`. Final `cn_extra_data_h5/` has ~318 `.day.bin` per stock. Also contains `source_code/` (sxsc-tushare library). |
| **Our Docker/rdagent infra** | root + `scripts/` | `Dockerfile`, `docker-compose.yml`, `sitecustomize.py`, `run_fin_factor_with_cap.py` — glue between qlib data and rdagent runtime. |
| **Workspace** | `rdagent_workspace/` | Runtime data for rdagent (HDF5 source data, templates). |
| **Outputs** | `DATA/`, `mlruns/` | Experiment results, MLflow tracking. |
| **Scratch/temp** | `git_ignore_folder/` | HDF5 files, logs, temp data. Mounted into Docker at `/data/git_ignore_folder`. |
| **Claude Code config** | `.claude/` | Skills (`factor-mining`, `gen_tushare_h5`), permissions (`settings.json`, `settings.local.json`). |

## Local modifications to upstream Qlib

We patch a few Qlib files for bug fixes or feature additions. These are mounted into Docker containers at runtime:

- **`qlib/data/storage/file_storage.py`** — int-casting fix for `fp.seek()` on Linux to prevent TypeError from numpy int64 values
- **`qlib/data/ops.py`** — custom modifications (mounted into container)
- **`qlib/contrib/data/handler_extra.py`** — our custom AlphaExtra handler. Supports precomputed mode: when YAML `factor_config.direct: true`, auto-scans `cn_extra_data/features/` for all `.day.bin` files and loads them directly — no expression evaluation at training time. Otherwise falls back to default expression-based mode with Kbar/Price/Rolling/Value/Quality/Growth/Leverage/Liquidity categories.

## Main workflows

### 0. Exploration data pipeline

Builds `cn_extra_data` with 58 base features + derived factors + all factors from `new_factor.md`. This generates the data consumed by AlphaExtra walk-forward and `/factor-mining`.

Pipeline phases:
- **Data fetch**: `tushare/get_tushare_data.py` — fetches daily/quarterly/annual data from TuShare API per stock, writes raw CSV to `tushare/extra_data/{SYMBOL}/`
- **Health check**: `tushare/check_health.py` — validates CSV completeness per stock (file presence, date coverage, row counts), auto-fills missing data via TuShare API. Supports `--check-only` for dry-run mode
- **HDF5 generation**: `rdagent_workspace/factor_data_template/generate.py` — reads raw CSVs, builds `daily_pv_all.h5` (58 fields)
- **Factor computation**: `scripts/practice/build_features_from_h5.py` — reads HDF5, computes Alpha158 + all derived factors from `new_factor.md`, applies cross-sectional z-score normalization, outputs qlib binary format to `cn_extra_data_h5/`

Uses Docker image `zhuhai123/local_qlib:v1-tushare`. Stock universe is defined by `tushare/cn_data/instruments/` (all, csi300, csi500, csi800, csi1000).

### 1. Alpha158 6-stage pipeline

```
bash run_alpha158_practice <experiment_name> [stage=N] [end_stage=M]
```

Stages: Stage1 (data health) → Stage2 (walk-forward training, LightGBM) → Stage3 (signal filtering) → Stage4 (portfolio/risk) → Stage5 (backtest) → Stage6 (summary).

Key env vars: `WALK_FORWARD_START_DATE`, `WALK_FORWARD_HISTORY_YEARS`, `WALK_FORWARD_SEGMENT_YEARS`, `TARGET_MARKET`, `TARGET_BENCHMARK`, `HOLD_NUM`, `CASH_TOTAL`, `TX_FEE_RATE`, `STAMP_DUTY_RATE`.

Additional variants for different markets: `run_alpha158` (csi300), `run_alpha158_csi500`, `run_alpha158_small` (small-cap), `run_alpha_360_csi500`.

Outputs land in `DATA/analysis_outputs/<experiment_name>/`.

### 2. AlphaExtra walk-forward (`run_new_factor_practice`)

```
bash run_new_factor_practice <experiment_name> [stage=N] [end_stage=N] [missing_threshold=N] [new_factor_only]
```

Three stages (additional `new_factor_only` flag skips Alpha158 and uses only `new_factor.md` factors):
- **Stage0**: Build features from H5 — reads `daily_pv_all.h5` (same H5 as `/factor-mining`), computes Alpha158 (158) + all independent factors from `new_factor.md` + overlapping factors (≈199 total, varies as factors are added), applies cross-sectional z-score normalization per date, outputs to `~/.qlib/qlib_data/cn_extra_data_h5/` (qlib binary format). Script: `scripts/practice/build_features_from_h5.py`.
- **Stage1**: Data health check on `cn_extra_data_h5` — filters stocks with too many missing values. Outputs filtered data to `cn_extra_data_h5_filtered/`.
- **Stage2**: Walk-forward training — uses AlphaExtra handler in `direct: true` mode (auto-discovers all `.day.bin` files), YAML template `workflow_config_lightgbm_AlphaExtra.yaml` with `provider_uri: /root/.qlib/qlib_data/cn_extra_data_h5`.

The H5 file is shared with `/factor-mining` — both pipelines use the same data source (`rdagent_workspace/factor_data_template/daily_pv_all.h5`). Use `H5_FILE` env var to switch to debug version.

If Stage1 generated filtered data, Stage2 auto-mounts the filtered dataset overlay while keeping the original as a symlink source.

### 3. RDAgent fin_factor (automated factor mining)

The canonical command (from `document_fin_factor.txt`):

```bash
HOST_PWD="$(pwd)"; docker run --rm \
  -e PYTHONPATH="$HOST_PWD" \
  -e OPENAI_API_KEY='...' \
  -e CHAT_MODEL='openai/glm-4.7' \
  -e OPENAI_API_BASE='...' \
  -v "$HOST_PWD:$HOST_PWD" \
  -v "$HOME/.qlib:/root/.qlib" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --env-file "$HOST_PWD/.env" \
  -w "$HOST_PWD" \
  zhuhai123/qlib-rdagent:v1 \
  rdagent fin_factor --step-n 1 --loop-n 1
```

Critical mount pattern: use `-v "$HOST_PWD:$HOST_PWD"` (not `-v "$HOST_PWD:/work"`) because `~/.qlib/qlib_data/cn_extra_data` is a symlink pointing to the host path — the host path must exist inside the container.

There is also a Claude Code skill `/factor-mining` that automates the full pipeline: dedup checking, proxy setup, HDF5 generation, fin_factor execution, result parsing, and updating `new_factor.md`/`fail_new_factor.md`.

**FORCE_LOCAL_STUB mode** — for testing without real LLM calls, set `FORCE_LOCAL_STUB=1` in the environment. `sitecustomize.py` will stub all LLM API calls to return a dummy Momentum_10 factor. Also activates the identical stub in `run_fin_factor_with_cap.py`.

### PIT (Point-in-Time) data pipeline

`run_data.sh` downloads quarterly/annual fundamental data via baostock and dumps it into qlib format at `~/.qlib/qlib_data/cn_data`. Uses `scripts/data_collector/pit/collector.py` and `scripts/dump_pit.py`. Runs inside Docker (`qlib-rdagent:latest`).

```bash
bash run_data.sh [run_tag]
```

## Data sources

| Directory | Contents | Format |
|-----------|----------|--------|
| `~/.qlib/qlib_data/cn_data` | Standard qlib CN data (downloaded + PIT) | `.day.bin` + `instruments/` + `calendars/` |
| `tushare/cn_extra_data/` | Full dataset: 58 base + 25 derived + 234 Alpha158 (Kbar/Price/Rolling/Fundamental/Price-Volume) + 1 sector = 318 bins/stock. All features pre-computed and cross-sectional normalized. | qlib binary format (symlinked to `~/.qlib/qlib_data/cn_extra_data`) |
| `tushare/cn_extra_data_improve/` | Legacy directory (no longer generated; `cn_extra_data` now includes all factors) | qlib binary format |
| `tushare/extra_data/` | Raw TuShare CSV downloads (~5500 stock dirs) | Various formats |
| `tushare/cn_data/instruments/` | Stock universe definitions (all, csi300, csi500, csi800, csi1000) | Tab-separated (code + date range) |
| `tushare/cn_data/sw_industry.csv` | Shenwan (申万) level-1 industry classification (SW2021) — stock→industry mapping, generated by `tushare/fetch_sw_industry.py` | CSV (symbol, sw_industry) |
| `tushare/new_factor.md` | Factor registry — source of truth for factor definitions, classification, and Alpha158 overlap analysis | Markdown tables |
| `tushare/source_code/` | Copy of the `sxsc-tushare` library (TuShare API wrapper) used by the data pipeline | Python package |
| `git_ignore_folder/` | Runtime HDF5 files (`daily_pv_all.h5`, `daily_pv_debug.h5`), logs, temp data. Mounted into Docker at `/data/git_ignore_folder`. | HDF5 + logs |

qlib init pattern:
```python
import qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_extra_data")
```

Key qlib data API gotcha: `D.instruments()` returns a dict `{'market': 'all', 'filter_pipe': []}` — not a list. Use `D.list_instruments(D.instruments(market='all'), freq='day', as_list=True)` to get actual instrument codes.

## Build & development

```bash
# Install qlib in editable mode with Cython extensions
make install

# Install with all optional deps
make dev

# Lint
make lint        # black + pylint + flake8 + mypy + nbqa
make black       # just black (line length 120)

# Build wheel
make build

# Run tests (requires test deps: make test)
pytest qlib/tests/

# Run a single test
pytest qlib/tests/test_data.py::TestClass::test_name
```

The `setup.py` compiles two Cython extensions: `qlib.data._libs.rolling` and `qlib.data._libs.expanding`. These live in `qlib/data/_libs/`. When running inside Docker, ensure the Linux-compiled `.so` files exist (`.cpython-310-x86_64-linux-gnu.so`); macOS `.so` files won't work.

## Docker images

- `zhuhai123/qlib-rdagent:v1` — rdagent + qlib, used for fin_factor and AlphaExtra walk-forward
- `zhuhai123/local_qlib:latest` — qlib only, used for the 6-stage pipeline
- `zhuhai123/local_qlib:v1-tushare` — qlib + tushare, used for data fetching (`get_tushare_data.py`, `check_health.py`)
- `qlib-rdagent:latest` — local base image built from `Dockerfile`

Build custom image: `bash build_docker_image.sh` or `docker compose build`.

Start an interactive container shell:
```bash
docker compose run --rm rdagent bash
```

## Claude Code skills

This repo defines custom skills in `.claude/skills/`:

- **`/factor-mining`** — automated factor mining pipeline. Runs rdagent fin_factor with DeepSeek LLM, checks for duplicates against `new_factor.md` and `fail_new_factor.md`, generates HDF5 data, parses results, and registers new factors.
- **`/gen_tushare_h5`** — generates HDF5 data files for a given index (CSI300/CSI1000/CSI500) from `tushare/extra_data` CSVs. Used to produce `daily_pv_{index}.h5` for factor mining and AlphaExtra Stage0.

Permissions are managed in `.claude/settings.json` (skill enablement) and `.claude/settings.local.json` (allow-listed Bash commands and MCP tools).

## CI / GitHub Actions (upstream Qlib)

The upstream Qlib's CI runs on push/PR to `main` across a matrix of OS (Windows, Ubuntu 22.04/24.04, macOS 14/15) and Python (3.8–3.12):

- **`test_qlib_from_source.yml`** — full lint (black, pylint, flake8, mypy, nbqa), docs-gen, data download, nbconvert, and pytest (excluding `--slow`). Uses `make dev` for setup, data from `scripts/get_data.py`.
- **`lint_title.yml`** — validates PR titles against conventional-commit format via commitlint.
- **`release.yml`** / **`stale.yml`** — release automation and stale-issue management.

Our custom pipelines (TuShare, Docker-based) are NOT covered by these workflows — they run locally or via skills.

## `patch_and_run.py` — combined entry point

`patch_and_run.py` at the repo root is the primary entry point for fin_factor with all patches applied. It:
1. Calls `scripts/patch_rdagent_python_paths.py` — fixes imports inside container
2. Calls `scripts/patch_rdagent_bigmodel_direct.py` — bypasses LiteLLM routing
3. Calls `scripts/patch_rdagent_llm_fallback.py` — adds retry/fallback logic
4. Runs `rdagent fin_factor` via `scripts/run_fin_factor_with_cap.py`

Usage: `python patch_and_run.py` (or via Docker, where sitecustomize.py provides equivalent patches).

## Secondary data pipeline: myquant

`scripts/practice/` contains a myquant data pipeline variant alongside the primary TuShare pipeline:

- `scripts/practice/download_myquant_data.py` — downloads daily data from myquant API
- `scripts/practice/run_myquant_download.sh` — shell wrapper for the download
- `scripts/practice/gm_client.py` / `scripts/practice/tushare_client.py` — API clients for myquant/GM and TuShare respectively

This pipeline feeds into the same qlib binary format and is used as an alternative data source for the stage scripts.

## Environment variables

### `.env` file (for fin_factor Docker runs)

```
CHAT_MODEL=glm-4.7-flash                    # LiteLLM model name
OPENAI_API_KEY=sk-...                       # API key (passed to container)
OPENAI_API_BASE=http://host:port/v1         # API base URL (or DeepSeek proxy)
```

The `/factor-mining` skill reads `DEEPSEEK_API_KEY` from the host environment and passes it to the container as `OPENAI_API_KEY`. This keeps keys out of `.env` and git history.

When using DeepSeek, a local forward proxy is needed because DeepSeek has no embedding endpoint (stubbed in `sitecustomize.py`) and may have network restrictions inside Docker. See the `/factor-mining` skill for proxy setup.

### Runtime env vars

| Variable | Purpose | Default |
|----------|---------|---------|
| `CHAT_MODEL` | LiteLLM model string | `openai/glm-4.7` |
| `OPENAI_API_KEY` | API key (fin_factor) | required |
| `DEEPSEEK_API_KEY` | DeepSeek API key (used by `/factor-mining` skill; passed to container as `OPENAI_API_KEY`) | required |
| `OPENAI_API_BASE` | API base URL | required |
| `RDAGENT_MAX_ROUNDS` | Max rdagent loops | `20` |
| `RDAGENT_RETRY_WAIT_SECONDS` | LLM retry interval | `15` |
| `FORCE_LOCAL_STUB` | Stub LLM for testing | unset |
| `DOCKER_IMAGE` | Override Docker image | `qlib-rdagent:v1` |
| `QLIB_HOST_DATA_DIR` | Host qlib data root | `$HOME/.qlib` |
| `H5_FILE` | Override HDF5 source for AlphaExtra/factor-mining (e.g. `daily_pv_debug.h5` for debug subset) | `daily_pv_all.h5` |
| `FULL_BACKTEST_REPLAY_YEARS` | Limit backtest replay window to N years (0 = full history) | `0` |
| `WALK_FORWARD_START_DATE` | Backtest start date (format: `YYYY-MM-DD`) | required |
| `WALK_FORWARD_HISTORY_YEARS` | Years of history per fold for training | required |
| `WALK_FORWARD_SEGMENT_YEARS` | Walk-forward segment length in years | `1` |

## Key scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/practice/run_stage2_walk_forward.py` | Main walk-forward training (130KB, very large) |
| `scripts/practice/run_stage2_walk_forward_extra.py` | AlphaExtra variant — thin wrapper that overrides `MODEL_SPECS` |
| `scripts/practice/stage1_data_health_extra.py` | AlphaExtra data health check + missing-value filtering |
| `scripts/generate_extra_daily_pv.py` | Generate HDF5 files from cn_extra_data for rdagent (standalone version) |
| `rdagent_workspace/factor_data_template/generate.py` | Generate HDF5 files from cn_extra_data for rdagent (template version, used by fin_factor) |
| `scripts/run_fin_factor_with_cap.py` | Entry point for fin_factor; calls `rdagent.scenarios.qlib.developer.factor_runner.develop()` with `max_rounds` cap |
| `scripts/practice/gen_practice_yaml.py` | Generate workflow YAML from template |
| `scripts/data_collector/pit/collector.py` | PIT fundamental data download + normalization (baostock) |
| `scripts/dump_pit.py` | Dump normalized PIT CSV data into qlib binary format |
| `tushare/explore_extra_data.py` | Explore/convert TuShare extra data to qlib format |
| `tushare/get_tushare_data.py` | TuShare API data fetching — downloads daily/quarterly/annual CSV per stock |
| `tushare/api_utils.py` | TuShare API utility classes (TushareAPI, RateLimiter, DistributedRateLimiter, symbol_to_ts_code) |
| `tushare/fetch_sw_industry.py` | Fetch Shenwan (申万) level-1 industry classification via TuShare index_classify/index_member, writes `tushare/cn_data/sw_industry.csv` |
| `tushare/check_health.py` | Per-stock data health validation and auto-completion via TuShare API |
| `tushare/update_data.py` | Incremental data update script |
| `scripts/practice/build_features_from_h5.py` | AlphaExtra Stage0: builds qlib binary features from HDF5 (Alpha158 + new_factor.md factors + cross-sectional normalization) |
| `scripts/generate_h5.py` | Generate HDF5 data files for a given index (CSI300/CSI1000/CSI500) from extra_data CSVs |
| `tushare/_gen_benchmark.py` | Generate benchmark index data |
| `sitecustomize.py` | Monkey-patches rdagent at startup (loaded via PYTHONPATH): skips redundant downloads, reuses HDF5 files, stubs embeddings, injects FORCE_LOCAL_STUB |
| `scripts/patch_rdagent_bigmodel_direct.py` | Patch rdagent to bypass LiteLLM routing and call the model API directly |
| `scripts/patch_rdagent_llm_fallback.py` | Patch rdagent with LLM fallback/retry logic |
| `scripts/patch_rdagent_python_paths.py` | Fix Python path imports for rdagent in container |
| `patch_and_run.py` | Combined entry point: applies patches then runs fin_factor |
| `build_docker_image.sh` | Build the `qlib-rdagent:latest` Docker image locally |

## RDAgent internals (in container)

- `FACTOR_COSTEER_SETTINGS.data_folder` defaults to `"git_ignore_folder/factor_implementation_source_data"` — this is where `daily_pv_all.h5` and `daily_pv_debug.h5` must live.
- `generate_data_folder_from_qlib()` in `rdagent/scenarios/qlib/experiment/utils.py` runs generate.py and copies outputs.
- `get_data_folder_intro()` reads HDF5 columns and generates markdown descriptions for LLM prompts.
- The original generate.py reads from `cn_data` with 6 fields only; our `sitecustomize.py` patches this to support cn_extra_data's 58 fields.
- Model config via env: `CHAT_MODEL`, `OPENAI_API_KEY`, `OPENAI_API_BASE` (LiteLLM format, e.g. `openai/glm-4.7`, `deepseek/deepseek-chat`).

## Known issues

- **Symlinks in Docker**: `~/.qlib/qlib_data/cn_extra_data` symlinks to host path; Docker must mount that exact host path.
- **OOM with full dataset**: Reading all 5413 instruments × 58 features at once kills the process (exit 137). Batch or use debug subset. The template `generate.py` uses `GENERATE_BATCH_SIZE` (default 400) to batch-process.
- **`D.instruments()` pitfall**: Returns dict, not list. Use `D.list_instruments()` for actual codes.
- **Config access**: `C.get_data_path()` and `C["data_path"]` don't work in this qlib version; read provider_uri from init logs.
- **Cross-platform Cython**: `.so` files compiled on macOS won't work in Linux Docker. Compile inside the container when needed.
- **DeepSeek embedding**: DeepSeek has no `/v1/embeddings` endpoint. `sitecustomize.py` already stubs `create_embedding` to return zero vectors.
