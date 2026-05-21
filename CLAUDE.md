# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a Chinese-market quantitative investment platform built on **Qlib** (Microsoft's AI-oriented quant framework) integrated with **RDAgent** for automated factor mining. The codebase is a fork/customization of upstream Qlib with TuShare-based data pipelines, a 6-stage alpha research workflow, and Docker-based rdagent execution.

## Repo architecture (what's ours vs upstream)

| Layer | Location | Role |
|-------|----------|------|
| **Upstream Qlib** | `qlib/` | Core library: data layer, model backends, backtest engine, workflow runner. Forked from Microsoft qlib; do not refactor casually. |
| **Our scripts & pipelines** | `scripts/` | Data collection (PIT, yahoo), rdagent wrappers, stage runners, analysis tools. This is where most custom work lives. |
| **Our data pipelines** | `tushare/` | TuShare data fetching/processing → qlib binary format, extra features (58 fields → 109 after `process_extra_data.py`), health checks. |
| **Our Docker/rdagent infra** | root + `scripts/` | `Dockerfile`, `docker-compose.yml`, `sitecustomize.py`, `run_fin_factor_with_cap.py` — glue between qlib data and rdagent runtime. |
| **Workspace** | `rdagent_workspace/` | Runtime data for rdagent (HDF5 source data, templates). |
| **Outputs** | `DATA/`, `mlruns/` | Experiment results, MLflow tracking. |

## Local modifications to upstream Qlib

We patch a few Qlib files for bug fixes or feature additions. These are mounted into Docker containers at runtime:

- **`qlib/data/storage/file_storage.py`** — int-casting fix for `fp.seek()` on Linux to prevent TypeError from numpy int64 values
- **`qlib/data/ops.py`** — custom modifications (mounted into container)
- **`qlib/contrib/data/handler_extra.py`** — our custom AlphaExtra handler (109 features across 9 categories: value, quality, growth, leverage, size/liquidity, kbar, price, rolling, price-volume). Generates factors from `cn_extra_data_improve`.

## Two main workflows

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
bash run_new_factor_practice <experiment_name> [stage=N] [end_stage=N] [missing_threshold=N]
```

Uses `cn_extra_data_improve` (109 features), AlphaExtra handler (`qlib/contrib/data/handler_extra.py`), and custom YAML template (`workflow_config_lightgbm_AlphaExtra.yaml`). Supports both Stage1 (data health + filter) and Stage2 (walk-forward training). Runs inside Docker (`zhuhai123/qlib-rdagent:v1`).

If Stage1 generated filtered data, Stage2 auto-mounts the filtered dataset overlay while keeping the original as a symlink source. Filtered data lands at `~/.qlib/qlib_data/cn_extra_data_improve_filtered`.

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
| `tushare/cn_extra_data/` | Extended 58-feature dataset | qlib binary format (symlinked to `~/.qlib/qlib_data/cn_extra_data`) |
| `tushare/cn_extra_data_improve/` | Improved extra data with 109 features (58 raw + 51 price-volume computed by `process_extra_data.py`) | qlib binary format |
| `tushare/extra_data/` | Raw TuShare downloads (~5500 items) | Various formats |

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
- `qlib-rdagent:latest` — local base image built from `Dockerfile`

Build custom image: `bash build_docker_image.sh` or `docker compose build`.

## Environment variables

### `.env` file (for fin_factor Docker runs)

```
CHAT_MODEL=glm-4.7-flash                    # LiteLLM model name
OPENAI_API_KEY=sk-...                       # API key
OPENAI_API_BASE=http://host:port/v1         # API base URL (or DeepSeek proxy)
```

When using DeepSeek, a local forward proxy is needed because DeepSeek has no embedding endpoint (stubbed in `sitecustomize.py`) and may have network restrictions inside Docker. See the `/factor-mining` skill for proxy setup.

### Runtime env vars

| Variable | Purpose | Default |
|----------|---------|---------|
| `CHAT_MODEL` | LiteLLM model string | `openai/glm-4.7` |
| `OPENAI_API_KEY` | API key | required |
| `OPENAI_API_BASE` | API base URL | required |
| `RDAGENT_MAX_ROUNDS` | Max rdagent loops | `20` |
| `RDAGENT_RETRY_WAIT_SECONDS` | LLM retry interval | `15` |
| `FORCE_LOCAL_STUB` | Stub LLM for testing | unset |
| `DOCKER_IMAGE` | Override Docker image | `qlib-rdagent:v1` |
| `QLIB_HOST_DATA_DIR` | Host qlib data root | `$HOME/.qlib` |

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
| `tushare/process_extra_data.py` | Process raw extra data into `cn_extra_data_improve` (58→109 features) |
| `sitecustomize.py` | Monkey-patches rdagent at startup (loaded via PYTHONPATH): skips redundant downloads, reuses HDF5 files, stubs embeddings, injects FORCE_LOCAL_STUB |

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
