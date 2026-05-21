#!/usr/bin/env bash
# run_explore_data.sh - 构建 cn_extra_data_improve 全量数据集
#
# 流程:
#   [Pre]  智能增量更新 — 已有数据做新鲜度检测+增量拉取; 无数据股票全量拉取
#   [Main] 并行处理      — test_tushare → check_health → process_extra_data (CSV→bin+因子)
#   [Post] 索引构建      — 日历 + instruments + 因子清单
#
# 因子定义: tushare/new_factor.md (30 个因子，含动量/反转/波动率/流动性/估值/量价/风险调整/质量/截面)
#
# 用法:
#   ./run_explore_data.sh                              # 全量 (含因子计算)
#   ./run_explore_data.sh --no-improve                 # 仅基础特征 (不含 new_factor.md 因子)
#   ./run_explore_data.sh --resume                     # 断点续跑
#   ./run_explore_data.sh -j 8 --resume                # 8路并发
#   ./run_explore_data.sh SZ000001 SH600000             # 指定股票
#
# 环境: 使用 zhuhai123/local_qlib:v1-tushare Docker 镜像

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_IMAGE="zhuhai123/local_qlib:v1-tushare"
EXTRA_DATA_DIR="${SCRIPT_DIR}/extra_data"
INSTRUMENTS_FILE="${SCRIPT_DIR}/cn_data/instruments/all.txt"
STAMP_DIR="${SCRIPT_DIR}/extra_data/.done"
LOG_DIR="${SCRIPT_DIR}/extra_data/.logs"
MAX_FAIL=10
RESUME=false
JOBS=20
IMPROVE=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    cat <<'EOF'
用法: ./run_explore_data.sh [选项] [股票代码...]

选项:
  -j, --jobs N        并发数 (默认 8)
  --resume            跳过 extra_data/.done 中已完成的股票
  --no-improve        不生成 cn_extra_data_improve
  --market INDEX      限定指数成分股 (csi300/csi500/csi800/csi1000/all, 默认 all)
  --max-fail N        允许连续失败 N 只后终止 (默认 10，0=不限)
  --instruments FILE  使用指定的股票列表文件 (默认 cn_data/instruments/all.txt)
  -h, --help          显示帮助

示例:
  ./run_explore_data.sh                           # 全量 + improve (8路并发)
  ./run_explore_data.sh --market csi800 -j 4      # 仅 CSI800 成分股
  ./run_explore_data.sh -j 4 --resume             # 4路并发，断点续跑
  ./run_explore_data.sh --no-improve              # 仅 cn_extra_data
EOF
    exit 0
}

# 解析参数
SYMBOLS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        --resume)        RESUME=true; shift ;;
        --no-improve)    IMPROVE=false; shift ;;
        --max-fail)      MAX_FAIL="$2"; shift 2 ;;
        --market)        TARGET_MARKET="$2"; shift 2 ;;
        --instruments)   INSTRUMENTS_FILE="$2"; shift 2 ;;
        -h|--help)       usage ;;
        -*)              err "未知选项: $1"; usage ;;
        *)               SYMBOLS+=("$1"); shift ;;
    esac
done

# 加载股票列表 (排除指数代码，保留个股，支持指数成分股过滤)
load_symbols() {
    if [[ ${#SYMBOLS[@]} -gt 0 ]]; then
        echo "${SYMBOLS[@]}"
        return
    fi

    local market="${TARGET_MARKET:-all}"

    # 如果指定了指数，从对应 csi*.txt 加载
    if [[ "$market" != "all" ]]; then
        local idx_file="${SCRIPT_DIR}/cn_data/instruments/${market}.txt"
        if [[ -f "$idx_file" ]]; then
            awk -F'\t' '!/^#/ && NF >= 1 {print $1}' "$idx_file" | sort -u
            return
        fi
        warn "指数文件不存在: $idx_file，回退到全量"
    fi

    if [[ ! -f "$INSTRUMENTS_FILE" ]]; then
        err "股票列表不存在: $INSTRUMENTS_FILE"
        exit 1
    fi
    # SH6xxxxx (沪市个股), SZ非399xxx (深市个股)
    awk -F'\t' '!/^#/ && NF >= 1 {
        code = substr($1, 3, 3)
        if ($1 ~ /^SH/ && code ~ /^6/) print $1
        else if ($1 ~ /^SZ/ && code !~ /^399/) print $1
    }' "$INSTRUMENTS_FILE"
}

ALL_SYMBOLS=($(load_symbols))

# --resume: 仅处理已有 daily.csv 的股票，跳过从未拉取成功的
if [[ "$RESUME" == true ]]; then
    mkdir -p "$STAMP_DIR"
    filtered=()
    for sym in "${ALL_SYMBOLS[@]}"; do
        if [[ -f "${EXTRA_DATA_DIR}/${sym}/daily.csv" ]]; then
            filtered+=("$sym")
        fi
    done
    log "resume 过滤: ${#ALL_SYMBOLS[@]} → ${#filtered[@]} 只 (已有数据)"
    ALL_SYMBOLS=("${filtered[@]}")
fi

TOTAL=${#ALL_SYMBOLS[@]}
log "股票总数: ${TOTAL}, 并发数: ${JOBS}, improve: ${IMPROVE}"
mkdir -p "$LOG_DIR"

# ============================================================
# 全局日历
# ============================================================
GLOBAL_CALENDAR="${SCRIPT_DIR}/cn_extra_data_improve/calendars/day.txt"
build_global_calendar() {
    log "构建全局交易日历..."
    mkdir -p "${SCRIPT_DIR}/cn_extra_data_improve/calendars"
    local tmp_cal
    tmp_cal=$(mktemp)
    while IFS= read -r -d '' csv; do
        awk -F',' 'NR>1{
            d=$2
            print substr(d,1,4)"-"substr(d,5,2)"-"substr(d,7,2)
        }' "$csv"
    done < <(find "${EXTRA_DATA_DIR}" -maxdepth 2 -name "daily.csv" -type f -print0) | sort -u > "$tmp_cal"
    local count
    count=$(wc -l < "$tmp_cal" | tr -d ' ')
    if [[ "$count" -gt 0 ]]; then
        mv "$tmp_cal" "$GLOBAL_CALENDAR"
        log "全局日历: ${count} 天 ($(head -1 "$GLOBAL_CALENDAR") ~ $(tail -1 "$GLOBAL_CALENDAR"))"
    else
        rm -f "$tmp_cal"
        warn "无已有 daily.csv，日历将在处理后构建"
    fi
}
build_global_calendar

# ============================================================
# [Pre] 智能增量更新
#   1. 对已有 daily.csv 的股票，检测新鲜度并增量拉取
#   2. 对没有 daily.csv 的股票，后续由 test_tushare.py 全量拉取
# ============================================================
smart_incremental_update() {
    log "========== [Pre] 智能增量更新 =========="

    local existing_stocks=()
    local new_stocks=()

    for sym in "${ALL_SYMBOLS[@]}"; do
        if [[ -f "${EXTRA_DATA_DIR}/${sym}/daily.csv" ]]; then
            existing_stocks+=("$sym")
        else
            new_stocks+=("$sym")
        fi
    done

    log "已有数据: ${#existing_stocks[@]} 只, 待拉取: ${#new_stocks[@]} 只"

    if [[ ${#existing_stocks[@]} -eq 0 ]]; then
        log "无已有数据，跳过增量检测"
        return
    fi

    # 对已有数据的股票运行新鲜度检测 + 增量拉取 (csv-only 模式)
    # process_extra_data.py --csv-only: 检测新鲜度 → 增量拉取 → 仅更新 CSV
    log "检测已有数据新鲜度并增量拉取..."
    docker run --rm \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        python3 process_extra_data.py --csv-only --symbols "${existing_stocks[@]}" 2>&1 | \
        while IFS= read -r line; do
            if echo "$line" | grep -qE '(INFO|WARNING|ERROR)'; then
                echo "  $line"
            fi
        done

    local exit_code=${PIPESTATUS[0]}
    if [[ "$exit_code" -ne 0 ]]; then
        warn "增量更新部分失败 (exit=$exit_code)，将继续处理"
    else
        ok "增量更新完成"
    fi
}

# --resume 时跳过增量检测 (数据已在首次运行时处理完毕)
if [[ "$RESUME" == true ]]; then
    log "--resume: 跳过增量检测，直接进入并行处理"
else
    smart_incremental_update
fi

# ============================================================
# 标记成功 (原子操作，线程安全)
# ============================================================
mark_done() {
    [[ "$RESUME" == true ]] && touch "${STAMP_DIR}/${1}.done"
}

is_done() {
    [[ "$RESUME" == true && -f "${STAMP_DIR}/${1}.done" ]]
}

# ============================================================
# 单只股票处理 (在子 shell 中运行，输出写入日志文件)
# ============================================================
process_one() {
    local symbol="$1"
    local idx="$2"
    local log_file="${LOG_DIR}/${symbol}.log"

    # --resume: 已完成股票跳过，或仅补跑 improve
    if is_done "$symbol"; then
        if [[ "$IMPROVE" == true ]]; then
            local sym_lower; sym_lower=$(echo "$symbol" | tr '[:upper:]' '[:lower:]')
            local improve_feat_dir="${SCRIPT_DIR}/cn_extra_data_improve/features/${sym_lower}"
            if [[ -d "$improve_feat_dir" ]] && ls "$improve_feat_dir"/*.bin >/dev/null 2>&1; then
                echo "SKIP" > "${log_file}.status"
                return 0
            fi
            # 有 .done 但缺 improve → 仅跑 improve
            echo "[$(date '+%H:%M:%S')] [${idx}/${TOTAL}] ${symbol} improve-only: CSV→bin+因子" > "$log_file"
            if docker run --rm \
                -v "${SCRIPT_DIR}:/workspace" \
                -w /workspace \
                "$DOCKER_IMAGE" \
                python3 process_extra_data.py --mode improve-stock --symbols "$symbol" >> "$log_file" 2>&1; then
                echo "[$(date '+%H:%M:%S')] [OK] ${symbol} improve 完成" >> "$log_file"
                echo "OK" > "${log_file}.status"
            else
                echo "[$(date '+%H:%M:%S')] [WARN] ${symbol} improve 失败" >> "$log_file"
                echo "OK" > "${log_file}.status"
            fi
            return 0
        fi
        echo "SKIP" > "${log_file}.status"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] [${idx}/${TOTAL}] ========== ${symbol} ==========" > "$log_file"

    # Step 1: 拉取数据
    echo "[$(date '+%H:%M:%S')] [${idx}/${TOTAL}] ${symbol} Step1: 拉取数据" >> "$log_file"
    if ! docker run --rm \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        python3 test_tushare.py "$symbol" >> "$log_file" 2>&1; then
        echo "[$(date '+%H:%M:%S')] [ERROR] ${symbol} Step1 失败: 数据拉取" >> "$log_file"
        echo "FAIL" > "${log_file}.status"
        return 1
    fi

    # Step 2: 健康检查 (不阻塞流程)
    echo "[$(date '+%H:%M:%S')] [${idx}/${TOTAL}] ${symbol} Step2: 健康检查" >> "$log_file"
    docker run --rm \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        python3 check_health.py "$symbol" >> "$log_file" 2>&1 || true

    # Step 3: CSV → bin + new_factor.md 因子计算 → cn_extra_data_improve
    if [[ "$IMPROVE" == true ]]; then
        echo "[$(date '+%H:%M:%S')] [${idx}/${TOTAL}] ${symbol} Step3: CSV→bin+因子" >> "$log_file"
        if ! docker run --rm \
            -v "${SCRIPT_DIR}:/workspace" \
            -w /workspace \
            "$DOCKER_IMAGE" \
            python3 process_extra_data.py --mode improve-stock --symbols "$symbol" >> "$log_file" 2>&1; then
            echo "[$(date '+%H:%M:%S')] [ERROR] ${symbol} Step3 失败: CSV→bin+因子" >> "$log_file"
            echo "FAIL" > "${log_file}.status"
            return 1
        fi
    fi

    echo "[$(date '+%H:%M:%S')] [OK] ${symbol} 全部完成" >> "$log_file"
    echo "OK" > "${log_file}.status"
    mark_done "$symbol"
    return 0
}

# ============================================================
# [Main] 并发执行主循环
# ============================================================
START_TIME=$(date +%s)
SUCCESS_COUNT=0
SKIP_COUNT=0
FAILED_COUNT=0
consecutive_fail=0
STOP=false

# 用文件跟踪并发任务 (兼容 bash 3.x)
RUNNING_DIR=$(mktemp -d)
trap "rm -rf '$RUNNING_DIR'" EXIT

running_count() {
    ls "$RUNNING_DIR" 2>/dev/null | wc -l | tr -d ' '
}

# 等待一个任务完成，回收结果
wait_one() {
    while true; do
        for f in "$RUNNING_DIR"/*.pid; do
            [[ -f "$f" ]] || continue
            local pid
            pid=$(cat "$f")
            if ! kill -0 "$pid" 2>/dev/null; then
                local symbol
                symbol=$(basename "$f" .pid)
                rm -f "$f"

                local status_file="${LOG_DIR}/${symbol}.log.status"
                local status="UNKNOWN"
                [[ -f "$status_file" ]] && status=$(cat "$status_file")

                if [[ "$status" == "OK" || "$status" == "SKIP" ]]; then
                    if [[ "$status" == "SKIP" ]]; then
                        SKIP_COUNT=$((SKIP_COUNT + 1))
                        log "[跳过] ${symbol} (已完成)"
                    else
                        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                        ok "${symbol} 完成 (成功:${SUCCESS_COUNT} 跳过:${SKIP_COUNT} 失败:${FAILED_COUNT})"
                    fi
                    consecutive_fail=0
                else
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                    consecutive_fail=$((consecutive_fail + 1))
                    err "${symbol} 失败 (成功:${SUCCESS_COUNT} 跳过:${SKIP_COUNT} 失败:${FAILED_COUNT})"
                    tail -3 "${LOG_DIR}/${symbol}.log" 2>/dev/null | while IFS= read -r line; do
                        echo "  $line"
                    done

                    if [[ "$MAX_FAIL" -gt 0 && "$consecutive_fail" -ge "$MAX_FAIL" ]]; then
                        err "连续失败 ${consecutive_fail} 只，停止提交新任务"
                        STOP=true
                    fi
                fi
                return 0
            fi
        done
        sleep 0.5
    done
}

# 主调度循环
submitted=0
for i in "${!ALL_SYMBOLS[@]}"; do
    [[ "$STOP" == true ]] && break

    symbol="${ALL_SYMBOLS[$i]}"
    idx=$((i + 1))

    # 如果已达到并发上限，等待一个任务完成
    while [[ $(running_count) -ge $JOBS ]]; do
        wait_one
    done

    # 提交后台任务
    process_one "$symbol" "$idx" &
    echo $! > "${RUNNING_DIR}/${symbol}.pid"
    submitted=$((submitted + 1))

    if (( submitted % JOBS == 0 )); then
        log "已提交 ${submitted}/${TOTAL}, 运行中 $(running_count), 成功 ${SUCCESS_COUNT}, 失败 ${FAILED_COUNT}"
    fi
done

# 等待所有剩余任务完成
while [[ $(running_count) -gt 0 ]]; do
    wait_one
done

# ============================================================
# 汇总
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "  运行汇总"
echo "=============================================="
echo "  总股票数:   ${TOTAL}"
echo "  并发数:     ${JOBS}"
echo "  成功:       ${SUCCESS_COUNT}"
echo "  跳过:       ${SKIP_COUNT}"
echo "  失败:       ${FAILED_COUNT}"
echo "  耗时:       ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
echo "=============================================="
echo ""

# ============================================================
# [Post] cn_extra_data_improve 索引构建 (特征已在并行阶段写入)
# ============================================================
build_improve_index() {
    log "========== [Post] cn_extra_data_improve 索引 =========="

    local improve_out="${SCRIPT_DIR}/cn_extra_data_improve"
    local qlib_improve="${HOME}/.qlib/qlib_data/cn_extra_data_improve"

    mkdir -p "${improve_out}/calendars" "${improve_out}/instruments"

    # 1. 日历 (已由 build_global_calendar 写入 cn_extra_data_improve/calendars/)
    local cal_file="${improve_out}/calendars/day.txt"
    if [[ -f "$cal_file" ]]; then
        local cal_count; cal_count=$(wc -l < "$cal_file" | tr -d ' ')
        log "  日历: ${cal_count} 天"
    else
        warn "  日历文件不存在，将由 process_extra_data.py 生成"
    fi

    # 2. 构建 instruments
    local inst_file="${improve_out}/instruments/all.txt"
    local tmp_inst=$(mktemp)
    local stock_count=0
    while IFS= read -r -d '' csv; do
        local sym start_d end_d
        sym=$(basename "$(dirname "$csv")")
        start_d=$(awk -F',' 'NR>1{print $2}' "$csv" | sort -n | head -1)
        end_d=$(awk -F',' 'NR>1{print $2}' "$csv" | sort -n | tail -1)
        if [[ -n "$start_d" && -n "$end_d" ]]; then
            printf '%s\t%s-%s-%s\t%s-%s-%s\n' \
                "$sym" \
                "${start_d:0:4}" "${start_d:4:2}" "${start_d:6:2}" \
                "${end_d:0:4}" "${end_d:4:2}" "${end_d:6:2}" >> "$tmp_inst"
            stock_count=$((stock_count + 1))
        fi
    done < <(find "${EXTRA_DATA_DIR}" -maxdepth 2 -name "daily.csv" -type f -print0 | sort -z)
    sort -o "$tmp_inst" "$tmp_inst"
    mv "$tmp_inst" "$inst_file"
    log "  股票列表: ${stock_count} 只"

    # 3. 复制指数成分股文件
    local cn_data_inst="${SCRIPT_DIR}/cn_data/instruments"
    if [[ -d "$cn_data_inst" ]]; then
        local idx_file
        for idx_file in "$cn_data_inst"/csi*.txt; do
            [[ -f "$idx_file" ]] || continue
            cp "$idx_file" "${improve_out}/instruments/"
        done
        log "  指数文件已复制"
    fi

    # 4. 链接到 qlib 标准路径
    if [[ ! -e "$qlib_improve" ]]; then
        mkdir -p "$(dirname "$qlib_improve")"
        ln -sf "$improve_out" "$qlib_improve"
        log "  已链接: $improve_out -> $qlib_improve"
    fi

    # 5. 数据质量摘要
    local n_stocks
    n_stocks=$(ls -d "${improve_out}/features"/*/ 2>/dev/null | wc -l | tr -d ' ')
    local n_bins=0
    local sample_dir
    sample_dir=$(ls -d "${improve_out}/features"/*/ 2>/dev/null | head -1)
    if [[ -n "$sample_dir" ]]; then
        n_bins=$(ls "${sample_dir}"/*.bin 2>/dev/null | wc -l | tr -d ' ')
    fi
    log "  数据质量: ${n_stocks} 只股票, ${n_bins} 特征/股"

    ok "cn_extra_data_improve 索引构建完成"
}

if [[ "$IMPROVE" == true ]]; then
    build_improve_index
fi

# ============================================================
# 最终状态
# ============================================================
if [[ "$IMPROVE" == true && -d "${SCRIPT_DIR}/cn_extra_data_improve/features" ]]; then
    IMPROVE_COUNT=$(ls -d "${SCRIPT_DIR}/cn_extra_data_improve/features"/*/ 2>/dev/null | wc -l | tr -d ' ')
    log "cn_extra_data_improve 已有 ${IMPROVE_COUNT} 只股票的特征数据 (基础特征 + new_factor.md 因子)"
fi

# 清理日志状态文件
rm -f "${LOG_DIR}"/*.status 2>/dev/null

if [[ "$FAILED_COUNT" -gt 0 ]]; then
    err "有 ${FAILED_COUNT} 只股票处理失败，可用 --resume 重新运行"
    err "查看失败详情: grep -l FAIL ${LOG_DIR}/*.log.status"
    exit 1
fi

ok "全部完成!"
exit 0
