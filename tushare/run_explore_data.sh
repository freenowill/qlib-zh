#!/usr/bin/env bash
# run_explore_data.sh - 构建 cn_extra_data 全量数据集 (用于 AlphaExtra stage2 训练)
#
# 功能: 将 test_tushare.py、check_health.py、explore_extra_data.py 串联起来，
#       对所有股票执行 数据拉取 → 健康检查 → qlib格式转换 的完整流程。
#
# 输出: cn_extra_data/ 目录，包含 58 个特征 (10 行情 + 15 估值 + 18 财务指标 + 15 财报)
#       供 AlphaExtra handler (stage2 训练) 使用
#
# 用法:
#   ./run_explore_data.sh                          # 处理 cn_data/instruments/all.txt 中所有股票
#   ./run_explore_data.sh SZ000001 SH600000        # 只处理指定股票
#   ./run_explore_data.sh --resume                 # 跳过已成功处理的股票
#   ./run_explore_data.sh --resume --max-fail 20   # 允许最多20只失败后继续
#   ./run_explore_data.sh -j 8 --resume            # 8路并发 (默认)
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
JOBS=8

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
  --resume            跳过 extra_data/ 中已有数据的股票
  --max-fail N        允许连续失败 N 只后终止 (默认 10，0=不限)
  --instruments FILE  使用指定的股票列表文件 (默认 cn_data/instruments/all.txt)
  -h, --help          显示帮助

示例:
  ./run_explore_data.sh                           # 全量处理 (8路并发)
  ./run_explore_data.sh -j 4 --resume             # 4路并发，断点续跑
  ./run_explore_data.sh --resume                  # 断点续跑
  ./run_explore_data.sh SZ000001 SH600000         # 只跑指定股票
  ./run_explore_data.sh --max-fail 0              # 永不停止
EOF
    exit 0
}

# 解析参数
SYMBOLS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)     JOBS="$2"; shift 2 ;;
        --resume)      RESUME=true; shift ;;
        --max-fail)    MAX_FAIL="$2"; shift 2 ;;
        --instruments) INSTRUMENTS_FILE="$2"; shift 2 ;;
        -h|--help)     usage ;;
        -*)            err "未知选项: $1"; usage ;;
        *)             SYMBOLS+=("$1"); shift ;;
    esac
done

# 加载股票列表 (排除指数代码，保留个股)
load_symbols() {
    if [[ ${#SYMBOLS[@]} -gt 0 ]]; then
        echo "${SYMBOLS[@]}"
        return
    fi
    if [[ ! -f "$INSTRUMENTS_FILE" ]]; then
        err "股票列表不存在: $INSTRUMENTS_FILE"
        exit 1
    fi
    # 保留: SH6xxxxx (沪市个股), SZ非399xxx (深市个股)
    # 排除: SH000xxx (上证指数), SZ399xxx (深证指数), BJ (北交所无日线数据)
    awk -F'\t' '!/^#/ && NF >= 1 {
        code = substr($1, 3, 3)
        if ($1 ~ /^SH/ && code ~ /^6/) print $1
        else if ($1 ~ /^SZ/ && code !~ /^399/) print $1
    }' "$INSTRUMENTS_FILE"
}

ALL_SYMBOLS=($(load_symbols))
TOTAL=${#ALL_SYMBOLS[@]}
log "股票总数: ${TOTAL}, 并发数: ${JOBS}"

if [[ "$RESUME" == true ]]; then
    mkdir -p "$STAMP_DIR"
fi
mkdir -p "$LOG_DIR"

# ============================================================
# 构建全局日历 (从已有 daily.csv 收集所有交易日期)
# ============================================================
GLOBAL_CALENDAR="${SCRIPT_DIR}/cn_extra_data/calendars/day.txt"
build_global_calendar() {
    log "构建全局交易日历..."
    mkdir -p "${SCRIPT_DIR}/cn_extra_data/calendars"
    local tmp_cal
    tmp_cal=$(mktemp)
    for csv in "${EXTRA_DATA_DIR}"/*/daily.csv; do
        [[ -f "$csv" ]] || continue
        awk -F',' 'NR>1{
            d=$2
            print substr(d,1,4)"-"substr(d,5,2)"-"substr(d,7,2)
        }' "$csv"
    done | sort -u > "$tmp_cal"
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

# 标记成功 (原子操作，线程安全)
mark_done() {
    [[ "$RESUME" == true ]] && touch "${STAMP_DIR}/${1}.done"
}

is_done() {
    [[ "$RESUME" == true && -f "${STAMP_DIR}/${1}.done" ]]
}

# 单只股票处理 (在子 shell 中运行，输出写入日志文件)
process_one() {
    local symbol="$1"
    local idx="$2"
    local log_file="${LOG_DIR}/${symbol}.log"

    if is_done "$symbol"; then
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

    # Step 3: 转换为 qlib bin (使用全局日历对齐)
    echo "[$(date '+%H:%M:%S')] [${idx}/${TOTAL}] ${symbol} Step3: qlib格式转换" >> "$log_file"
    local calendar_arg=""
    [[ -f "$GLOBAL_CALENDAR" ]] && calendar_arg="--calendar /workspace/cn_extra_data/calendars/day.txt"
    if ! docker run --rm \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        python3 explore_extra_data.py "$symbol" $calendar_arg >> "$log_file" 2>&1; then
        echo "[$(date '+%H:%M:%S')] [ERROR] ${symbol} Step3 失败: 格式转换" >> "$log_file"
        echo "FAIL" > "${log_file}.status"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] [OK] ${symbol} 全部完成" >> "$log_file"
    echo "OK" > "${log_file}.status"
    mark_done "$symbol"
    return 0
}

# ============================================================
# 并发执行主循环
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
    # 轮询检查已完成的任务
    while true; do
        for f in "$RUNNING_DIR"/*.pid; do
            [[ -f "$f" ]] || continue
            local pid
            pid=$(cat "$f")
            if ! kill -0 "$pid" 2>/dev/null; then
                local symbol
                symbol=$(basename "$f" .pid)
                rm -f "$f"

                # 读取状态
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
        # 没有完成的任务，短暂等待后重试
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

    # 每提交 JOBS 个任务打印一次进度
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
# 重建 qlib 索引 (calendar + instruments)
# ============================================================
rebuild_index() {
    log "重建 qlib 索引..."

    local extra_dir="${SCRIPT_DIR}/extra_data"
    local qlib_dir="${SCRIPT_DIR}/cn_extra_data"
    local cal_file="${qlib_dir}/calendars/day.txt"
    local inst_file="${qlib_dir}/instruments/all.txt"

    mkdir -p "${qlib_dir}/calendars" "${qlib_dir}/instruments"

    # 1. 从所有 daily.csv 收集全部交易日期 (刷新全局日历)
    log "  刷新交易日历..."
    local all_dates_file
    all_dates_file=$(mktemp)
    for csv in "${extra_dir}"/*/daily.csv; do
        [[ -f "$csv" ]] || continue
        awk -F',' 'NR>1{
            d=$2
            print substr(d,1,4)"-"substr(d,5,2)"-"substr(d,7,2)
        }' "$csv"
    done | sort -u > "$all_dates_file"

    local date_count
    date_count=$(wc -l < "$all_dates_file" | tr -d ' ')
    if [[ "$date_count" -eq 0 ]]; then
        err "未找到任何交易日期"
        rm -f "$all_dates_file"
        return 1
    fi
    mv "$all_dates_file" "$cal_file"
    log "  日历: ${date_count} 天 ($(head -1 "$cal_file") ~ $(tail -1 "$cal_file"))"

    # 2. 从所有 daily.csv 构建 instruments (每只股票的实际日期范围)
    log "  构建股票列表..."
    local tmp_inst
    tmp_inst=$(mktemp)
    local stock_count=0

    for csv in "${extra_dir}"/*/daily.csv; do
        [[ -f "$csv" ]] || continue
        local sym
        sym=$(basename "$(dirname "$csv")")
        # 从 daily.csv 获取日期范围
        local start_date end_date
        start_date=$(awk -F',' 'NR>1{print $2}' "$csv" | sort | head -1)
        end_date=$(awk -F',' 'NR>1{print $2}' "$csv" | sort | tail -1)
        if [[ -n "$start_date" && -n "$end_date" ]]; then
            # 转为 YYYY-MM-DD 格式
            start_date="${start_date:0:4}-${start_date:4:2}-${start_date:6:2}"
            end_date="${end_date:0:4}-${end_date:4:2}-${end_date:6:2}"
            echo -e "${sym}\t${start_date}\t${end_date}"
            stock_count=$((stock_count + 1))
        fi
    done | sort > "$tmp_inst"

    mv "$tmp_inst" "$inst_file"
    log "  股票列表: ${stock_count} 只"

    # 3. 链接到 qlib 标准路径
    local qlib_data_dir="${HOME}/.qlib/qlib_data/cn_extra_data"
    if [[ ! -e "$qlib_data_dir" ]]; then
        mkdir -p "$(dirname "$qlib_data_dir")"
        ln -sf "$qlib_dir" "$qlib_data_dir"
        log "  已链接: $qlib_dir -> $qlib_data_dir"
    fi

    ok "qlib 索引重建完成 (日历 ${date_count} 天, 股票 ${stock_count} 只)"
}

rebuild_index

if [[ -d "${SCRIPT_DIR}/cn_extra_data/features" ]]; then
    FEATURE_COUNT=$(ls -d "${SCRIPT_DIR}/cn_extra_data/features"/*/ 2>/dev/null | wc -l | tr -d ' ')
    log "cn_extra_data 已有 ${FEATURE_COUNT} 只股票的特征数据"
    if [[ "$FEATURE_COUNT" -lt "$TOTAL" ]]; then
        REMAINING=$((TOTAL - FEATURE_COUNT))
        warn "距离 AlphaExtra 全量训练还差 ${REMAINING} 只股票"
    else
        ok "cn_extra_data 已覆盖全部 ${TOTAL} 只股票，可用于 AlphaExtra stage2 训练"
    fi
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
