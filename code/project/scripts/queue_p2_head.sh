#!/usr/bin/env bash
# P2 检测头实验链（新实例 端口 30453）
#
# 实验设计
# --------
# 唯一变量：检测头输入由 P3-P5 改为 P2-P5（四条分支都加，受 nl 一致性约束）。
#   基线: mdp150_M0_concat_s0/s1/s2   μ=0.7551  σ=0.0094  2σ=0.0188
#   实验: mdp150_M0P2_s0
#
# 判定
#   超过 0.7739 -> 小目标是真瓶颈，P2 有效（可作为改进点写进第 5 章）
#   低于 0.7363 -> P2 显著有害，强化「瓶颈不在检测尺度」的结论
#   区间内       -> 无显著差异，同样支持「架构侧边际收益趋零」
#
# 显存说明
#   P2 特征图在 640 输入下为 160×160，比 P3 大 4 倍。
#   2026-11-01 实测（RTX 4090D 24GB，fp32 / amp=false，imgsz=640）：
#     batch 32 -> OOM；batch 24 -> PyTorch 已占 22.89 GiB 仍 OOM（需 >23.6 GiB）
#     batch 16 -> 峰值 17.9 GiB，稳定；288 it/epoch，约 1.86 it/s
#   nbs=64 时 accumulate = round(64/batch)：batch16 -> 累积 4 步 = 等价 64，
#   与基线（batch32 + 累积 2 步 = 64）一致；batch24 反而等价 72，可比性更差。
#   故默认 batch 16，并在 logs/mdp150_M0P2_s0.batch_note 里留痕。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat-p2.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0P2_s0
LOG=logs/p2_chain.log
NEED=${NEED:-20000}
MAX_TRY=${MAX_TRY:-40}
WARM_WAIT=${WARM_WAIT:-360}
SLOW_IT=${SLOW_IT:-5}
BATCH=${BATCH:-16}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

recent_it_s() {
    local f="$1"
    [ -f "$f" ] || { echo 0; return; }
    local v
    v=$(grep -aoE '[ <][0-9]+\.[0-9]+s/it' "$f" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    echo "${v:-0}"
}

wait_gpu() {
    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then break; fi
        log "等待显存：空闲 ${f:-NA} MiB < ${NEED} MiB"
        sleep 120
    done
}

[ -f "$CFG" ] || { log "致命：缺配置 $CFG"; exit 2; }

log "=== P2 检测头实验链启动（${NAME}）==="

attempt=0
while [ "$attempt" -lt "$MAX_TRY" ]; do
    attempt=$((attempt + 1))

    run_dir=$(ls -d YOLO26/ResTest/${NAME}* 2>/dev/null | head -1)
    if [ -n "$run_dir" ] && [ -f "$run_dir/results.csv" ]; then
        last_ep=$(tail -1 "$run_dir/results.csv" | cut -d, -f1 | tr -d ' ')
        if [ "${last_ep:-0}" -ge 149 ] 2>/dev/null; then
            log "${NAME} 已完成（末轮 ep${last_ep}），跳过"
            exit 0
        fi
        mv "$run_dir" "YOLO26/ResTest/_aborted/${NAME}_$(date +%H%M%S)" 2>/dev/null
        log "${NAME} 残留目录已挪至 _aborted"
    fi

    wait_gpu
    log "第 ${attempt}/${MAX_TRY} 次：起跑 ${NAME}  batch=${BATCH}  空闲=$(free_mib) MiB"

    traylog="logs/${NAME}.log"
    : > "$traylog"
    $PY -u train_v26.py \
        --config "$CFG" \
        --label-from dual \
        --data "$DATA" \
        --name "$NAME" \
        --seed 0 \
        --epochs 150 \
        --patience 30 \
        --batch "$BATCH" \
        --workers 4 \
        --amp false \
        >> "$traylog" 2>&1 &
    pid=$!

    sleep "$WARM_WAIT"
    if kill -0 "$pid" 2>/dev/null; then
        it=$(recent_it_s "$traylog")
        log "  速度检查：s/it=${it}"
        if awk "BEGIN{exit !(${it:-0} > ${SLOW_IT})}"; then
            log "  s/it=${it} > ${SLOW_IT}，判定被抢占，杀掉重来"
            kill -9 "$pid" 2>/dev/null
            pkill -9 -f "train_v26.py.*--name ${NAME}" 2>/dev/null
            sleep 30
            continue
        fi
    fi

    wait "$pid"
    code=$?
    if [ "$code" -eq 0 ]; then
        log "${NAME} 完成 exit=0（batch=${BATCH}）"
        exit 0
    fi
    if grep -aq "CUDA out of memory" "$traylog"; then
        if [ "$BATCH" -gt 16 ]; then
            log "${NAME} OOM（batch=${BATCH}），降到 16 重跑并留痕"
            echo "batch ${BATCH} OOM -> 16（等价 batch=64，nbs=64；发生于 $(date '+%F %T')）" >> "logs/${NAME}.batch_note"
            BATCH=16
            NEED=20000
            sleep 300
            continue
        fi
        log "${NAME} OOM（batch=${BATCH}），300s 后重试"
        sleep 300
        continue
    fi
    log "${NAME} 异常退出 exit=${code}（非 OOM）；见 ${traylog}"
    exit "$code"
done

log "${NAME} 尝试次数用尽，放弃"
exit 1
