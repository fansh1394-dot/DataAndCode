#!/usr/bin/env bash
# E10：输入分辨率 640 -> 800（唯一变量）
#
# 实验设计
# --------
#   基线 : mdp150_M0_concat_s0/s1/s2  imgsz=640  batch=32  μ=0.7551 σ=0.0094
#   实验 : mdp150_M0Img800_s0          imgsz=800  batch=8（OOM 兜底降到 6 / 4）
#
# 为什么 batch 只有 8：2026-11-03 起该卡上有 15128 MiB 无法回收的幽灵显存
# （nvidia-smi 里查不到持有进程、Util=0%，无 sudo 无法 reset），
# 实际可用仅 9.4 GB，800 分辨率下 32/16 都放不下。
# → E10 的混杂变量是「imgsz 640->800 且 batch 32->8」，结果显著时须在干净卡上以 b32 复核。
#
# 判定
#   > 0.7739（+2σ）-> 分辨率是真杠杆，可作为第 5 章正向结论
#   0.7363~0.7739    -> 不显著，强化「架构/损失侧已到边界，只能靠数据与输入尺度」
#   < 0.7363         -> 显著负（放大插值伪影），同样是有价值的负结论
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0Img800_s0
LOG=logs/e10_img800.log
NEED=${NEED:-8200}
MAX_TRY=${MAX_TRY:-40}
WARM_WAIT=${WARM_WAIT:-360}
SLOW_IT=${SLOW_IT:-9}
BATCH=${BATCH:-8}

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

log "=== E10 分辨率实验启动（${NAME}  imgsz=800  batch=${BATCH}）==="

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
    log "第 ${attempt}/${MAX_TRY} 次：起跑 ${NAME}  imgsz=800  batch=${BATCH}  空闲=$(free_mib) MiB"

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
        --imgsz 800 \
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
        log "${NAME} 完成 exit=0（imgsz=800 batch=${BATCH}）"
        exit 0
    fi
    if grep -aq "CUDA out of memory" "$traylog"; then
        if [ "$BATCH" -gt 8 ]; then
            BATCH=8
        elif [ "$BATCH" -gt 6 ]; then
            BATCH=6
        elif [ "$BATCH" -gt 4 ]; then
            BATCH=4
        else
            log "${NAME} OOM（batch=${BATCH} 已到下限），300s 后重试"
            sleep 300
            continue
        fi
        log "${NAME} OOM，batch 降到 ${BATCH} 重跑并留痕"
        echo "batch OOM -> ${BATCH}（发生于 $(date '+%F %T')）" >> "logs/${NAME}.batch_note"
        sleep 300
        continue
    fi
    log "${NAME} 异常退出 exit=${code}（非 OOM）；见 ${traylog}"
    exit "$code"
done

log "${NAME} 尝试次数用尽，放弃"
exit 1
