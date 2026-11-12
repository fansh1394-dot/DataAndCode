#!/usr/bin/env bash
# 守候式起跑：等待 GPU 空闲显存达标后，自动启动「三头 + ChannelPartitionFusion」150 轮训练。
# 现状：GPU 被外部租户反复抢占，需 (1) 门槛调高到能容纳 batch 32 的 ~19 GB；
#       (2) 若起跑瞬间被抢占导致 CUDA OOM，清理现场后重试，而不是直接放弃。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

NEED=${NEED:-21000}          # 起跑所需空闲显存（MiB）：batch 32 自身约 19 GB
NAME=${NAME:-mdp150_M3part3h_s0}
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-part3h.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/${NAME}.launch.log
TRAINLOG=logs/${NAME}.log
MAX_TRY=${MAX_TRY:-20}
mkdir -p logs YOLO26/ResTest/_aborted

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

attempt=0
while [ "$attempt" -lt "$MAX_TRY" ]; do
    attempt=$((attempt + 1))
    log "第 ${attempt}/${MAX_TRY} 次尝试：等待空闲显存 >= ${NEED} MiB"
    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then break; fi
        log "空闲 ${f:-NA} MiB < ${NEED} MiB，120s 后重试"
        sleep 120
    done

    # 现场清理：上一轮 OOM 留下的半成品目录挪走，保证本次是独立空目录
    if [ -e "YOLO26/ResTest/${NAME}" ]; then
        dst="YOLO26/ResTest/_aborted/${NAME}_try${attempt}_$(date +%H%M%S)"
        mv "YOLO26/ResTest/${NAME}" "$dst"
        log "已把残留目录移到 ${dst}"
    fi

    log "显存达标（空闲 ${f} MiB），开始训练"
    $PY -u train_v26.py \
        --config "$CFG" \
        --label-from dual \
        --data "$DATA" \
        --name "$NAME" \
        --seed 0 \
        --epochs 150 \
        --patience 30 \
        --batch 32 \
        --workers 4 \
        --amp false >> "$TRAINLOG" 2>&1
    code=$?

    if [ "$code" -eq 0 ]; then
        log "训练正常结束，exit=0；日志 ${TRAINLOG}"
        exit 0
    fi

    if grep -q "CUDA out of memory" "$TRAINLOG"; then
        log "本次因 CUDA OOM 退出（exit=${code}），300s 后重试"
        sleep 300
        continue
    fi

    log "训练异常退出（exit=${code}），非 OOM，放弃重试；日志 ${TRAINLOG}"
    exit "$code"
done

log "重试次数用尽（${MAX_TRY}），放弃"
exit 1
