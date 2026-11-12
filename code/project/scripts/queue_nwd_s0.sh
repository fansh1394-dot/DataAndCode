#!/usr/bin/env bash
# 极小目标专项 · 方法一：NWD 回归损失（唯一变量）
#
# 基线 mdp150_M0_concat_s0 配方逐字复制，唯一差异是 --loss-box nwd。
# 判据（极小目标桶 lt16，fuse 头，test 614 帧 / 552 GT）：
#   AP50-95: 基线三种子 0.5354 / 0.5556 / 0.5348，σ≈0.012，2σ≈0.024
#   正向：lt16 AP50-95 > 0.5594（基线均值 0.5486 + 2σ）且总体 mAP50-95 不掉出 2σ
#
# 2026-11-04 10:20 加固：本实例是共享卡，其他租户会瞬时占满显存。
# 起跑前的可用显存检查与实际分配之间存在竞态，故加 OOM 重试。
#
# 2026-11-04 10:55 恢复 batch 32（与基线逐字一致）。此前 OOM 是因为另一租户
# 的 20.99 GiB 进程占着这张共享卡，现已释放，空闲 21.5 GiB。
# 若要临时降显存，可用环境变量 BATCH=16 启动（此时 accumulate=4，等效批量仍为 64）。
set -u

# batch 32 峰值 ~20.3 GiB，本卡常态只剩 ~21.4 GiB。默认的 caching allocator 在
# 大块/小块混排时会因碎片化申请不到连续段，backward 里 cuBLAS 建 handle 时报
# CUBLAS_STATUS_ALLOC_FAILED（表面看不像 OOM，但同属显存不足）。
# expandable_segments 让显存段可增长、显著缓解碎片，不改变任何数值结果。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0NWD_s0
LOG=YOLO26/logs/${NAME}.log
RUN=YOLO26/ResTest/${NAME}
NEED=${NEED:-20000}
BATCH=${BATCH:-32}
MAX_TRY=${MAX_TRY:-30}
mkdir -p YOLO26/logs logs

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/nwd_chain.log; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

attempt=0
while [ "$attempt" -lt "$MAX_TRY" ]; do
    attempt=$((attempt + 1))
    if [ -f "$RUN/results.csv" ]; then
        ep=$(tail -1 "$RUN/results.csv" | cut -d, -f1 | tr -d ' ')
        if [ "${ep:-0}" -ge 149 ] 2>/dev/null; then log "${NAME} 已完成 ep${ep}，跳过"; exit 0; fi
        mkdir -p YOLO26/ResTest/_aborted
        mv "$RUN" "YOLO26/ResTest/_aborted/${NAME}_retry$(date +%H%M%S)" 2>/dev/null
        log "残留目录已挪至 _aborted"
    fi

    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then break; fi
        log "等待显存：空闲 ${f:-NA} MiB < ${NEED}"
        sleep 120
    done

    log "第 ${attempt}/${MAX_TRY} 次起跑 ${NAME}（空闲 $(free_mib) MiB）"
    : > "$LOG"
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
        --loss-box nwd \
        >> "$LOG" 2>&1
    CODE=$?

    if [ "$CODE" -eq 0 ]; then log "${NAME} 完成 exit=0"; exit 0; fi
    if grep -aqE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUBLAS_STATUS_INTERNAL_ERROR|CUDNN_STATUS_NOT_INITIALIZED" "$LOG"; then
        log "${NAME} 显存不足（exit=${CODE}），300s 后重试"
        sleep 300
        continue
    fi
    log "${NAME} 异常退出 exit=${CODE}（非 OOM），放弃"
    exit "$CODE"
done
log "${NAME} 尝试次数用尽，放弃"
exit 1
