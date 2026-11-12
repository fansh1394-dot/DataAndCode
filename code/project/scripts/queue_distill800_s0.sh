#!/usr/bin/env bash
# 跨分辨率特征蒸馏（尺度蒸馏）· 800 教师 → 640 学生
#
# 背景：唯一被验证有效的极小目标干预是「640 训练 + 800 推理」（lt16 AP50-95
# 0.5419 → 0.5731，AP75 0.5538 → 0.6143，≈2.6σ），但端侧时延从 78.6 ms 涨到
# 约 120 ms。伪标签精炼已被实测否决（教师框与 GT 中位 IoU 仅 0.803，上限就是
# 教师自身 AP，640 学生拿不到 800 的网格密度）。本脚本走特征侧：教师按 800
# 前向，其融合颈 C3k2 输出（层 48/51/54，见 yolo26 配置）由 F.interpolate
# 对齐到学生的 640 空间。
#
# 协议与 mdp150_M0_concat_s0 逐字一致，唯一差异是 --distill。
# 判定线：lt16 AP50-95 ≥ 0.565、AP75 ≥ 0.582、总体 mAP50-95 ≥ 0.7449。
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
DIST=ultralytics/cfg/mm_distill_dual800.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0DT800_s0
SMOKE=mdp2_DT800_smoke
LOG=YOLO26/logs/${NAME}.log
RUN=YOLO26/ResTest/${NAME}
NEED=${NEED:-20000}
BATCH=${BATCH:-32}
MAX_TRY=${MAX_TRY:-10}
mkdir -p YOLO26/logs logs

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/distill800_chain.log; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

wait_gpu() {
    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then break; fi
        log "等待显存：空闲 ${f:-NA} MiB < ${NEED} MiB"
        sleep 120
    done
}

# ---- 1. 冒烟：2 轮，只验证管线不验证指标（OOM 可重试，非 OOM 才终止） ----
smoke_ok=0
attempt=0
while [ "$attempt" -lt "$MAX_TRY" ]; do
    attempt=$((attempt + 1))
    wait_gpu
    rm -rf "YOLO26/ResTest/${SMOKE}" 2>/dev/null
    log "冒烟 第 ${attempt}/${MAX_TRY} 次 batch=${BATCH}（空闲 $(free_mib) MiB）"
    : > "YOLO26/logs/${SMOKE}.log"
    $PY -u train_v26.py \
        --config "$CFG" \
        --label-from dual \
        --data "$DATA" \
        --name "$SMOKE" \
        --seed 0 \
        --epochs 2 \
        --patience 30 \
        --batch "$BATCH" \
        --workers 4 \
        --amp false \
        --distill "$DIST" feature \
        >> "YOLO26/logs/${SMOKE}.log" 2>&1
    CODE=$?
    if [ "$CODE" -eq 0 ]; then smoke_ok=1; break; fi
    if grep -aqE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUBLAS_STATUS_INTERNAL_ERROR|CUDNN_STATUS_NOT_INITIALIZED" "YOLO26/logs/${SMOKE}.log"; then
        log "冒烟 OOM（exit=${CODE}）"
        if [ "$BATCH" -gt 16 ]; then
            BATCH=16
            log "冒烟 batch 下调为 16（正式训练同步使用该值）"
        fi
        sleep 300
        continue
    fi
    log "冒烟非 OOM 失败 exit=${CODE}，终止；末 20 行日志："
    tail -n 20 "YOLO26/logs/${SMOKE}.log" | tee -a logs/distill800_chain.log
    exit "$CODE"
done
[ "$smoke_ok" -eq 1 ] || { log "冒烟重试次数用尽，放弃"; exit 1; }
log "冒烟通过（batch=${BATCH}）"

# ---- 2. 正式：150 轮 ----
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

    wait_gpu
    log "第 ${attempt}/${MAX_TRY} 次起跑 ${NAME}  batch=${BATCH}（空闲 $(free_mib) MiB）"
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
        --distill "$DIST" feature \
        >> "$LOG" 2>&1
    CODE=$?

    if [ "$CODE" -eq 0 ]; then log "${NAME} 完成 exit=0"; exit 0; fi
    if grep -aqE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUBLAS_STATUS_INTERNAL_ERROR|CUDNN_STATUS_NOT_INITIALIZED" "$LOG"; then
        log "${NAME} 显存不足（exit=${CODE}）"
        if [ "$BATCH" -gt 16 ]; then
            BATCH=16
            log "batch 下调为 16（与基线 batch=32 的差异需在结论文档中标注）"
            sleep 300
            continue
        fi
        sleep 300
        continue
    fi
    log "${NAME} 异常退出 exit=${CODE}（非 OOM），放弃；见 ${LOG}"
    exit "$CODE"
done
log "${NAME} 尝试次数用尽，放弃"
exit 1
