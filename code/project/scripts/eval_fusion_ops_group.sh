#!/usr/bin/env bash
# A 阶段：为融合算子替换实验补齐 test 集评测
#
# 为什么必须补：训练期 best.pt 是按 val 集挑选的，val 最优不等于 test 最优。
# 论文表中所有数字必须同口径（同一 test 集、同一 batch、同一 labels-mode），
# 否则跨组比较不成立。
#
# 注意：CMCE 此刻可能仍在训练并占用显存。batch 降到 16，
# 若显存不足则等待，不抢训练。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/mendeley_split/data.yaml
OUT_DIR=YOLO26/ResTest/_test_eval
LOG=logs/eval_fusion_ops.log
BATCH=${BATCH:-16}
NEED=${NEED:-9000}       # 与训练共存的保守门槛

mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== 融合算子组 test 评测：concat -> CGAFusion -> PIM -> CMCE ==="

for NAME in mdp150_M0_concat_s0 mdp150_M0CGAFusion_s0 mdp150_M0PIM_s0 mdp150_M0CMCE_s0; do
    W="YOLO26/ResTest/${NAME}/weights/best.pt"
    OUT="${OUT_DIR}/${NAME}_test.json"

    if [ ! -f "$W" ]; then
        log "跳过 $NAME：缺权重 $W"
        continue
    fi
    if [ -f "$OUT" ]; then
        log "跳过 $NAME：已有 $OUT"
        continue
    fi

    # 等显存：与训练共存时只要够跑 batch=16 即可，不要求全空
    for _ in $(seq 1 120); do
        f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ] && break
        log "  等待显存：空闲 ${f:-NA} MiB < ${NEED} MiB"
        sleep 60
    done

    log "开始评测 $NAME（空闲 ${f} MiB）"
    $PY -u scripts/eval_split.py \
        --weights "$W" \
        --data "$DATA" \
        --split test \
        --labels-mode dual \
        --imgsz 640 \
        --batch "$BATCH" \
        --workers 4 \
        --tag "${NAME}_test" \
        --out "$OUT" >> "logs/eval_${NAME}.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -f "$OUT" ]; then
        log "  $NAME 完成 -> $OUT"
    else
        log "  $NAME 失败 exit=${code}，见 logs/eval_${NAME}.log"
    fi
done

log "=== A 阶段评测结束 ==="
