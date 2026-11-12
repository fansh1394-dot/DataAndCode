#!/usr/bin/env bash
# IR 错位退化曲线（test 集，评测 only；权重 = 基线 mdp150_M0_concat_s0/best.pt）
#
# 口径
#   - split=test / labels-mode=dual / imgsz=640，与 abc_test.sh 完全一致
#   - batch=8（训练链 mdp150_M0DFL_s0 正在占 GPU，故降 batch；batch 不影响 mAP）
#   - 平移单位是 640 输入空间像素（eval_split.py --ir-shift 内部按每帧 letterbox
#     比例换算回原图），旋转绕图像中心
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/mendeley_split/data.yaml
W=YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt
OUTD=YOLO26/ResTest/_misalign
LOG=logs/misalign_curve.log
BATCH=${BATCH:-8}
WORKERS=${WORKERS:-2}

mkdir -p "$OUTD"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_one() {
    local tag="$1"
    shift
    local out="$OUTD/${tag}.json"
    if [ -f "$out" ]; then log "跳过 ${tag}（已存在）"; return 0; fi
    log "起跑 ${tag}  $*"
    $PY -u scripts/eval_split.py \
        --weights "$W" \
        --data "$DATA" \
        --split test \
        --labels-mode dual \
        --imgsz 640 \
        --batch "$BATCH" \
        --workers "$WORKERS" \
        --tag "$tag" \
        --out "$out" \
        "$@" >> "logs/mis_${tag}.log" 2>&1
    local code=$?
    if [ "$code" -ne 0 ]; then log "  ${tag} 失败 exit=${code}"; return "$code"; fi
    local v
    v=$(grep -oE 'fuse=[0-9.]+' "logs/mis_${tag}.log" | tail -1)
    log "  完成 ${tag}  ${v}"
    return 0
}

[ -f "$W" ] || { log "致命：缺权重 $W"; exit 2; }

log "=== IR 平移（水平，正=向右）==="
for DX in 1 2 3 4 6 12 16 24; do
    tag=$(printf 's0_dx%02d' "$DX")
    run_one "$tag" --ir-shift "${DX},0" || exit 7
done

log "=== IR 旋转（绕中心，正=逆时针）==="
for DEG in 0.5 1 2 3 5; do
    tag="s0_rot${DEG/./p}"
    run_one "$tag" --ir-rotate "$DEG" || exit 8
done

log "=== 全部完成 ==="
