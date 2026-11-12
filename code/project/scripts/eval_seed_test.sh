#!/usr/bin/env bash
# 补齐种子组 s1/s2 的 test 评测，用于统一种子方差标定口径
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/mendeley_split/data.yaml
OUT_DIR=YOLO26/ResTest/_test_eval
LOG=logs/eval_seed_test.log

mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== 种子组 s1/s2 test 评测 ==="
for NAME in mdp150_M0_concat_s1 mdp150_M0_concat_s2; do
    W="YOLO26/ResTest/${NAME}/weights/best.pt"
    OUT="${OUT_DIR}/${NAME}_test.json"
    [ -f "$W" ] || { log "跳过 $NAME：缺权重"; continue; }
    [ -f "$OUT" ] && { log "跳过 $NAME：已有结果"; continue; }

    log "评测 $NAME"
    $PY -u scripts/eval_split.py \
        --weights "$W" --data "$DATA" --split test --labels-mode dual \
        --imgsz 640 --batch 16 --workers 4 \
        --tag "${NAME}_test" --out "$OUT" >> "logs/eval_${NAME}.log" 2>&1
    code=$?
    [ "$code" -eq 0 ] && [ -f "$OUT" ] && log "  $NAME 完成" || log "  $NAME 失败 exit=${code}"
done
log "=== 结束 ==="
