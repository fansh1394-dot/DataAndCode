#!/usr/bin/env bash
# 顺序评测 M0mask(对照) 与 M0maskDist(实验)，产出整 test 集 JSON。
#
# 背景：M0mask 的评测在 2026-10-29 14:49 因 OOM 失败 —— 当时 M0maskDist 正在训练，
# 占 21.9 GiB，评测只剩 46 MiB 可用。现在两条训练都已结束、GPU 全空，重跑即可。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/mendeley_split/data.yaml
OUT_DIR=YOLO26/ResTest/_test_eval
LOG=logs/eval_m0_pair.log
BATCH=${BATCH:-16}   # 四分支评测显存偏大，比默认 32 降一档更稳

mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== M0mask / M0maskDist 顺序评测（batch=${BATCH}）==="

for NAME in mdp150_M0mask_s0 mdp150_M0maskDist_s0; do
    W="YOLO26/ResTest/${NAME}/weights/best.pt"
    OUT="${OUT_DIR}/${NAME}_test.json"

    if [ ! -f "$W" ]; then
        log "致命：找不到权重 $W，跳过 $NAME"
        continue
    fi
    if [ -f "$OUT" ]; then
        log "$NAME 已有结果，跳过：$OUT"
        continue
    fi

    log "开始评测 $NAME"
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
        log "$NAME 评测完成 -> $OUT"
    else
        log "$NAME 评测失败 exit=${code}，见 logs/eval_${NAME}.log"
    fi
done

log "=== 评测链结束 ==="
