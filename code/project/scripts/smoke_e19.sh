#!/usr/bin/env bash
# E19 冒烟：验证裁切放大进入真实训练管线（不做指标）
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
PY=/opt/conda/envs/rivermind/bin/python
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
NAME=smoke_E19_crop_b2
LOG=YOLO26/logs/${NAME}.log
mkdir -p YOLO26/logs logs
: > "$LOG"
echo "[$(date '+%F %T')] smoke start" | tee -a "$LOG"
timeout 300 $PY -u train_v26.py \
    --config "$CFG" \
    --label-from dual \
    --data "$DATA" \
    --name "$NAME" \
    --seed 0 \
    --epochs 1 \
    --batch 2 \
    --workers 2 \
    --amp false \
    --mm-crop-zoom 0.6 \
    --mm-crop-zoom-side 0.6 0.85 \
    >> "$LOG" 2>&1
echo "[$(date '+%F %T')] smoke end exit=$?" | tee -a "$LOG"
