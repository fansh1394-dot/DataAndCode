#!/bin/bash
# E28：nc=2 主路线训练启动器（warm start + 显存门禁）
# 背景：GPU 与其它租户共享，当前仅约 4.6 GB 空闲。脚本按空闲显存选 batch 并等待，
# 不做「反复重试」：只在显存足够时启动一次，日志写 E28/e28_train.log。
set -e

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=/root/rivermind-fs/birddrone/E28
LOG=$OUT/e28_train.log
mkdir -p "$OUT"

WAIT_H=${WAIT_H:-43200}   # 最长等待 12h
DEADLINE=$(( $(date +%s) + WAIT_H ))

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -ge 20000 ]; then
    BATCH=32
    echo "[E28] $(date '+%F %T') free=${FREE}MiB -> batch=$BATCH" | tee -a "$LOG"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[E28] $(date '+%F %T') 等待超时（free=${FREE}MiB），未启动" | tee -a "$LOG"
    exit 3
  fi
  echo "[E28] $(date '+%F %T') free=${FREE}MiB < 20000MiB，等待中" >> "$LOG"
  sleep 60
done

cd "$PROJ"
PYTHONPATH=. $PY -u train_v26.py \
  --route C \
  --pretrained YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt \
  --config ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml \
  --build-nc 2 \
  --data /root/rivermind-fs/birddrone/bd_nc2/data.yaml \
  --label-from dual \
  --epochs 150 --patience 30 \
  --batch "$BATCH" --imgsz 640 --workers 4 \
  --amp false --seed 0 --device 0 \
  --name mdp150_M0BD_nc2_s0 \
  --mm-async-geo-p 0.5 \
  --mm-async-geo-translate-px 0 6 \
  --mm-async-geo-target x \
  >> "$LOG" 2>&1
