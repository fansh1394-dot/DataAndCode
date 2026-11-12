#!/bin/bash
# E36：三类检测器（bird/drone/other），batch 32 固定
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=/root/rivermind-fs/birddrone/E36
mkdir -p "$OUT"

TRY=0
while [ "$TRY" -lt 3 ]; do
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE" -ge 21500 ]; then break; fi
    echo "[E36] $(date '+%F %T') free=${FREE}MiB < 21500，等待 batch32" >> "$OUT/e36_wait.log"
    sleep 60
  done
  NAME=mdp150_M0BD_nc3_s0
  [ "$TRY" -gt 0 ] && NAME="${NAME}_r${TRY}"
  LOG="$OUT/e36_train_r${TRY}.log"
  echo "[E36] $(date '+%F %T') try=$TRY free=${FREE}MiB batch=32 name=$NAME" | tee -a "$LOG"
  cd "$PROJ"
  PYTHONPATH=. $PY -u train_v26.py \
    --route C \
    --pretrained /root/rivermind-fs/birddrone/nc3_init.pt \
    --config ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml \
    --build-nc 3 \
    --data /root/rivermind-fs/birddrone/bd_nc3/data_nc3.yaml \
    --label-from dual \
    --epochs 20 --patience 8 \
    --batch 32 --imgsz 640 --workers 8 \
    --amp false --seed 0 --device 0 \
    --lr0 0.002 \
    --name "$NAME" \
    --mm-async-geo-p 0.5 \
    --mm-async-geo-translate-px 0 6 \
    --mm-async-geo-target x \
    >> "$LOG" 2>&1
  if grep -q "OutOfMemoryError" "$LOG"; then
    echo "[E36] $(date '+%F %T') OOM，5 分钟后重试" | tee -a "$LOG"
    TRY=$((TRY + 1)); sleep 300; continue
  fi
  echo "[E36] $(date '+%F %T') 结束（非 OOM）" | tee -a "$LOG"
  break
done
