#!/bin/bash
# 补跑：nc=3 seed0 统一 150 轮基线（等待第一批队列结束后自动启动）
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=/root/rivermind-fs/birddrone/batch1
mkdir -p "$OUT"

wait_gpu() {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$FREE" -ge 21500 ] && { BATCH=32; return; }
    [ "$FREE" -ge 16500 ] && { BATCH=24; return; }
    echo "[extra] $(date '+%F %T') free=${FREE}MiB 等待" >> "$OUT/queue.log"
    sleep 60
  done
}

run_job() {  # entry, name, extra args...
  local entry=$1; local name=$2; shift 2
  local log="$OUT/${name}.log"
  local attempt=0
  while [ $attempt -lt 2 ]; do
    wait_gpu
    echo "[extra] $(date '+%F %T') $name entry=$entry batch=$BATCH attempt=$attempt" | tee -a "$log"
    cd "$PROJ"
    PYTHONPATH=. $PY -u "$entry" "$@" --batch "$BATCH" \
      --imgsz 640 --workers 8 --amp false --device 0 >> "$log" 2>&1
    if grep -q "OutOfMemoryError" "$log"; then
      attempt=$((attempt+1)); sleep 300; continue
    fi
    break
  done
}

# 5) nc=3 seed0：150 轮同口径基线
run_job train_v26.py nc3_seed0_150 \
  --route C \
  --pretrained /root/rivermind-fs/birddrone/nc3_init.pt \
  --config /3250410011/root/rivermind-data/MutilModel_3485988898/ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml \
  --build-nc 3 \
  --data /root/rivermind-fs/birddrone/bd_nc3/data_nc3.yaml \
  --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 --seed 0 \
  --name bdc3_nc3_s0_150

echo "[extra] $(date '+%F %T') EXTRA DONE" | tee -a "$OUT/queue.log"
