#!/bin/bash
# 第二批消融：other 比例 0% / 30%，融合结构 gated / cga3h / cmce
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=/root/rivermind-fs/birddrone/batch2
CFG=$PROJ/ultralytics/cfg/models/mm/detect/yolo26/base
INIT=/root/rivermind-fs/birddrone/nc3_init.pt
BD=/root/rivermind-fs/birddrone
mkdir -p "$OUT"

wait_gpu() {
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$FREE" -ge 21500 ] && { BATCH=32; return; }
    [ "$FREE" -ge 16500 ] && { BATCH=24; return; }
    echo "[batch2] $(date '+%F %T') free=${FREE}MiB 等待" >> "$OUT/queue.log"
    sleep 60
  done
}

run_job() {  # entry, name, extra args...
  local entry=$1; local name=$2; shift 2
  local log="$OUT/${name}.log"
  local attempt=0
  while [ $attempt -lt 2 ]; do
    wait_gpu
    echo "[batch2] $(date '+%F %T') $name entry=$entry batch=$BATCH attempt=$attempt" | tee -a "$log"
    cd "$PROJ"
    PYTHONPATH=. $PY -u "$entry" "$@" --batch "$BATCH" \
      --imgsz 640 --workers 8 --amp false --device 0 >> "$log" 2>&1
    if grep -q "OutOfMemoryError" "$log"; then
      attempt=$((attempt+1)); sleep 300; continue
    fi
    break
  done
}

# B2: other 比例 0%（nc=3，合成帧去掉 class2）
run_job train_v26.py bdc3_other0_s0 \
  --route C --pretrained "$INIT" --config "$CFG/yolo26n-mm-dual-fusion-concat.yaml" \
  --build-nc 3 --data "$BD/bd_nc3_other0/data_other0.yaml" --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 \
  --name bdc3_other0_s0

# B3: other 比例 30%（nc=3）
run_job train_v26.py bdc3_other30_s0 \
  --route C --pretrained "$INIT" --config "$CFG/yolo26n-mm-dual-fusion-concat.yaml" \
  --build-nc 3 --data "$BD/bd_nc3_other30/data_other30.yaml" --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 \
  --name bdc3_other30_s0

# B4: 融合结构 gated
run_job train_v26.py bdc3_gated_s0 \
  --route C --pretrained "$INIT" --config "$CFG/yolo26n-mm-dual-fusion-gated.yaml" \
  --build-nc 3 --data "$BD/bd_nc3/data_nc3.yaml" --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 \
  --name bdc3_gated_s0

# B5: 融合结构 cga3h
run_job train_v26.py bdc3_cga3h_s0 \
  --route C --pretrained "$INIT" --config "$CFG/yolo26n-mm-dual-fusion-cga3h.yaml" \
  --build-nc 3 --data "$BD/bd_nc3/data_nc3.yaml" --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 \
  --name bdc3_cga3h_s0

# B6: 融合结构 cmce3h
run_job train_v26.py bdc3_cmce_s0 \
  --route C --pretrained "$INIT" --config "$CFG/yolo26n-mm-dual-fusion-cmce3h.yaml" \
  --build-nc 3 --data "$BD/bd_nc3/data_nc3.yaml" --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 \
  --name bdc3_cmce_s0

echo "[batch2] $(date '+%F %T') ALL DONE" | tee -a "$OUT/queue.log"
