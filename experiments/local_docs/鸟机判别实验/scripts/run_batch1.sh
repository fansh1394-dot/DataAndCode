#!/bin/bash
# 第一批对比/消融：顺序执行 4 个训练（共享 GPU 自适应 + OOM 重试）
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
    echo "[batch1] $(date '+%F %T') free=${FREE}MiB 等待" >> "$OUT/queue.log"
    sleep 60
  done
}

consolidate() {  # name：YOLO11 结果从 ResTest/ 归档到 YOLO26/ResTest/，与 YOLO26 结果集中存放
  local name=$1
  local src="$PROJ/ResTest/$name"
  [ -d "$src" ] || return 0
  mkdir -p "$PROJ/YOLO26/ResTest"
  mv "$src" "$PROJ/YOLO26/ResTest/$name"
  echo "[batch1] $(date '+%F %T') 归档 $name -> YOLO26/ResTest/" | tee -a "$OUT/queue.log"
}

run_job() {  # entry, name, extra args...
  local entry=$1; local name=$2; shift 2
  local log="$OUT/${name}.log"
  local attempt=0
  while [ $attempt -lt 2 ]; do
    wait_gpu
    echo "[batch1] $(date '+%F %T') $name entry=$entry batch=$BATCH attempt=$attempt" | tee -a "$log"
    cd "$PROJ"
    PYTHONPATH=. $PY -u "$entry" "$@" --batch "$BATCH" \
      --imgsz 640 --workers 8 --amp false --device 0 >> "$log" 2>&1
    if grep -q "OutOfMemoryError" "$log"; then
      attempt=$((attempt+1)); sleep 300; continue
    fi
    break
  done
}

# 1) RGB-only：YOLO11n 主流单模态基线（COCO 预训练，走通用入口）
run_job train_full.py rgb_yolo11n_s0 \
  --route B \
  --pretrained yolo11n.pt \
  --config /3250410011/root/rivermind-data/MutilModel_3485988898/ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml \
  --single-modal rgb \
  --data /root/rivermind-fs/birddrone/bd_nc3/data_nc3.yaml \
  --label-from dual \
  --epochs 150 --patience 30 --lr0 0.01 \
  --name bdc3_rgbonly_yolo11n_s0
consolidate bdc3_rgbonly_yolo11n_s0

# 2) IR-only：YOLO26n 单模态（从零，150 ep）
run_job train_v26.py ir_yolo26n_s0 \
  --route A \
  --config /3250410011/root/rivermind-data/MutilModel_3485988898/ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-single-baseline.yaml \
  --single-modal x \
  --data /root/rivermind-fs/birddrone/bd_nc3/data_nc3.yaml \
  --label-from ir \
  --epochs 150 --patience 30 --lr0 0.01 \
  --name bdc3_ironly_yolo26n_s0

# 3) IR 配准消融：未配准 AVID IR（与 E36 同初始化/同轮数）
run_job train_v26.py nc3_unreg_s0 \
  --route C \
  --pretrained /root/rivermind-fs/birddrone/nc3_init.pt \
  --config /3250410011/root/rivermind-data/MutilModel_3485988898/ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml \
  --build-nc 3 \
  --data /root/rivermind-fs/birddrone/bd_nc3_unreg/data_unreg.yaml \
  --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 \
  --name bdc3_nc3_unreg_s0

# 4) nc=3 seed1（稳健性）
run_job train_v26.py nc3_seed1 \
  --route C \
  --pretrained /root/rivermind-fs/birddrone/nc3_init.pt \
  --config /3250410011/root/rivermind-data/MutilModel_3485988898/ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml \
  --build-nc 3 \
  --data /root/rivermind-fs/birddrone/bd_nc3/data_nc3.yaml \
  --label-from dual \
  --epochs 150 --patience 30 --lr0 0.002 --seed 1 \
  --name bdc3_nc3_s1

echo "[batch1] $(date '+%F %T') ALL DONE" | tee -a "$OUT/queue.log"
