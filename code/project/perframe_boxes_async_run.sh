#!/usr/bin/env bash
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
TAG=async
W=$PROJ/YOLO26/ResTest/mdp150_M0Async_s0/weights/best.pt
LOG=$PROJ/logs/perframe_boxes_$TAG
mkdir -p "$LOG" "$PROJ/perframe_boxes_$TAG"
cd "$PROJ" || exit 1
for L in 0 1 2 4 6 8; do
  [ -f "$PROJ/perframe_boxes_$TAG/boxes_$L.json" ] && continue
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    TAG=$TAG WEIGHTS=$W LEVEL=$L "$PY" -u perframe_boxes_m.py > "$LOG/lv$L.log" 2>&1
  echo "$(date '+%T') level=$L rc=$?" >> "$LOG/_progress.log"
done
echo "ALL_DONE $(date '+%F %T')" >> "$LOG/_progress.log"
