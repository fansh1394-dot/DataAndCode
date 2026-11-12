#!/usr/bin/env bash
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
LOG=$PROJ/logs/perframe_boxes
mkdir -p "$LOG" "$PROJ/perframe_boxes"
cd "$PROJ" || exit 1
for L in 0 1 2 4 6 8; do
  [ -f "$PROJ/perframe_boxes/boxes_$L.json" ] && continue
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 LEVEL=$L \
    "$PY" -u perframe_boxes.py > "$LOG/lv$L.log" 2>&1
  echo "$(date '+%T') level=$L rc=$?" >> "$LOG/_progress.log"
done
echo "ALL_DONE $(date '+%F %T')" >> "$LOG/_progress.log"
