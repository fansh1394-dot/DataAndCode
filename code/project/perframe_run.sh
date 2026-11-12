#!/usr/bin/env bash
# 逐帧置信度导出：6 个错位档，3 路并发（CPU）
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
LOG=$PROJ/logs/perframe
mkdir -p "$LOG"
cd "$PROJ" || exit 1

for L in 0 1 2 4 6 8; do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 LEVEL=$L \
    "$PY" -u perframe_conf.py > "$LOG/lv$L.log" 2>&1
  echo "$(date '+%T') level=$L rc=$?" >> "$LOG/_progress.log"
done
echo "ALL_DONE $(date '+%F %T')" >> "$LOG/_progress.log"
