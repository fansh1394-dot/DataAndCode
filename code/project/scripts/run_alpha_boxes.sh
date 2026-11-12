#!/usr/bin/env bash
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=$PROJ/alpha_boxes
LOG=$PROJ/logs/alpha_boxes
mkdir -p "$OUT" "$LOG"
cd "$PROJ" || exit 1

for L in 0 2 4 8; do
  for A in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2; do
    F="$OUT/boxes_a${A}_l${L}.json"
    [ -f "$F" ] && continue
    ALPHA=$A LEVEL=$L "$PY" -u alpha_boxes_eval.py > "$LOG/a${A}_l${L}.log" 2>&1
    echo "$(date '+%T') alpha=$A level=$L rc=$?" >> "$LOG/_progress.log"
  done
done
echo "ALPHA_BOXES_DONE $(date '+%F %T')" >> "$LOG/_progress.log"
