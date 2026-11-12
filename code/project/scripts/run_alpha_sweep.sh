#!/usr/bin/env bash
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=$PROJ/ResTest/_test_eval/alpha_sweep
LOG=$PROJ/logs/alpha
mkdir -p "$OUT" "$LOG"
cd "$PROJ" || exit 1

for L in 0 2 4 8; do
  for A in 1.0 0.8 0.6 0.4 0.2 0.0; do
    F="$OUT/a${A}_l${L}.json"
    [ -f "$F" ] && continue
    ALPHA=$A LEVEL=$L TAG=alpha "$PY" -u alpha_sweep_eval.py \
      > "$LOG/a${A}_l${L}.log" 2>&1
    echo "$(date '+%T') alpha=$A level=$L rc=$?" >> "$LOG/_progress.log"
  done
done
echo "ALPHA_SWEEP_DONE $(date '+%F %T')" >> "$LOG/_progress.log"
