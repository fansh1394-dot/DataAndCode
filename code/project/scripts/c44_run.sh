#!/usr/bin/env bash
# C4-4 留出验证：按 sequence 分 5 折，每折在 6 个错位档位上评测
#   b = 基线 mdp150_M0_concat_s0；a = 错位鲁棒 mdp150_M0Async_s0
# GPU 已被 E10 占满（剩 50 MiB），全部走 CPU；5 路并发，每路 8 线程
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
BASE=/3250410011/root/rivermind-data/mendeley_split
PY=/opt/conda/envs/rivermind/bin/python
OUT=$PROJ/c44_out
LOG=$PROJ/logs/c44
mkdir -p "$OUT" "$LOG"
cd "$PROJ" || exit 1

LEVELS="0 1 2 4 6 8"
JOBS=5

run_one() {
  local short=$1 sub=$2 lvl=$3
  local tag="c44_${short}_${sub}_${lvl}"
  [ -f "$OUT/$tag.json" ] && return 0
  local w
  case "$short" in
    b) w="YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt" ;;
    a) w="YOLO26/ResTest/mdp150_M0Async_s0/weights/best.pt" ;;
  esac
  local shift=""
  [ "$lvl" != "0" ] && shift="--ir-shift ${lvl},0"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  "$PY" -u scripts/eval_split.py \
    --weights "$w" \
    --data "$BASE/data.yaml" --split test --labels-mode dual \
    --val-list "lists/c44/${sub}.txt" \
    --tag "$tag" --batch 8 --workers 2 --device cpu \
    --out "$OUT/$tag.json" $shift \
    > "$LOG/$tag.log" 2>&1
  echo "$(date '+%T') $tag rc=$?" >> "$LOG/_progress.log"
}
export -f run_one
export PROJ BASE PY OUT LOG

{
  for m in b a; do
    for f in fold0 fold1 fold2 fold3 fold4 kind_real kind_synth; do
      for l in $LEVELS; do echo "$m $f $l"; done
    done
  done
} > "$LOG/_tasks.txt"

echo "[$(date '+%F %T')] 任务数=$(wc -l < "$LOG/_tasks.txt")  并发=$JOBS" | tee "$LOG/_run.log"
cat "$LOG/_tasks.txt" | xargs -P "$JOBS" -L 1 bash -c 'run_one "$0" "$1" "$2"'
echo "[$(date '+%F %T')] ALL_DONE  产出=$(ls "$OUT"/*.json 2>/dev/null | wc -l)" | tee -a "$LOG/_run.log"
