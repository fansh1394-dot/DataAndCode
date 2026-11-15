#!/bin/bash
B=/root/rivermind-fs/birddrone/batch1
R=/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest
echo "=== 进程 ==="
pgrep -af 'run_batch1[.]sh|train_full[.]py|train_v26[.]py|wait_and_extra[.]sh|push_to_github[.]sh' | head -8
echo "=== queue.log ==="
cat "$B"/queue.log
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== 各日志进度 ==="
for f in rgb_yolo11n_s0 ir_yolo26n_s0 nc3_unreg_s0 nc3_seed1 nc3_seed0_150; do
  L=$B/$f.log
  if [ -f "$L" ]; then
    echo "--- $f ---"
    tr '\r' '\n' < "$L" | grep -E '^ +[0-9]+/150' | tail -1
    tr '\r' '\n' < "$L" | grep -E 'epochs completed|EarlyStopping|OutOfMemoryError|Traceback' | tail -2
  else
    echo "--- $f --- (无日志)"
  fi
done
echo "=== 输出目录 ==="
for d in "$R"/bdc3_*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  if [ -f "$d/results.csv" ]; then
    rows=$(($(wc -l < "$d/results.csv") - 1))
    best=$(sort -t, -k9 -g "$d/results.csv" | tail -1 | cut -d, -f1,9)
    echo "$n  epochs=$rows  best(ep,mAP50-95)=$best  weights=$(ls "$d/weights" 2>/dev/null | tr '\n' ' ')"
  else
    echo "$n  (无 results.csv)"
  fi
done
