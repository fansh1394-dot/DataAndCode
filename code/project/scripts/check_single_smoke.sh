#!/bin/bash
cd /3250410011/root/rivermind-data/MutilModel_3485988898
for n in _smoke_rgbonly_baseline _smoke_xonly_baseline; do
  echo "===== $n ====="
  if [ -f "_d2p1/$n.log" ]; then
    tr '\r' '\n' < "_d2p1/$n.log" | grep -aE "single_modal|Transferred|Starting training|epochs completed|all +[0-9]|Error|Traceback|ValueError" | tail -8
  else
    echo "[无日志]"
  fi
  if [ -f "ResTest/$n/results.csv" ]; then
    echo "results.csv 行数: $(wc -l < "ResTest/$n/results.csv")"
  else
    echo "[无 results.csv]"
  fi
  echo ""
done
echo "=== 训练进程 ==="
pgrep -af train_full.py | head -3
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
