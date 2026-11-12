#!/bin/bash
# 清理训练残留进程（dataloader worker 在主进程死后仍存活）
# 用法: bash scripts/kill_workers.sh
set -u
PATTERN='train_full.py.*v2c_E1_rgbonly_s1'
for i in 1 2 3; do
  PIDS=$(pgrep -f "$PATTERN" | grep -v "^$$\$" | tr '\n' ' ')
  if [ -z "$PIDS" ]; then
    echo "[kill_workers] 无残留"
    break
  fi
  echo "[kill_workers] 第 $i 轮，终止: $PIDS"
  kill -9 $PIDS 2>/dev/null
  sleep 3
done
echo "--- 剩余 ---"
pgrep -af "$PATTERN" || echo "clean"
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- 训练进程(全项目) ---"
pgrep -af '[t]rain_full.py' || echo "no train_full"
