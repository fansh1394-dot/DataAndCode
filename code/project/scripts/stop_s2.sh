#!/bin/bash
# 停止 s2 全部进程（chain 启动的与手动启动的都算），再确认 GPU 释放
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python

for i in 1 2 3; do
  PIDS=$(pgrep -f 'train_full.py.*v2c_E1_rgbonly_s2' | tr '\n' ' ')
  [ -z "$PIDS" ] && break
  echo "[stop_s2] 第 $i 轮 kill -9: $PIDS"
  kill -9 $PIDS 2>/dev/null
  sleep 4
done

echo "--- 剩余 s2 进程 ---"
pgrep -af 'train_full.py.*v2c_E1_rgbonly_s2' || echo clean
echo "--- 全项目训练进程 ---"
pgrep -af '[t]rain_full.py' || echo clean
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "--- s2 目录 ---"
ls -la ResTest/v2c_E1_rgbonly_s2 2>/dev/null | head -8
