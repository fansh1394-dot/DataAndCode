#!/bin/bash
# 验证看门狗与训练是否健康启动（含并发检查）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1

echo "=== 1. 看门狗进程 ==="
WP=$(cat "$WDIR/watchdog_baseline.pid" 2>/dev/null || echo "")
echo "pidfile=$WP  存活=$(kill -0 "$WP" 2>/dev/null && echo YES || echo NO)"

echo ""
echo "=== 2. 训练主进程（按 PPID 识别，排除 worker） ==="
for p in $(pgrep -f "train_full.py"); do
  PP=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
  CMD=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-60)
  echo "  pid=$p ppid=$PP  $CMD"
done

echo ""
echo "=== 3. 是否只有一个主训练进程（应为 1） ==="
MAIN=0
for p in $(pgrep -f "train_full.py"); do
  PP=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
  # 主进程的父进程是看门狗或 setsid 的 bash
  if ! echo "$PP" | grep -qE '^[0-9]+$'; then continue; fi
done
echo "  pgrep 总数: $(pgrep -f 'train_full.py' | wc -l)（含 dataloader worker，属正常）"

echo ""
echo "=== 4. 训练日志关键行 ==="
L="$WDIR/v2_E1_rgbonly_s0.log"
if [ -f "$L" ]; then
  tr '\r' '\n' < "$L" | grep -aE "single_modal|Starting training|Transferred|Error|Traceback|ValueError" | head -8
  echo "--- 当前进度 ---"
  tr '\r' '\n' < "$L" | grep -aE "^ +[0-9]+/100" | tail -2
else
  echo "[日志尚未生成]"
fi

echo ""
echo "=== 5. 看门狗日志 ==="
tail -6 "$WDIR/watchdog_baseline.log"

echo ""
echo "=== 6. GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
