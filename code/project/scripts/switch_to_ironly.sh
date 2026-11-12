#!/bin/bash
# 切换到 IR-only 单基线：先停掉 RGB 链式看门狗与其训练进程，再启动 IR-only 看门狗
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1
cd "$PROJ" || exit 1

echo "=== 1. 停止旧看门狗 ==="
OLDWD=$(cat "$WDIR/watchdog_baseline.pid" 2>/dev/null || echo "")
if [ -n "$OLDWD" ] && kill -0 "$OLDWD" 2>/dev/null; then
  kill -TERM "$OLDWD" 2>/dev/null && echo "  已 TERM 看门狗 pid=$OLDWD"
  sleep 3
  kill -9 "$OLDWD" 2>/dev/null && echo "  已 KILL 看门狗 pid=$OLDWD"
else
  echo "  旧看门狗未运行"
fi
rm -f "$WDIR/watchdog_baseline.pid"

echo ""
echo "=== 2. 停止 RGB-only 训练进程 ==="
for pid in $(pgrep -f "v2_E1_rgbonly_s0"); do
  if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
    kill -9 "$pid" 2>/dev/null && echo "  killed $pid"
  fi
done
sleep 5
echo "  剩余 train_full 进程: $(pgrep -f 'train_full.py' | wc -l)"

echo ""
echo "=== 3. 清理未完成的 RGB 目录（仅几轮，不构成结论） ==="
if [ -d "$PROJ/ResTest/v2_E1_rgbonly_s0" ]; then
  n=$(( $(wc -l < "$PROJ/ResTest/v2_E1_rgbonly_s0/results.csv" 2>/dev/null || echo 1) - 1 ))
  echo "  v2_E1_rgbonly_s0 已完成 $n 轮，删除"
  rm -rf "$PROJ/ResTest/v2_E1_rgbonly_s0"
fi
rm -f "$WDIR/v2_E1_rgbonly_s0.log"

echo ""
echo "=== 4. GPU 状态 ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

echo ""
echo "=== 5. 启动 IR-only 看门狗 ==="
setsid nohup bash "$PROJ/scripts/watchdog_ironly.sh" \
  > "$WDIR/watchdog_ironly_stdout.log" 2>&1 < /dev/null &
sleep 25

echo "看门狗 pid: $(cat "$WDIR/watchdog_ironly.pid" 2>/dev/null || echo '[未写入]')"
echo ""
echo "=== 6. 看门狗日志 ==="
tail -10 "$WDIR/watchdog_ironly.log" 2>/dev/null || echo "[日志未生成]"
echo ""
echo "=== 7. 训练进程 ==="
pgrep -af "v2_E2_ironly_s0" | head -2
echo ""
echo "=== 8. 启动日志 ==="
tr '\r' '\n' < "$WDIR/v2_E2_ironly_s0.log" 2>/dev/null | grep -aE "single_modal|Starting training|Transferred|Error|Traceback" | head -5
echo ""
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
