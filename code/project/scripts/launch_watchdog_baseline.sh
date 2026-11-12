#!/bin/bash
# 启动链式看门狗（脱离 SSH 会话，nohup + setsid）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1
cd "$PROJ" || exit 1

echo "=== 启动前检查 ==="
echo "现有训练进程数: $(pgrep -f 'train_full.py' | wc -l)"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

echo ""
echo "=== 清理残留训练进程（若有） ==="
for pid in $(pgrep -f "train_full.py"); do
  if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
    kill -9 "$pid" 2>/dev/null && echo "  killed $pid"
  fi
done
sleep 3

echo ""
echo "=== 启动看门狗 ==="
setsid nohup bash "$PROJ/scripts/watchdog_baseline.sh" \
  > "$WDIR/watchdog_stdout.log" 2>&1 < /dev/null &
sleep 5

echo "看门狗 pid: $(cat "$WDIR/watchdog_baseline.pid" 2>/dev/null || echo '[未写入]')"
echo ""
echo "=== 看门狗日志（前 20 行） ==="
sleep 20
tail -20 "$WDIR/watchdog_baseline.log" 2>/dev/null || echo "[日志未生成]"
echo ""
echo "=== 训练进程 ==="
pgrep -af train_full.py | head -3
