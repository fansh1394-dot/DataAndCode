#!/bin/bash
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
echo "=== 链日志尾 ==="
tail -8 YOLO26/logs/v26_chain2b.log
echo
echo "=== 停链（先 chain 再 train，排除自身 PID）==="
MYPID=$$
P=$(cat YOLO26/logs/v26_chain2b.pid 2>/dev/null)
echo "pidfile=$P"
for x in $(pgrep -f '.c2b_run.sh'); do
  if [ "$x" != "$MYPID" ]; then
    kill -TERM "$x" 2>/dev/null && echo "killed chain=$x"
  fi
done
for x in $(pgrep -f 'train_v26.py'); do
  kill -TERM "$x" 2>/dev/null && echo "killed train=$x"
done
sleep 6
echo
echo "=== 残留 ==="
pgrep -af 'c2b_run|train_v26' | grep -v "$MYPID" || echo "无"
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
