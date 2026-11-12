#!/bin/bash
# 等待第一批队列（run_batch1.sh）退出后，再启动补跑任务
set -u
OUT=/root/rivermind-fs/birddrone/batch1
while pgrep -f 'run_batch1[.]sh' > /dev/null; do
  sleep 300
done
echo "[waiter] $(date '+%F %T') 第一批队列已退出，启动补跑" >> "$OUT/queue.log"
bash /root/rivermind-fs/birddrone/run_extra_nc3_s0.sh
