#!/usr/bin/env bash
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
mkdir -p logs
setsid nohup bash scripts/queue_e19_crop.sh > logs/e19_launch.out 2>&1 < /dev/null &
echo $! > logs/e19.pid
sleep 3
echo "PID=$(cat logs/e19.pid)"
ps -p "$(cat logs/e19.pid)" -o pid,etime,cmd --no-headers 2>/dev/null || echo "进程已退出"
tail -n 3 logs/e19_chain.log 2>/dev/null
