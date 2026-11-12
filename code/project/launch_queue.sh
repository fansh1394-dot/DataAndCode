#!/bin/bash
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
mkdir -p logs
setsid nohup bash scripts/queue_distill800_s0.sh >> logs/distill800_chain.log 2>&1 < /dev/null &
echo "QUEUE_PID=$!"
sleep 3
ps -o pid,cmd --no-headers -p "$!" 2>/dev/null
