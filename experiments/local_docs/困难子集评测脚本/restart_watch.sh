#!/bin/bash
# 重启守护：先优雅停掉旧实例（pidfile 守卫会阻止新实例启动），再启动新版。
set -u
L=/root/rivermind-data/logs
PIDF=$L/.watch_S3.pid

echo "--- 停止旧守护 ---"
if [ -f "$PIDF" ]; then
    P=$(cat "$PIDF")
    echo "旧 pid=$P"
    kill "$P" 2>/dev/null && echo "已发 TERM"
    for i in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$P" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$P" 2>/dev/null; then
        echo "TERM 未生效，KILL"
        kill -9 "$P" 2>/dev/null
        sleep 1
    fi
    rm -f "$PIDF"
    echo "pidfile 已清理"
else
    echo "无 pidfile"
fi
pgrep -fa "watch_s3_final_eval.sh" 2>&1 | head -5 || true

echo
echo "--- 旧任务现场留证 ---"
if [ -f "$L/watch_S3_boot.log" ]; then
    cp -f "$L/watch_S3_boot.log" "$L/watch_S3_boot.log.prev_$(date +%m%d_%H%M%S)"
    echo "boot log 已备份"
fi

echo
echo "--- 用新版重启 ---"
bash /root/rivermind-data/scripts/launch_watch_eval.sh S3 100
echo
echo "--- 3s 后检查滚动日志 ---"
sleep 3
ls -l "$L/watch_S3.log" 2>&1
tail -5 "$L/watch_S3.log" 2>&1
