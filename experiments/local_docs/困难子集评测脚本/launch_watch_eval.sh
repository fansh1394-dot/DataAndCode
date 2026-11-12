#!/bin/bash
# 启动 S3 终评守护（脱离会话，立即返回）。
#
# 为什么单独一层：run_remote.ps1 用 `bash -s` 把命令喂给远端 shell，
# setsid nohup + 重定向 + & 这一串必须在远端自己的脚本里执行，
# 否则 PowerShell 5 会破坏内联的 `$!`/重定向写法。
set -u
ROOT=/root/rivermind-data
L=$ROOT/logs
TAG="${1:-S3}"
TOTAL="${2:-100}"

mkdir -p "$L"
PIDF="$L/.watch_${TAG}.pid"

if [ -f "$PIDF" ]; then
    OLD=$(cat "$PIDF" 2>/dev/null || echo "")
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "ALREADY_WATCHING pid=$OLD"
        ps -o pid,etime,cmd -p "$OLD" 2>/dev/null | tail -1
        exit 0
    fi
    echo "陈旧 pidfile（pid=$OLD 已不存在），清理后重启"
    rm -f "$PIDF"
fi

# 实测：单轮 ≈ 780s（13 min），S3 还剩 70 轮 ≈ 15.2h。
# MAX_WAIT_H 给 24h，留约 9h 余量，防止服务器被挤占变慢时静默超时。
setsid nohup bash "$ROOT/scripts/watch_s3_final_eval.sh" "$TAG" "$TOTAL" 120 24 \
    > "$L/watch_${TAG}_boot.log" 2>&1 < /dev/null &
NEW=$!
echo "LAUNCHED watchdog pid=$NEW  TAG=$TAG"
echo "  boot log = $L/watch_${TAG}_boot.log"
echo "  实时日志 = $L/watch_${TAG}.log"
sleep 3
echo "--- 启动后 3s 巡检 ---"
if kill -0 "$NEW" 2>/dev/null; then
    echo "守护存活"
else
    echo "守护已退出，boot log:"
    tail -20 "$L/watch_${TAG}_boot.log" 2>/dev/null
fi
[ -f "$L/watch_${TAG}.log" ] && tail -5 "$L/watch_${TAG}.log"
echo
echo "--- 当前训练进度 ---"
wc -l < "$ROOT/MutilModel_3485988898/ResTest/v2_S3_s0/results.csv" 2>/dev/null | sed 's/^/rows(incl header)=/'
