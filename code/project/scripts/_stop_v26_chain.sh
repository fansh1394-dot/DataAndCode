#!/bin/bash
# 停止 v26_chain2 及其子训练进程（按 pidfile + 命令行精确匹配，避免自杀）
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
MYPID=$$
PIDF=YOLO26/logs/v26_chain2.pid

if [ -f "$PIDF" ]; then
  CP=$(cat "$PIDF" 2>/dev/null || echo 0)
  if [ -n "$CP" ] && [ "$CP" != "0" ] && kill -0 "$CP" 2>/dev/null; then
    echo "kill chain pid=$CP"
    kill -TERM "$CP" 2>/dev/null
  fi
fi

# 逐进程解析 cmdline，排除自己
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$MYPID" ] && continue
  cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
  case "$cmd" in
    *v26_chain2.sh*|*train_v26.py*)
      echo "kill pid=$pid : ${cmd:0:80}"
      kill -TERM "$pid" 2>/dev/null
      ;;
  esac
done

sleep 3
echo '--- remaining ---'
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$MYPID" ] && continue
  cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
  case "$cmd" in
    *v26_chain2.sh*|*train_v26.py*) echo "STILL: $pid ${cmd:0:70}" ;;
  esac
done
rm -f "$PIDF"
echo done
