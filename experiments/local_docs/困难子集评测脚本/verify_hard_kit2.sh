#!/bin/bash
# 复核加固后的困难子集评测套件是否就位（语法、关键片段、幂等、目标状态）。
set -u
S=/root/rivermind-data/scripts
L=/root/rivermind-data/logs
RUN=/root/rivermind-data/MutilModel_3485988898/ResTest/v2_S3_s0

echo "=============== 1) 脚本指纹 ==============="
md5sum "$S/compare_hard_final.sh" "$S/watch_s3_final_eval.sh" "$S/launch_watch_eval.sh" 2>&1
ls -l "$S/compare_hard_final.sh" "$S/watch_s3_final_eval.sh" "$S/launch_watch_eval.sh" 2>&1 | awk '{print $5, $9}'

echo
echo "=============== 2) bash 语法 ==============="
for f in compare_hard_final.sh watch_s3_final_eval.sh launch_watch_eval.sh; do
    if bash -n "$S/$f" 2>/tmp/e_$f; then echo "SYNTAX_OK   $f"; else echo "SYNTAX_FAIL $f"; cat /tmp/e_$f; fi
    rm -f /tmp/e_$f
done

echo
echo "=============== 3) 加固点是否到位 ==============="
echo "json_ok 出现次数      : $(grep -c 'json_ok' "$S/watch_s3_final_eval.sh")"
echo "MAXH 默认值(应为24)   : $(grep -oE 'MAXH=\"\$\{4:-[0-9]+\}\"' "$S/watch_s3_final_eval.sh")"
echo "launch 传参 120 24    : $(grep -c '120 24' "$S/launch_watch_eval.sh")"
echo "坏 JSON 改名留证      : $(grep -c 'bad_' "$S/watch_s3_final_eval.sh")"
echo "CKPT_EP 日志          : $(grep -c 'CKPT_EP' "$S/watch_s3_final_eval.sh")"
echo "摘要 grep 关键词      : $(grep -o '核心判定|融合是否可用' "$S/watch_s3_final_eval.sh" | head -1)"
echo "compare 参数化 TAG    : $(grep -c 'TAG_A' "$S/compare_hard_final.sh")"
echo "解析修复(cur=None)    : $(grep -c '报数口径提醒' "$S/compare_hard_final.sh")"

echo
echo "=============== 4) 依赖脚本 ==============="
for f in eval_hard.sh eval_hard_subset.py; do
    if [ -f "$S/$f" ]; then echo "OK   $S/$f  ($(wc -c < "$S/$f") bytes)"; else echo "MISS $S/$f"; fi
done

echo
echo "=============== 5) 幂等 / 现状 ==============="
if [ -f "$L/.watch_S3.pid" ]; then
    P=$(cat "$L/.watch_S3.pid")
    if kill -0 "$P" 2>/dev/null; then echo "守护已存在 pid=$P"; else echo "陈旧 pidfile pid=$P（进程不存在）"; fi
else
    echo "尚无守护 pidfile（可安全启动）"
fi
echo "评测进程: $(pgrep -fc 'eval_hard_subset.py' 2>/dev/null || echo 0) 个"
echo "训练进程: $(pgrep -fc 'train_full.py.*--name v2_S3_s0' 2>/dev/null || echo 0) 个"
echo "S3 已完成轮数: $(( $(wc -l < "$RUN/results.csv" 2>/dev/null || echo 1) - 1 ))/100"
echo -n "hard_eval_S3.json: "; [ -f "$L/hard_eval_S3.json" ] && echo YES || echo NO
echo -n "N0 hard_eval_N0.json: "; [ -f "$L/hard_eval_N0.json" ] && echo YES || echo NO

echo
echo "=============== 6) GPU ==============="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1
