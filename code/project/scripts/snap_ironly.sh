#!/bin/bash
# IR-only 基线进度快照
R=/3250410011/root/rivermind-data/MutilModel_3485988898
CSV=$R/ResTest/v2_E2_ironly_s0/results.csv
LOG=$R/_d2p1/v2_E2_ironly_s0.log

echo "=== 进程 ==="
pgrep -f "v2_E2_ironly_s0" >/dev/null && echo RUNNING || echo DONE
echo "=== 看门狗 ==="
WP=$(cat "$R/_d2p1/watchdog_ironly.pid" 2>/dev/null || echo "")
echo "pid=$WP 存活=$(kill -0 "$WP" 2>/dev/null && echo YES || echo NO)"
echo ""
echo "=== 进度 ==="
if [ -f "$CSV" ]; then
  echo "已完成 $(($(wc -l < "$CSV") - 1)) / 100 轮"
  echo ""
  awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
  {printf "ep%-3s  mAP50=%-8.4f  mAP50-95=%-8.4f  P=%-7.4f  R=%-7.4f  box=%-8.4f  cls=%-8.4f\n",
    $1, $(h["metrics/mAP50(B)"]), $(h["metrics/mAP50-95(B)"]),
    $(h["metrics/precision(B)"]), $(h["metrics/recall(B)"]),
    $(h["train/box_loss"]), $(h["train/cls_loss"])}' "$CSV"
else
  echo "[results.csv 未生成]"
  tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aE "^ +[0-9]+/100" | tail -1
fi
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
