#!/bin/bash
# IR-only 10ep 探测进度
R=/3250410011/root/rivermind-data/MutilModel_3485988898
CSV=$R/ResTest/_probe_xonly_10ep/results.csv
LOG=$R/_d2p1/_probe_xonly_10ep.log

echo "=== 进程 ==="
pgrep -f "_probe_xonly_10ep" >/dev/null && echo RUNNING || echo DONE
echo "=== 进度 ==="
if [ -f "$CSV" ]; then
  echo "已完成 $(($(wc -l < "$CSV") - 1)) / 10 轮"
  echo ""
  echo "=== 逐轮指标（列: epoch mAP50 mAP50-95 P R） ==="
  awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
  {printf "ep%-3s  mAP50=%-8.4f  mAP50-95=%-8.4f  P=%-7.4f  R=%-7.4f\n",
    $1, $(h["metrics/mAP50(B)"]), $(h["metrics/mAP50-95(B)"]),
    $(h["metrics/precision(B)"]), $(h["metrics/recall(B)"])}' "$CSV"
else
  echo "[results.csv 未生成]"
fi
echo ""
echo "=== 训练损失（判断是否在学） ==="
[ -f "$CSV" ] && awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
{printf "ep%-3s  box=%-9.4f  cls=%-9.4f  dfl=%s\n", $1,
  $(h["train/box_loss"]), $(h["train/cls_loss"]), $(h["train/dfl_loss"])}' "$CSV"
echo ""
echo "=== 日志关键行 ==="
[ -f "$LOG" ] && tr '\r' '\n' < "$LOG" | grep -aE "Starting training|epochs completed|Error|Traceback" | tail -3
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
