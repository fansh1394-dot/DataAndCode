#!/bin/bash
# 单模态基线（E1 RGB-only / E2 IR-only）进度快照
R=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$R/_d2p1

echo "=== 看门狗 ==="
WP=$(cat "$WDIR/watchdog_baseline.pid" 2>/dev/null || echo "")
echo "pid=$WP 存活=$(kill -0 "$WP" 2>/dev/null && echo YES || echo NO)"
echo "--- 最近日志 ---"
tail -6 "$WDIR/watchdog_baseline.log" 2>/dev/null

for NAME in v2_E1_rgbonly_s0 v2_E2_ironly_s0; do
  CSV=$R/ResTest/$NAME/results.csv
  LOG=$WDIR/$NAME.log
  echo ""
  echo "========== $NAME =========="
  if [ ! -f "$LOG" ]; then
    echo "  [尚未开始]"
    continue
  fi
  pgrep -f "$NAME" >/dev/null && echo "  状态: RUNNING" || echo "  状态: 已结束"
  if [ -f "$CSV" ]; then
    echo "  进度: $(($(wc -l < "$CSV") - 1)) / 100 轮"
    echo "  逐轮（每 5 轮或最后 3 轮）:"
    awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; n=0; next}
    {n++; e=$1; m50=$(h["metrics/mAP50(B)"]); m95=$(h["metrics/mAP50-95(B)"]);
     b=$(h["train/box_loss"]); c=$(h["train/cls_loss"]);
     printf "    ep%-3s mAP50=%-7.4f mAP50-95=%-7.4f box=%-7.4f cls=%-7.4f\n", e, m50, m95, b, c}' "$CSV" | awk 'NR%5==1 || NR>=(n-2)' 
    echo "  --- 末轮 ---"
    awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
    {m50=$(h["metrics/mAP50(B)"]); m95=$(h["metrics/mAP50-95(B)"]); e=$1}
    END{printf "    ep%s  mAP50=%.4f  mAP50-95=%.4f\n", e, m50, m95}' "$CSV"
  else
    echo "  [results.csv 未生成，当前进度]"
    tr '\r' '\n' < "$LOG" | grep -aE "^ +[0-9]+/100" | tail -1
  fi
  echo "  --- 每轮耗时（从日志时间戳估算）---"
  tr '\r' '\n' < "$LOG" | grep -aoE "[0-9]+/100 .*\[[0-9:]+<[0-9:]+" | tail -1 | grep -oE "<[0-9:]+" | head -1
done

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== 主机负载 ==="
uptime | sed 's/^ *//'
