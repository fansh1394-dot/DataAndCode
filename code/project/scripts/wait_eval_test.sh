#!/bin/bash
# 等待 test 集四方评测收尾，并打印四个结果 JSON（供本地主控抓取）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
LOG=$PROJ/_d2p1/eval_test_all.log
for i in $(seq 1 120); do
  if grep -qa "全部完成" "$LOG" 2>/dev/null; then
    echo "DONE marker found after ${i} checks"
    break
  fi
  sleep 20
done
echo "=== 四个结果 JSON ==="
for t in v2_E1_rgbonly_test v2_E2_ironly_test v2_E3_ironly_lblir_test v2_C4_deform_test; do
  echo "##### $t #####"
  cat "$PROJ/ResTest/_test_eval/$t.json" 2>/dev/null || echo "(缺失)"
  echo
done
echo "=== 日志尾部 ==="
tr '\r' '\n' < "$LOG" | grep -aE "cp=|====|rc=|全部完成" | tail -20
