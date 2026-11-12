#!/bin/bash
# D3-P2 进度快照
# 列号：mAP50-95 rgb(22) ir(27) fuse(32) fuse_ir(37)；cls loss rgb(4) ir(7) fuse(10) fuse_ir(13)
R=/3250410011/root/rivermind-data/MutilModel_3485988898
CSV=$R/ResTest/v2_D3P2_forceir_headonly_s0/results.csv
LOG=$R/_d2p1/v2_D3P2_forceir_headonly_s0.log

echo "=== 进程 ==="
pgrep -f "train_full.py.*v2_D3P2" >/dev/null && echo RUNNING || echo DONE
echo "=== 已完成轮数 ==="
[ -f "$CSV" ] && echo "$(($(wc -l < "$CSV") - 1)) / 40" || echo "results.csv 未生成"
echo ""
echo "=== 逐轮 mAP50-95（fuse 头本次输入仅含 IR） ==="
printf "ep   rgb95     ir95      fuse95    fuse_ir95\n"
[ -f "$CSV" ] && awk -F, 'NR==1{next} {printf "%-4s %-9.4f %-9.4f %-9.4f %s\n", $1,$22,$27,$32,$37}' "$CSV"
echo ""
echo "=== 逐轮 cls loss ==="
printf "ep   cls_rgb   cls_ir    cls_fuse  cls_fuse_ir\n"
[ -f "$CSV" ] && awk -F, 'NR==1{next} {printf "%-4s %-9.4f %-9.4f %-9.4f %s\n", $1,$4,$7,$10,$13}' "$CSV"
echo ""
echo "=== 结束标志 ==="
[ -f "$LOG" ] && tr '\r' '\n' < "$LOG" | grep -aE "epochs completed|EarlyStopping" | tail -2
echo ""
echo "=== 判读（fuse95）==="
[ -f "$CSV" ] && awk -F, 'NR==1{next} {f=$32} END{
  if (f+0 > 0.15) printf "  最新 fuse95=%.4f -> **IR 特征可用**，瓶颈在训练信号分配\n", f;
  else if (f+0 > 0.01) printf "  最新 fuse95=%.4f -> 部分可用，介于地板与可用之间\n", f;
  else printf "  最新 fuse95=%.4f -> **IR 特征对 fuse 头不可用**（贴地板）\n", f;
}' "$CSV"
