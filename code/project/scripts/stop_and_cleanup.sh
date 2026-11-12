#!/bin/bash
# ============================================================
# 实例迁移前：停止所有训练 + 清理本轮产生的结果
#
# 保留项（重要）：
#   - ResTest/v2_D3P2_forceir_headonly_s0   七条否定链的终局证据（40 轮已跑满）
#   - 其余历史 run（C4 / N0 / S3 / S6a / S9_fix 等）均为既有对照，不动
#
# 删除项（本轮新产生、未完成或仅冒烟）：
#   - v2_E2_ironly_s0            刚启动的 IR-only 100ep（仅几轮）
#   - _smoke_rgbonly_baseline    RGB 单模态冒烟（2 轮）
#   - _smoke_xonly_baseline      IR 单模态冒烟（2 轮）
#   - _probe_xonly_10ep          IR 10 轮探测（趋势已记录，产物可弃）
#   - _smoke_v2_D3_partition     D3-P1 通道分工冒烟（2 轮，机制已证伪）
# ============================================================
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1

echo "############ 1. 停止看门狗 ############"
for f in "$WDIR/watchdog_ironly.pid" "$WDIR/watchdog_baseline.pid"; do
  if [ -f "$f" ]; then
    P=$(cat "$f" 2>/dev/null || echo "")
    if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then
      kill -TERM "$P" 2>/dev/null && echo "  TERM $(basename "$f") pid=$P"
      sleep 3
      kill -9 "$P" 2>/dev/null && echo "  KILL pid=$P"
    fi
    rm -f "$f"
  fi
done

echo ""
echo "############ 2. 停止所有训练进程 ############"
for pid in $(pgrep -f "train_full.py"); do
  if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
    kill -9 "$pid" 2>/dev/null && echo "  killed $pid"
  fi
done
sleep 5
echo "  剩余 train_full.py 进程: $(pgrep -f 'train_full.py' | wc -l)"

echo ""
echo "############ 3. GPU 状态 ############"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

echo ""
echo "############ 4. 待删除清单（先列出，含大小） ############"
TARGETS="v2_E2_ironly_s0 _smoke_rgbonly_baseline _smoke_xonly_baseline _probe_xonly_10ep _smoke_v2_D3_partition"
TOTAL=0
for t in $TARGETS; do
  if [ -d "$PROJ/ResTest/$t" ]; then
    SZ=$(du -sm "$PROJ/ResTest/$t" 2>/dev/null | cut -f1)
    echo "  [目录] $t  ${SZ} MB"
    TOTAL=$((TOTAL + SZ))
  else
    echo "  [目录] $t  (不存在)"
  fi
done
echo "  ---- 合计约 ${TOTAL} MB ----"

echo ""
echo "############ 5. 执行删除 ############"
for t in $TARGETS; do
  rm -rf "$PROJ/ResTest/$t" && echo "  已删 ResTest/$t"
done
# 对应日志与临时产物
rm -f "$WDIR/v2_E2_ironly_s0.log" "$WDIR/watchdog_ironly.log" "$WDIR/watchdog_ironly_stdout.log"
rm -f "$WDIR/v2_E1_rgbonly_s0.log" "$WDIR/watchdog_baseline.log" "$WDIR/watchdog_stdout.log"
rm -f "$WDIR/_smoke_rgbonly_baseline.log" "$WDIR/_smoke_xonly_baseline.log"
rm -f "$WDIR/_probe_xonly_10ep.log" "$WDIR/_smoke_v2_D3_partition.log"
echo "  已删 _d2p1 下相关日志"

echo ""
echo "############ 6. 清理结果：剩余 ResTest 目录 ############"
ls -1 "$PROJ/ResTest/" | sort

echo ""
echo "############ 7. 确认 D3-P2 证据仍在 ############"
D=$PROJ/ResTest/v2_D3P2_forceir_headonly_s0
if [ -f "$D/results.csv" ]; then
  echo "  results.csv: $(($(wc -l < "$D/results.csv") - 1)) 轮  ($(du -sh "$D" | cut -f1))"
  echo "  末轮 fuse mAP50-95:"
  awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} {v=$(h["metrics/fuse/metrics/mAP50-95(B)"])} END{printf "    %.4f\n", v}' "$D/results.csv"
  ls -la "$D/weights/" 2>/dev/null | grep -E "best|last"
else
  echo "  [警告] D3-P2 结果缺失！"
fi

echo ""
echo "############ 8. 磁盘占用 ############"
df -h /3250410011/root | tail -1
