#!/bin/bash
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
echo
echo "===== 容器内训练/探针进程 ====="
pgrep -af 'train_v26.py|train_full.py|_probe|_bench_fwd' | head -10
echo "(空则无)"
echo
echo "===== 链进程 ====="
for f in v26_chain2b v26_chain2; do
  P=YOLO26/logs/$f.pid
  if [ -f "$P" ]; then
    I=$(cat "$P" 2>/dev/null || echo 0)
    kill -0 "$I" 2>/dev/null && echo "$f alive pid=$I" || echo "$f DEAD pid=$I"
  else echo "$f no pidfile"; fi
done
echo
echo "===== 各 run 状态 ====="
for d in YOLO26/ResTest/*/; do
  n=$(basename "$d"); csv="$d/results.csv"
  r=0; [ -f "$csv" ] && r=$(( $(wc -l < "$csv") - 1 ))
  printf "%-28s ep=%-4s best=%s last=%s\n" "$n" "$r" \
    "$([ -f "$d/weights/best.pt" ] && echo Y || echo -)" \
    "$([ -f "$d/weights/last.pt" ] && echo Y || echo -)"
done
echo
echo "===== 链日志尾 30 行 ====="
tail -30 YOLO26/logs/v26_chain2b.log 2>/dev/null | tr '\r' '\n' | grep -av '^\s*$' | grep -avE '%\|' | tail -20
echo
echo "===== E4 日志尾（若有）====="
L=YOLO26/logs/v26_E4_deform_s0.log
[ -f "$L" ] && tr '\r' '\n' < "$L" | grep -aE 'OutOfMemory|epochs completed|Error|batch=32|显存' | tail -8 || echo "no E4 log"
