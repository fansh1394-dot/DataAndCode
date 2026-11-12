#!/bin/bash
# 启动 v26_chain2b（先 cp 快照，绝不直接跑源文件）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
SRC=$PROJ/YOLO26/scripts/v26_chain2b.sh
SNAP=$PROJ/YOLO26/scripts/.c2b_run.sh
cd "$PROJ" || exit 1

bash -n "$SRC" || { echo "语法检查失败"; exit 1; }
cp -f "$SRC" "$SNAP" || { echo "cp 失败"; exit 1; }
echo "snapshot: $(md5sum "$SNAP" | cut -c1-32)  $(stat -c%s "$SNAP") bytes"

# 清掉可能残留的探针
pkill -f '/tmp/_probe_final.py' 2>/dev/null
pkill -f '/tmp/_bench_fwd.py' 2>/dev/null
sleep 2

pgrep -af 'train_v26.py' | head -3 && { echo "⚠️ 已有训练在跑，请先确认"; }
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"

nohup setsid bash "$SNAP" > /dev/null 2>&1 &
sleep 8
echo "chain pid: $(pgrep -f '.c2b_run.sh' | head -1)"
echo "pidfile:   $(cat "$PROJ/YOLO26/logs/v26_chain2b.pid" 2>/dev/null)"
