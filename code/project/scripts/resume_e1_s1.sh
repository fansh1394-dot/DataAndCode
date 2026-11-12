#!/bin/bash
# 用冻结后的 last.pt 在新实例上续训 s1（原地续，不新建目录）
# 用法（在克隆后的项目根执行）: bash scripts/resume_e1_s1.sh
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=${PY:-/opt/conda/envs/rivermind/bin/python}
NAME=v2c_E1_rgbonly_s1
LOG=YOLO26/logs/${NAME}_resume.log
PIDF=YOLO26/logs/${NAME}.resume.pid

mkdir -p YOLO26/logs

# --- 闸门 1：ckpt 必须带 optimizer（否则 resume 会从 epoch 0 重跑） ---
"$PY" - "$PROJ/ResTest/$NAME/weights/last.pt" <<'PY' || exit 2
import sys
sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")
import torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ok = ck.get("optimizer") is not None and ck.get("model") is not None
print(f"[gate] epoch={ck.get('epoch')} optimizer={'OK' if ck.get('optimizer') else 'MISSING'} "
      f"model={'OK' if ck.get('model') is not None else 'MISSING'} "
      f"best_fitness={ck.get('best_fitness')}")
sys.exit(0 if ok else 2)
PY

# --- 闸门 2：防并发 ---
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "[resume] 已在运行 pid=$(cat "$PIDF")"; exit 0
fi

# --- 闸门 3：显存 ---
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "[resume] 空闲显存 ${FREE} MiB"
if [ "$FREE" -lt 12000 ]; then echo "[resume] 显存不足 12000 MiB，先释放再跑"; exit 3; fi

# --- 启动：--resume 会让 trainer 从 ckpt 恢复 epoch/优化器/EMA/调度器 ---
nohup setsid "$PY" -u train_full.py --route A \
  --config ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml \
  --single-modal rgb --label-from keep \
  --data /root/rivermind-data/uav_data_v2/data.yaml \
  --epochs 150 --batch 32 --imgsz 640 --workers 4 \
  --seed 1 --safe-augment --name "$NAME" --amp false \
  --resume > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[resume] 已启动 pid=$!  log=$LOG"

sleep 60
echo "--- 是否真的续训（应看到 Resuming training ... from epoch 105）---"
tr '\r' '\n' < "$LOG" | grep -aE "Resuming training|Starting training for|Traceback|Error" | head -5
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
