#!/usr/bin/env bash
# E19 · 裁切放大（640 推理零代价）
# 基线配方逐字复制，唯一差异是 --mm-crop-zoom 0.6 + side 0.6~0.85。
# 设计：以 GT 为中心裁切原图 0.6~0.85 边长后 letterbox 到 640，等效放大 1.18~1.67 倍，
# 提高极小目标在训练输入中的网格占位；推理仍是 640。
# 判定线同 §6.1：lt16 AP50-95 >= 0.565 / AP75 >= 0.582 / 整体 >= 0.7449。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0E19crop_s0
LOG=YOLO26/logs/${NAME}.log
mkdir -p YOLO26/logs logs

echo "[$(date '+%F %T')] 等待显存 >= 20000 MiB" | tee -a logs/e19_chain.log
while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "${f}" ] && [ "${f}" -ge 20000 ]; then break; fi
    echo "[$(date '+%F %T')] 等待：空闲 ${f:-NA} MiB" | tee -a logs/e19_chain.log
    sleep 120
done

echo "[$(date '+%F %T')] 起跑 ${NAME}" | tee -a logs/e19_chain.log
: > "$LOG"
$PY -u train_v26.py \
    --config "$CFG" \
    --label-from dual \
    --data "$DATA" \
    --name "$NAME" \
    --seed 0 \
    --epochs 150 \
    --patience 30 \
    --batch 32 \
    --workers 4 \
    --amp false \
    --mm-crop-zoom 0.6 \
    --mm-crop-zoom-side 0.6 0.85 \
    >> "$LOG" 2>&1

CODE=$?
echo "[$(date '+%F %T')] ${NAME} 结束 exit=${CODE}" | tee -a logs/e19_chain.log
