#!/usr/bin/env bash
# E24b：DT800 配方的种子重复（seed 0 -> 1，batch 16 不变）
# 起因：E24（batch 32）在 24 GB 上两次 CUDA OOM（峰值 23.2~23.4 GiB），
#       batch 32 + 800 教师特征蒸馏不可行，改为用种子重复量化 run-to-run 方差，
#       回答「AP75 0.5996 / lt16 0.5544 能否复现」。
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0DT800_s1
LOG=YOLO26/logs/${NAME}.log
OUT=YOLO26/ResTest/${NAME}
mkdir -p YOLO26/logs logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -f "$OUT/weights/best.pt" ] && [ ! -f "$OUT/weights/last.pt" ]; then
    rm -rf "$OUT"
fi

echo "[$(date '+%F %T')] 等待显存 >= 20000 MiB" | tee -a logs/e24b_chain.log
while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "${f}" ] && [ "${f}" -ge 20000 ]; then break; fi
    echo "[$(date '+%F %T')] 等待：空闲 ${f:-NA} MiB" | tee -a logs/e24b_chain.log
    sleep 120
done

echo "[$(date '+%F %T')] 起跑 ${NAME}" | tee -a logs/e24b_chain.log
: > "$LOG"
$PY -u train_full.py \
    --route A \
    --config "$CFG" \
    --project YOLO26/ResTest \
    --label-from dual \
    --data "$DATA" \
    --name "$NAME" \
    --seed 1 \
    --epochs 150 \
    --patience 30 \
    --batch 16 \
    --workers 4 \
    --amp false \
    --distill ultralytics/cfg/mm_distill_dual800.yaml feature \
    >> "$LOG" 2>&1
CODE=$?
echo "[$(date '+%F %T')] ${NAME} 结束 exit=${CODE}" | tee -a logs/e24b_chain.log
