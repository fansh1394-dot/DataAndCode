#!/usr/bin/env bash
# E24：DT800 配方 + batch 32（排除 batch 16 的协议偏差）
# 与 mdp150_M0DT800_s0 唯一差异：batch 16 -> 32。其余（640 训练、dual800 教师特征蒸馏、
# seed 0、150 ep、patience 30、mosaic 1.0、close_mosaic 15、amp false）逐字相同。
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0E24dt800b32_s0
LOG=YOLO26/logs/${NAME}.log
OUT=YOLO26/ResTest/${NAME}
mkdir -p YOLO26/logs logs

# 首轮（08:32）在 epoch 2 处 CUDA OOM：batch32 峰值 23.2G/23.64G 只剩 ~440 MiB。
# 打开可扩展段分配器回收 694 MiB 的 reserved-unallocated 碎片后再试一次。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 清掉首轮未产出权重的半成品目录，避免 ultralytics 因目录已存在而改名成 *2
if [ ! -f "$OUT/weights/best.pt" ] && [ ! -f "$OUT/weights/last.pt" ]; then
    rm -rf "$OUT"
fi

echo "[$(date '+%F %T')] 等待显存 >= 20000 MiB" | tee -a logs/e24_chain.log
while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "${f}" ] && [ "${f}" -ge 20000 ]; then break; fi
    echo "[$(date '+%F %T')] 等待：空闲 ${f:-NA} MiB" | tee -a logs/e24_chain.log
    sleep 120
done

echo "[$(date '+%F %T')] 起跑 ${NAME}" | tee -a logs/e24_chain.log
: > "$LOG"
$PY -u train_full.py \
    --route A \
    --config "$CFG" \
    --project YOLO26/ResTest \
    --label-from dual \
    --data "$DATA" \
    --name "$NAME" \
    --seed 0 \
    --epochs 150 \
    --patience 30 \
    --batch 32 \
    --workers 4 \
    --amp false \
    --distill ultralytics/cfg/mm_distill_dual800.yaml feature \
    >> "$LOG" 2>&1
CODE=$?
echo "[$(date '+%F %T')] ${NAME} 结束 exit=${CODE}" | tee -a logs/e24_chain.log
