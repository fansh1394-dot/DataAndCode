#!/usr/bin/env bash
# 极小目标专项 · 方法二：CIoU + λ·NWD 混合回归（唯一变量）
#
# 基线配方逐字复制，唯一差异是 --loss-box ciou_nwd。
# 设计依据：纯 NWD（λ=1）收敛后每目标原始损失约 0.20，而 CiOU 约 0.006，
# 在 box 权重 7.5 下回归梯度约为分类的 40 倍，会压垮分类分支；
# 取 λ=0.05 后 NWD 贡献约 0.010，与 CiOU 同量级，只提供额外定位梯度。
# 判据同 NWD 轮：lt16 AP50-95 需 > 0.5594（基线三种子均值 + 2σ）。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0CNW_s0
LOG=YOLO26/logs/${NAME}.log
mkdir -p YOLO26/logs logs

echo "[$(date '+%F %T')] 等待显存 >= 20000 MiB" | tee -a logs/cnw_chain.log
while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "${f}" ] && [ "${f}" -ge 20000 ]; then break; fi
    echo "[$(date '+%F %T')] 等待：空闲 ${f:-NA} MiB" | tee -a logs/cnw_chain.log
    sleep 120
done

echo "[$(date '+%F %T')] 起跑 ${NAME}" | tee -a logs/cnw_chain.log
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
    --loss-box ciou_nwd \
    >> "$LOG" 2>&1

CODE=$?
echo "[$(date '+%F %T')] ${NAME} 结束 exit=${CODE}" | tee -a logs/cnw_chain.log
