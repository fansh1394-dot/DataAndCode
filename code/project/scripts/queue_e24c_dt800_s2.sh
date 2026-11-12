#!/usr/bin/env bash
# E24c：DT800 配方的 seed 2（与 seed 0 / seed 1 组成三次重复）
# 目的：给训练侧最强改动（跨分辨率特征蒸馏）一个均值±标准差，判断 AP75 的
#       0.5996/0.5790 波动是种子噪声还是真实效应。
# 协议与 E11 / E24b 完全一致，仅 --seed 2。
# 自带容错：显存等待 + 崩溃后回退到最新未 strip 检查点续训。
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0DT800_s2
LOG=YOLO26/logs/${NAME}.log
RUN=YOLO26/ResTest/${NAME}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p YOLO26/logs logs

STARTED=0
for attempt in 1 2 3 4 5 6; do
    if [ -f "$RUN/weights/last.pt" ]; then
        sz=$(stat -c%s "$RUN/weights/last.pt")
        if [ "$sz" -lt 20000000 ]; then
            newest=$(ls "$RUN"/weights/epoch*.pt 2>/dev/null | sort -V | tail -1)
            if [ -n "$newest" ]; then
                cp "$newest" "$RUN/weights/last.pt"
                echo "[$(date '+%F %T')] last.pt 已 strip（${sz} B），回退到 ${newest}" | tee -a logs/e24c_chain.log
            else
                echo "[$(date '+%F %T')] 无可用检查点，无法续训" | tee -a logs/e24c_chain.log
                exit 4
            fi
        fi
    fi
    while true; do
        f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "${f}" ] && [ "${f}" -ge 20000 ]; then break; fi
        echo "[$(date '+%F %T')] 第${attempt}次：等待显存，当前空闲 ${f:-NA} MiB" | tee -a logs/e24c_chain.log
        sleep 120
    done

    if [ "$STARTED" -eq 0 ]; then
        echo "[$(date '+%F %T')] 起跑 ${NAME}（seed 2）" | tee -a logs/e24c_chain.log
        : > "$LOG"
        EXTRA=""
        STARTED=1
    else
        echo "[$(date '+%F %T')] 第${attempt}次续训 ${NAME}" | tee -a logs/e24c_chain.log
        EXTRA="--resume"
    fi

    $PY -u train_full.py \
        --route A \
        --config "$CFG" \
        --project YOLO26/ResTest \
        --label-from dual \
        --data "$DATA" \
        --name "$NAME" \
        --seed 2 \
        --epochs 150 \
        --patience 30 \
        --batch 16 \
        --workers 4 \
        --amp false \
        --distill ultralytics/cfg/mm_distill_dual800.yaml feature \
        $EXTRA >> "$LOG" 2>&1
    code=$?
    echo "[$(date '+%F %T')] 第${attempt}次结束 exit=${code}" | tee -a logs/e24c_chain.log
    if [ "$code" -eq 0 ]; then break; fi
    sleep 180
done
