#!/usr/bin/env bash
# E24b 续训：mdp150_M0DT800_s1
# 10:33 崩溃：同机另一进程占 14.53 GiB + 本进程 9.10 GiB > 23.64 GiB，发生在 final_eval。
# 收尾时 strip_optimizer 已把 last.pt 的 epoch 置 -1（无法 resume），故从最近一个未 strip 的
# epochN.pt（25 MB，带优化器）恢复；若日后再次崩溃，脚本会自动做同样的回退。
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
NAME=mdp150_M0DT800_s1
LOG=YOLO26/logs/${NAME}.log
RUN=YOLO26/ResTest/${NAME}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 首次执行：保留崩溃现场，把 epoch30.pt 顶到 last.pt
if [ ! -f "$RUN/results_crash_1033.csv" ]; then
    cp "$RUN/results.csv" "$RUN/results_crash_1033.csv"
    [ -f "$RUN/weights/best.pt" ] && cp "$RUN/weights/best.pt" "$RUN/weights/best_ep32_stripped.pt"
fi

for attempt in 1 2 3 4 5 6; do
    # last.pt 若已被 strip（约 13 MB）则回退到最新的 epochN.pt（约 25 MB）
    if [ -f "$RUN/weights/last.pt" ]; then
        sz=$(stat -c%s "$RUN/weights/last.pt")
        if [ "$sz" -lt 20000000 ]; then
            newest=$(ls "$RUN"/weights/epoch*.pt 2>/dev/null | sort -V | tail -1)
            if [ -n "$newest" ]; then
                cp "$newest" "$RUN/weights/last.pt"
                echo "[$(date '+%F %T')] last.pt 已 strip（${sz} B），回退到 ${newest}" | tee -a logs/e24b_chain.log
            else
                echo "[$(date '+%F %T')] 无可用 epochN.pt，无法续训" | tee -a logs/e24b_chain.log
                exit 4
            fi
        fi
    fi
    while true; do
        f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "${f}" ] && [ "${f}" -ge 20000 ]; then break; fi
        echo "[$(date '+%F %T')] 第${attempt}次：等待显存，当前空闲 ${f:-NA} MiB" | tee -a logs/e24b_chain.log
        sleep 120
    done
    echo "[$(date '+%F %T')] 第${attempt}次续训 ${NAME}" | tee -a logs/e24b_chain.log
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
        --resume >> "$LOG" 2>&1
    code=$?
    echo "[$(date '+%F %T')] 第${attempt}次结束 exit=${code}" | tee -a logs/e24b_chain.log
    if [ "$code" -eq 0 ]; then break; fi
    sleep 180
done
