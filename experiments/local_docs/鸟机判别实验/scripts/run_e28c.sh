#!/bin/bash
# E28c：Mendeley:AVID = 3:1 二阶段收尾（含显存自适应与 OOM 重试）
# 共享 GPU 上其它租户会随时抢占显存，因此：
#   - 每次尝试按当前空闲显存选最大可行 batch；
#   - 若 OOM，换新实验名、降低 batch 后重试，最多 3 次；
#   - 非 OOM 错误直接退出，不做无意义重试。
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=/root/rivermind-fs/birddrone/E28c
mkdir -p "$OUT"

pick_batch() {
  local free=$1
  local cand
  for cand in 32 24 16 8; do
    # 经验：总占用 ≈ 1.5GB + 0.68GB/样本
    local need=$(( cand * 700 + 2000 ))
    if [ "$free" -ge "$need" ]; then echo "$cand"; return 0; fi
  done
  echo 0
}

TRY=0
while [ "$TRY" -lt 3 ]; do
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    B=$(pick_batch "$FREE")
    if [ "$B" -gt 0 ]; then break; fi
    echo "[E28c] $(date '+%F %T') free=${FREE}MiB，显存不足，等待" >> "$OUT/e28c_wait.log"
    sleep 60
  done

  NAME=mdp150_M0BD_nc2_3to1_s0
  [ "$TRY" -gt 0 ] && NAME="${NAME}_r${TRY}"
  LOG="$OUT/e28c_train_r${TRY}.log"
  echo "[E28c] $(date '+%F %T') try=$TRY free=${FREE}MiB batch=$B name=$NAME" | tee -a "$LOG"

  cd "$PROJ"
  PYTHONPATH=. $PY -u train_v26.py \
    --route C \
    --pretrained YOLO26/ResTest/mdp150_M0BD_nc2_bal_s0/weights/last.pt \
    --config ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml \
    --build-nc 2 \
    --data /root/rivermind-fs/birddrone/bd_nc2/data_3to1.yaml \
    --label-from dual \
    --epochs 25 --patience 10 \
    --batch "$B" --imgsz 640 --workers 8 \
    --amp false --seed 0 --device 0 \
    --lr0 0.001 \
    --name "$NAME" \
    --mm-async-geo-p 0.5 \
    --mm-async-geo-translate-px 0 6 \
    --mm-async-geo-target x \
    >> "$LOG" 2>&1

  if grep -q "OutOfMemoryError" "$LOG"; then
    echo "[E28c] $(date '+%F %T') OOM，5 分钟后重试" | tee -a "$LOG"
    TRY=$((TRY + 1))
    sleep 300
    continue
  fi
  echo "[E28c] $(date '+%F %T') 结束（非 OOM），退出" | tee -a "$LOG"
  break
done
