#!/bin/bash
# E28b：从 E28 best 出发，做 AVID:Mendeley ≈ 1:1 的平衡微调
set -e
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
OUT=/root/rivermind-fs/birddrone/E28b
mkdir -p "$OUT"

cd "$PROJ"
PYTHONPATH=. $PY -u train_v26.py \
  --route C \
  --pretrained YOLO26/ResTest/mdp150_M0BD_nc2_s0/weights/best.pt \
  --config ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml \
  --build-nc 2 \
  --data /root/rivermind-fs/birddrone/bd_nc2/data_bal.yaml \
  --label-from dual \
  --epochs 40 --patience 15 \
  --batch 32 --imgsz 640 --workers 8 \
  --amp false --seed 0 --device 0 \
  --lr0 0.002 \
  --name mdp150_M0BD_nc2_bal_s0 \
  --mm-async-geo-p 0.5 \
  --mm-async-geo-translate-px 0 6 \
  --mm-async-geo-target x \
  > "$OUT/e28b_train.log" 2>&1
