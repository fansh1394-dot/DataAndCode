#!/usr/bin/env bash
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
PY=/opt/conda/envs/rivermind/bin/python
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-gated.yaml
ADATA=$PWD/_d2p1/async_yamls/data_train_async.yaml
NAME=mdp1_G1probe
MM_ALPHA_ANCHOR_WEIGHT=0.05 MM_ALPHA_MIN=0.2 \
$PY -u train_v26.py \
  --config "$CFG" --label-from dual --data "$ADATA" --name "$NAME" --seed 0 \
  --epochs 1 --patience 30 --batch 16 --imgsz 640 --workers 0 --amp false \
  --mm-async-geo-p 0.5 --mm-async-geo-translate-px 1 8 \
  > "logs/${NAME}.log" 2>&1
echo "PROBE_DONE rc=$?" >> "logs/${NAME}.log"
