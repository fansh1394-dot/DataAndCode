#!/usr/bin/env bash
set -u
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
PY=/opt/conda/envs/rivermind/bin/python
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-gated.yaml
ADATA=$PWD/_d2p1/async_yamls/data_train_async.yaml
NAME=mdp2_G1smoke
$PY -u train_v26.py \
  --config "$CFG" --label-from dual --data "$ADATA" --name "$NAME" --seed 0 \
  --epochs 2 --patience 30 --batch 16 --imgsz 640 --workers 4 --amp false \
  --mm-async-geo-p 0.5 --mm-async-geo-translate-px 1 8 \
  --mm-alpha-anchor-weight 0.05 \
  > "logs/${NAME}.log" 2>&1
echo "SMOKE_DONE rc=$?" >> "logs/${NAME}.log"
