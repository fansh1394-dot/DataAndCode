#!/bin/bash
set -e
cd /3250410011/root/rivermind-data/MutilModel_3485988898
sed 's/"save_json": False/"save_json": True/' scripts/eval_split.py > /tmp/eval_avid_bird.py
LIST=${LIST:-avid_bird_val_flat.txt}
OUT=${OUT:-/root/rivermind-fs/birddrone/E27/E27_avid_bird_640.json}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-16}
TAG=${TAG:-E27_avid_bird_640}
DUMP=${DUMP:-/root/rivermind-fs/birddrone/E27/preds.jsonl}
mkdir -p /root/rivermind-fs/birddrone/E27
rm -f "$DUMP"
PYTHONPATH=. /opt/conda/envs/rivermind/bin/python /tmp/eval_avid_bird.py \
  --weights YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt \
  --data /root/rivermind-fs/birddrone/data_avid_nc1_flat.yaml \
  --split val \
  --val-list "$LIST" \
  --labels-mode vis \
  --imgsz $IMGSZ \
  --batch $BATCH \
  --workers 4 \
  --device 0 \
  --tag $TAG \
  --dump-preds "$DUMP" \
  --out $OUT
