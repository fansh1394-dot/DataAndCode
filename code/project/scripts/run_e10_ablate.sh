#!/usr/bin/env bash
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
cd "$PROJ" || exit 1
"$PY" -u scripts/eval_split.py \
  --weights YOLO26/ResTest/mdp150_M0Img800_s0/weights/best.pt \
  --data /3250410011/root/rivermind-data/mendeley_split/data.yaml \
  --split test --labels-mode dual --imgsz 640 --batch 8 --workers 4 --device 0 \
  --tag e10_img800_at640 --out "$PROJ/ResTest/_test_eval/e10_img800_at640.json" \
  > logs/e10/img800_at640.log 2>&1
echo "AT640_DONE rc=$?" >> logs/e10/_progress.log

"$PY" -u scripts/eval_split.py \
  --weights YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt \
  --data /3250410011/root/rivermind-data/mendeley_split/data.yaml \
  --split test --labels-mode dual --imgsz 800 --batch 8 --workers 4 --device 0 \
  --tag e10_base_at800 --out "$PROJ/ResTest/_test_eval/e10_base_at800.json" \
  > logs/e10/base_at800.log 2>&1
echo "BASE800_DONE rc=$?" >> logs/e10/_progress.log
