#!/usr/bin/env bash
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
mkdir -p "$PROJ/logs/e10" "$PROJ/ResTest/_test_eval"
cd "$PROJ" || exit 1
"$PY" -u scripts/eval_split.py \
  --weights YOLO26/ResTest/mdp150_M0Img800_s0/weights/best.pt \
  --data /3250410011/root/rivermind-data/mendeley_split/data.yaml \
  --split test --labels-mode dual --imgsz 800 --batch 8 --workers 4 --device 0 \
  --tag e10_img800_test --out "$PROJ/ResTest/_test_eval/e10_img800_test.json" \
  > logs/e10/img800_test.log 2>&1
echo "E10_TEST_DONE rc=$? $(date '+%F %T')" >> logs/e10/_progress.log
