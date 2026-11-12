#!/usr/bin/env bash
# E19 评估链：逐帧框导出 → 分桶 AP → 时序一致性
set -u
cd /3250410011/root/rivermind-data || exit 1
PY=/opt/conda/envs/rivermind/bin/python
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
TAG=e19crop
W=$PROJ/YOLO26/ResTest/mdp150_M0E19crop_s0/weights/best.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 1) 逐帧框导出 ${TAG} ==="
WEIGHTS=$W TAG=$TAG IMGSZ=640 DEVICE=0 $PY pf_export.py 2>&1 | tail -n 12

echo "=== 2) 分桶 AP ==="
$PY bucket_ap.py --tag $TAG --out $PROJ/reports/bucket_${TAG}.json 2>&1 | tail -n 12

echo "=== 3) 时序指标 ==="
$PY $PROJ/scripts/temporal_consistency.py --tag $TAG 2>&1 | tail -n 14
