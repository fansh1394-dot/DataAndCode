#!/usr/bin/env bash
# E24c 评估链：DT800 配方的种子重复（seed 2）
set -u
cd /3250410011/root/rivermind-data || exit 1
PY=/opt/conda/envs/rivermind/bin/python
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
TAG=e24c_dt800s2
W=$PROJ/YOLO26/ResTest/mdp150_M0DT800_s2/weights/best.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 1) 逐帧框导出 ${TAG} ==="
WEIGHTS=$W TAG=$TAG IMGSZ=640 DEVICE=0 $PY pf_export.py 2>&1 | tail -n 12

echo "=== 2) 分桶 AP ==="
$PY bucket_ap.py --tag $TAG --out $PROJ/reports/bucket_${TAG}.json 2>&1 | tail -n 14

echo "=== 3) 时序指标 ==="
$PY $PROJ/scripts/temporal_consistency.py --tag $TAG 2>&1 | tail -n 16

echo "=== 4) 无 GT 帧误检 ==="
$PY $PROJ/scripts/dt800_negfp.py $TAG 2>&1 | tail -n 10
