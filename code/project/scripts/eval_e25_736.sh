#!/usr/bin/env bash
# E25 全测试集分桶：640 训练权重在 736 推理尺度下的 AP50-95 / AP75（补齐三点曲线的精度轴）
set -u
cd /3250410011/root/rivermind-data || exit 1
PY=/opt/conda/envs/rivermind/bin/python
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
TAG=e25_736
W=$PROJ/YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 1) 逐帧框导出 ${TAG}（imgsz=736） ==="
WEIGHTS=$W TAG=$TAG IMGSZ=736 DEVICE=0 $PY pf_export.py 2>&1 | tail -n 15

echo "=== 2) 分桶 AP ==="
$PY bucket_ap.py --tag $TAG --out $PROJ/reports/bucket_${TAG}.json 2>&1 | tail -n 15

echo "=== 3) 时序指标 ==="
$PY $PROJ/scripts/temporal_consistency.py --tag $TAG 2>&1 | tail -n 16
