#!/usr/bin/env bash
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
mkdir -p "$PROJ/logs/perframe"
cd "$PROJ" || exit 1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /opt/conda/envs/rivermind/bin/python -u ratio_pad_capture.py > logs/perframe/ratio_pad.log 2>&1
echo "RATIO_PAD_DONE rc=$? $(date '+%F %T')" >> logs/perframe/ratio_pad.log
