#!/bin/bash
# 扫描在线估计窗口 N 对 P1/P2 精度的影响
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
PY=/opt/conda/envs/rivermind/bin/python
SPLIT=${1:-test}
for N in 4 8 16 32 64; do
  echo "########## N=$N ##########"
  $PY scripts/_verify_align_oracle.py --split "$SPLIT" --fit-n "$N" 2>&1 | grep -E '解析视差|P0 |P1 |P2 |可用仿射|std\(A\) 中位|拟合残差中位'
done
