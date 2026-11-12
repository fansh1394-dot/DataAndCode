#!/bin/bash
# A 方案：跨域召回率按目标尺寸分桶，域内/跨域对照
set -uo pipefail
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
W="$PROJ/ResTest/v2c_E1_rgbonly_s0/weights/best.pt"

cd "$PROJ" || exit 1
export MM_PROJ="$PROJ"

for CONF in 0.5 0.05; do
  for DOM in mendeley v2; do
    echo "############ domain=$DOM conf=$CONF ############"
    "$PY" -u scripts/_size_bucket_xdom.py --weights "$W" --domain "$DOM" \
          --conf "$CONF" --tag "${DOM}_c${CONF}" 2>&1 \
      | grep -vE 'Warning|warn|def |check_for|^\s*$' | tail -14
  done
done
