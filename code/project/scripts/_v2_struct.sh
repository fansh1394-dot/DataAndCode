#!/bin/bash
# 查看 v2 数据集结构（供 DIV-A 导出脚本参考）
set -uo pipefail
R=/root/rivermind-data/uav_data_v2
PY=/opt/conda/envs/rivermind/bin/python

echo "=== 顶层 ==="
ls "$R"
echo
echo "=== data.yaml ==="
cat "$R/data.yaml"
echo
echo "=== metadata 结构 ==="
"$PY" - "$R" <<'PY'
import json, sys, collections
R = sys.argv[1]
d = json.load(open(f'{R}/metadata/train.json'))
print('top keys:', list(d.keys()))
k = list(d['frames'])[0]
print('sample frame:', k, d['frames'][k])
c = collections.Counter(v['sequence'] for v in d['frames'].values())
print('n_seqs:', len(c), 'frames:', len(d['frames']))
print('seq sample:', list(c.items())[:3])
PY
echo
echo "=== 目录 ==="
for d in images images_ir labels labels_ir; do
  echo -n "  $d/train: "
  ls "$R/$d/train" 2>/dev/null | head -2 | tr '\n' ' '
  echo " (count=$(ls "$R/$d/train" 2>/dev/null | wc -l))"
done
