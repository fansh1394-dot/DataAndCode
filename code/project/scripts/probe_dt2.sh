#!/bin/bash
PY=/opt/conda/envs/rivermind/bin/python
D=/3250410011/root/rivermind-data/mendeley_split
echo "=== metadata train ==="
$PY - <<'PY'
import json
p='/3250410011/root/rivermind-data/mendeley_split/metadata/train.json'
d=json.load(open(p))
print(type(d), len(d))
if isinstance(d, dict):
    for k in list(d)[:2]: print(k, d[k])
else:
    print(d[:3])
PY
echo "=== bucket script head ==="
head -40 scripts/_size_bucket_xdom.py
echo "=== run_sizebucket ==="
cat scripts/_run_sizebucket.sh
echo "=== bucket_s0.json ==="
cat perframe_boxes2/bucket_s0.json
echo "=== boxes_s0 sample ==="
$PY - <<'PY'
import json
d=json.load(open('/3250410011/root/rivermind-data/MutilModel_3485988898/perframe_boxes2/boxes_s0.json'))
ks=list(d)[:2]
print(len(d))
for k in ks:
    print(k, {b:(len(v[b]['xyxy']),[c for c in v[b]['conf'][:5]]) for b in v})
PY
