#!/bin/bash
D=/3250410011/root/rivermind-data/mendeley_split
echo "=== label sample (rgb) ==="
f=$(ls $D/labels/train | head -1); echo "$f"; cat "$D/labels/train/$f"
echo "=== label sample (ir) ==="
g=$(ls $D/labels_ir/train | head -1); echo "$g"; cat "$D/labels_ir/train/$g"
echo "=== metadata train head ==="
python - <<'PY'
import json
p='/3250410011/root/rivermind-data/mendeley_split/metadata/train.json'
d=json.load(open(p))
print(type(d), len(d))
if isinstance(d, dict):
    ks=list(d.keys())[:2]
    for k in ks: print(k, d[k])
else:
    print(d[:2])
PY
echo "=== perframe dir listing ==="
ls -la perframe_boxes2 | head -30
echo "=== pf2_run.sh ==="
ls scripts/pf2_run.sh 2>/dev/null && cat scripts/pf2_run.sh
echo "=== bucket eval scripts ==="
ls scripts/ | grep -iE 'bucket|sizebucket' | head -20
