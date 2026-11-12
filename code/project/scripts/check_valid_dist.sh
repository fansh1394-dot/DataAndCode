#!/bin/bash
# 核对 valid_rgb / valid_ir 的帧级分布（决定单模态基线的评测口径是否可比）
D=/root/rivermind-data/uav_data_v2
F=$D/metadata/val.json
/opt/conda/envs/rivermind/bin/python - "$F" <<'PYEOF'
import json, sys
p = sys.argv[1]
raw = json.load(open(p))
frames = raw.get('frames', raw)
from collections import Counter
c_valid = Counter()
c_kind = Counter()
for k, v in frames.items():
    if not isinstance(v, dict):
        continue
    c_valid[(bool(v.get('valid_rgb')), bool(v.get('valid_ir')))] += 1
    c_kind[v.get('kind')] += 1
print("总帧数:", len(frames))
print("\nkind 分布:", dict(c_kind))
print("\n(valid_rgb, valid_ir) 分布:")
for k, n in sorted(c_valid.items(), key=lambda x: -x[1]):
    print(f"  rgb={k[0]!s:<5} ir={k[1]!s:<5} -> {n}")
PYEOF
