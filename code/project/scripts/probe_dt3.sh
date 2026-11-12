#!/bin/bash
PY=/opt/conda/envs/rivermind/bin/python
$PY - <<'PY'
import glob, os
from PIL import Image
fs = sorted(glob.glob('/3250410011/root/rivermind-data/mendeley_split/images/train/*'))[:8]
for f in fs:
    with Image.open(f) as im:
        print(os.path.basename(f), im.size)
fs2 = sorted(glob.glob('/3250410011/root/rivermind-data/mendeley_split/images_ir/train/*'))[:3]
for f in fs2:
    with Image.open(f) as im:
        print('IR', os.path.basename(f), im.size)
PY
echo "=== 训练脚本参数入口 ==="
grep -n "loss-box\|loss_box" train_full.py | head -20
echo "=== 是否有 distill 示例 yaml ==="
find . -name '*distill*.yaml' -not -path './.git/*' | head
echo "=== dual_loss distill 接入 ==="
grep -rn "distill" ultralytics/nn/mm/dual_loss.py | head -30
