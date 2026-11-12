"""读原始 json 的 gt_rect / exist，与导出后的 labels 对齐，定位 y 偏移的来源。"""
import json
import random
from pathlib import Path

import numpy as np

SRC = Path('/root/rivermind-data/uav_data_origin')  # 若不存在则用自动查找
META = Path('/root/rivermind-data/uav_data_v2/metadata/train.json')
D = Path('/root/rivermind-data/uav_data_v2')

frames = json.load(open(META))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(1)
sel = random.sample(both, 300)

# 找原始数据根
cands = [SRC]
cands += [p for p in Path('/root/rivermind-data').iterdir() if p.is_dir()]
root = None
for c in cands:
    if (c / 'train').exists() or any((c / n).exists() for n in ('train', 'val', 'test')):
        for split in ('train', 'val', 'test'):
            sp = c / split
            if sp.is_dir():
                subs = [p for p in sp.iterdir() if p.is_dir()]
                if subs and (subs[0] / 'infrared.json').exists():
                    root = c
                    break
    if root:
        break
print('source root:', root)

if root is None:
    raise SystemExit('source not found')

cache = {}
out = []
for s in sel:
    seq = '_'.join(s.split('_')[:4])
    idx = int(s.split('_')[4])
    if seq not in cache:
        found = None
        for split in ('train', 'val', 'test'):
            p = root / split / seq / 'infrared.json'
            if p.exists():
                found = json.loads(p.read_text())
                vp = root / split / seq / 'visible.json'
                vfound = json.loads(vp.read_text()) if vp.exists() else None
                break
        cache[seq] = (found, vfound)
    irj, visj = cache[seq]
    if irj is None:
        continue
    try:
        ir = np.loadtxt(D / 'labels_ir/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
        vb = np.loadtxt(D / 'labels/train' / (s + '.txt'), ndmin=2).reshape(-1, 5)[0]
    except Exception:
        continue
    r = irj['gt_rect'][idx] if idx < len(irj.get('gt_rect') or []) else None
    if not r or len(r) < 4:
        continue
    x, y, w, h = [float(v) for v in r[:4]]
    # 原始尺寸：ir 视频 640x512；导出画布同为 640x512
    out.append((s, x, y, w, h, ir[1] * 640, ir[2] * 512, vb[2] * 512))
res = out
print('n', len(res))
arr = np.array([[o[1], o[2], o[5], o[6]] for o in res])
print('raw cx mean %.1f  label_ir cx mean %.1f  (diff %.2f px)' % (arr[:, 0].mean(), arr[:, 2].mean(), (arr[:, 2] - arr[:, 0]).mean()))
print('raw cy mean %.1f  label_ir cy mean %.1f  (diff %.2f px)' % (arr[:, 1].mean(), arr[:, 3].mean(), (arr[:, 3] - arr[:, 1]).mean()))
# 逐帧一致性
dx = np.abs(arr[:, 2] - arr[:, 0]); dy = np.abs(arr[:, 3] - arr[:, 1])
print('label vs raw: |dx| p50 %.2f max %.2f ; |dy| p50 %.2f max %.2f' % (
    np.percentile(dx, 50), dx.max(), np.percentile(dy, 50), dy.max()))
for o in res[:5]:
    print(o[0], 'raw cy %.1f' % o[2], 'ir_label cy %.1f' % o[6], 'rgb_label cy %.1f' % o[7])
