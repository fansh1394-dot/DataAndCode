import json
import random
from pathlib import Path

import numpy as np

D = Path('/root/rivermind-data/uav_data_v2')
frames = json.load(open(D / 'metadata/train.json'))['frames']
both = [s for s, v in frames.items() if v['kind'] == 'both']
print('both frames', len(both))
random.seed(0)
sel = random.sample(both, 300)
for s in sel[:3]:
    print(s, (D / 'labels' / (s + '.txt')).exists(), (D / 'labels_ir' / (s + '.txt')).exists())
print('sample files:', [p.name for p in list((D / 'labels').glob('*.txt'))[:3]])
print('sample ir files:', [p.name for p in list((D / 'labels_ir').glob('*.txt'))[:3]])
stats = []
for s in sel:
    p1 = D / 'labels' / 'train' / (s + '.txt')
    p2 = D / 'labels_ir' / 'train' / (s + '.txt')
    if not (p1.exists() and p2.exists()):
        continue
    try:
        a = np.loadtxt(p1, ndmin=2)
        c = np.loadtxt(p2, ndmin=2)
    except Exception:
        continue
    if a.ndim != 2 or c.ndim != 2 or len(a) != len(c) or len(a) != 1:
        continue
    stats.append([a[0][1] - c[0][1], a[0][2] - c[0][2]])
st = np.array(stats)
print('n', len(st))
if len(st):
    print('dx norm mean %.4f std %.4f  -> %.1f px(@640)' % (st[:, 0].mean(), st[:, 0].std(), st[:, 0].std() * 640))
    print('dy norm mean %.4f std %.4f  -> %.1f px(@512)' % (st[:, 1].mean(), st[:, 1].std(), st[:, 1].std() * 512))
    for name, col in (('dx', 0), ('dy', 1)):
        v = np.abs(st[:, col])
        print('%s |.| p50 %.4f p75 %.4f p90 %.4f p99 %.4f' % (name, *np.percentile(v, [50, 75, 90, 99])))
