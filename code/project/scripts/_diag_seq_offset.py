import json
import random
from collections import defaultdict

import numpy as np

D = '/root/rivermind-data/uav_data_v2/'
frames = json.load(open(D + 'metadata/train.json'))['frames']
both = sorted([s for s, v in frames.items() if v['kind'] == 'both'])
random.seed(1)
sel = random.sample(both, 2000)
per_seq = defaultdict(list)
for s in sel:
    try:
        a = np.loadtxt(D + 'labels/train/' + s + '.txt', ndmin=2).reshape(-1, 5)
        c = np.loadtxt(D + 'labels_ir/train/' + s + '.txt', ndmin=2).reshape(-1, 5)
    except Exception:
        continue
    if a.shape[0] != 1 or c.shape[0] != 1:
        continue
    seq = '_'.join(s.split('_')[:3])  # yyyymmdd_hhmmss_L
    per_seq[seq].append((a[0][1] - c[0][1], a[0][2] - c[0][2]))
print('seqs', len(per_seq), 'key sample', list(per_seq)[:2])
rows = []
for seq, v in per_seq.items():
    if len(v) < 5:
        continue
    arr = np.array(v)
    rows.append((seq, len(v), arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 1].std()))
rows.sort(key=lambda r: -r[2])
print(f"{'seq':28s} {'n':>4s} {'dx':>8s} {'dy':>8s} {'dy_std':>8s}")
for r in rows[:12]:
    print(f"{r[0]:28s} {r[1]:4d} {r[2]:8.4f} {r[3]:8.4f} {r[4]:8.4f}")
print('...')
for r in rows[-5:]:
    print(f"{r[0]:28s} {r[1]:4d} {r[2]:8.4f} {r[3]:8.4f} {r[4]:8.4f}")
dys = np.array([r[3] for r in rows])
print('seq-level dy: p10 %.4f p50 %.4f p90 %.4f' % tuple(np.percentile(dys, [10, 50, 90])))
print('seq-level |dy|<=0.03 frac = %.3f' % float(np.mean(np.abs(dys) <= 0.03)))
