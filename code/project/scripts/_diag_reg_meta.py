import json
import statistics
from pathlib import Path

R = Path('/root/rivermind-data/uav_data_v2_reg')
m = json.load(open(R / '_reg_meta.json'))
print('mode     :', m['mode'])
print('oracle   :', m['oracle'])
print('direction:', m.get('direction'))
print('acceptance:', json.dumps(m.get('acceptance'), ensure_ascii=False))
print('summary  :', json.dumps(m.get('summary'), ensure_ascii=False))
seqs = m['sequences']
print('n_seq    :', len(seqs))
k0 = list(seqs)[0]
print('sample   :', k0, json.dumps(seqs[k0], ensure_ascii=False)[:400])

rep = json.load(open(R / '_fit_report.json'))
for k in ('mode', 'oracle', 'n_both_frames', 'frame_err_mean_px', 'frame_err_median_px',
          'frame_err_p95_px', 'frame_ok_frac_2px', 'scale_ratio_w_mean'):
    v = rep.get(k)
    print(f'rep.{k:22s} = {v}')

# 逐序列 rmse 分布
rows = []
for ln in (R / 'reg_quality.tsv').read_text().strip().splitlines()[1:]:
    p = ln.split('\t')
    if len(p) >= 9:
        try:
            rows.append((p[0], float(p[2]), float(p[5]), float(p[7]), p[9] if len(p) > 9 else ''))
        except ValueError:
            pass
rms = [r[1] for r in rows]
print(f'\n序列数={len(rows)}  fit_rmse: p50={statistics.median(rms):.2f} min={min(rms):.2f} max={max(rms):.2f}')
ok2 = sum(1 for v in rms if v <= 2.0)
print(f'fit_rmse <= 2px 的序列: {ok2}/{len(rows)}')
acc = sum(1 for r in rows if r[4].strip().lower() in ('true', '1'))
print(f'accept=True 的序列: {acc}/{len(rows)}')
