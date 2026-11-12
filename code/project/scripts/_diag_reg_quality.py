import json
from pathlib import Path

R = Path('/root/rivermind-data/uav_data_v2_reg')
rep = json.load(open(R / '_fit_report.json'))
print('=== _fit_report.json keys ===', list(rep)[:10] if isinstance(rep, dict) else type(rep))
if isinstance(rep, dict):
    for k, v in list(rep.items())[:3]:
        print(f'  {k}: {str(v)[:300]}')
meta = json.load(open(R / '_reg_meta.json'))
print('=== _reg_meta keys ===', list(meta)[:10])
if isinstance(meta, dict):
    seqs = meta.get('sequences') or meta
    n_ok = 0
    rmses = []
    tot = 0
    for k, v in (seqs.items() if isinstance(seqs, dict) else []):
        if not isinstance(v, dict):
            continue
        tot += 1
        rm = v.get('rmse_px')
        if rm is not None:
            rmses.append(rm)
            if rm <= 2.0:
                n_ok += 1
    print(f'  序列数={tot}  rmse<=2px 的序列={n_ok}')
    if rmses:
        import statistics
        print(f'  rmse p50={statistics.median(rmses):.3f}  min={min(rmses):.3f}  max={max(rmses):.3f}')
print('=== reg_quality.tsv (head) ===')
lines = (R / 'reg_quality.tsv').read_text().strip().splitlines()
for ln in lines[:8]:
    print('  ', ln)
print(f'  ... 共 {len(lines)} 行')
