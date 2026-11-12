import csv
import sys

p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
print('epochs', len(rows))
keys = [c for c in rows[0] if 'fuse' in c and 'mAP50-95' in c and 'metrics/' in c]
print('cols', keys)
if keys:
    k = keys[0]
    v = [(int(float(r['epoch'])), float(r[k])) for r in rows if r.get(k) not in (None, '')]
    v.sort(key=lambda t: -t[1])
    print('top5', v[:5])
    print('last', v[-1])
    print('ep50', [x for x in v if x[0] == 50])
