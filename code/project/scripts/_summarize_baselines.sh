#!/bin/bash
# 汇总当前所有 test 正式口径结果，并做尺度分桶分析
set -uo pipefail
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
cd "$PROJ" || exit 1

echo "=== 全部 test 正式口径结果 ==="
"$PY" - <<'PY'
import json, glob, os
base = '/3250410011/root/rivermind-data/MutilModel_3485988898/ResTest/_test_eval'
rows = []
for p in sorted(glob.glob(os.path.join(base, '*.json'))):
    d = json.load(open(p))
    m = d.get('metrics', {})
    tag = d.get('tag', os.path.basename(p))
    lm = d.get('labels_mode', '?')
    def g(k):
        return m.get(f'metrics/{k}/metrics/mAP50-95(B)')
    br = {b: g(b) for b in ('rgb', 'ir', 'fuse', 'fuse_ir')
          if f'metrics/{b}/metrics/mAP50-95(B)' in m}
    overall = m.get('metrics/mAP50-95(B)')
    p50 = m.get('metrics/mAP50(B)')
    prec = m.get('metrics/precision(B)')
    rec = m.get('metrics/recall(B)')
    rows.append((tag, lm, overall, p50, prec, rec, br))
print(f"{'tag':<34}{'labels':<8}{'mAP50-95':>10}{'mAP50':>9}{'P':>8}{'R':>8}  branches")
for t, lm, o, p50, pr, rc, br in sorted(rows, key=lambda x: -(x[2] or 0)):
    print(f"{t:<34}{lm:<8}{(o or 0):>10.6f}{(p50 or 0):>9.4f}"
          f"{(pr or 0):>8.4f}{(rc or 0):>8.4f}  {br if br else ''}")
PY

echo
echo "=== 各 run 的最终 val 指标（训练期 final_validation）==="
"$PY" - <<'PY'
import json, glob, os
base = '/3250410011/root/rivermind-data/MutilModel_3485988898/ResTest'
for d in sorted(glob.glob(os.path.join(base, 'v2c_*'))):
    f = os.path.join(d, 'final_validation.json')
    if not os.path.isfile(f):
        continue
    try:
        j = json.load(open(f))
    except Exception:
        continue
    print(f"  {os.path.basename(d):<30} {json.dumps(j, ensure_ascii=False)[:200]}")
PY
