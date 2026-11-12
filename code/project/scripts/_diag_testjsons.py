import json
from pathlib import Path

P = Path('/3250410011/root/rivermind-data/MutilModel_3485988898/ResTest/_test_eval')
for name in ('v2c_R1_reg_early_s0_test.json', 'late', 'v2c_E1_rgbonly_s0_test.json',
             'v2c_E1_rgbonly_s1_test.json', 'v2c_E3_ironly_lblir_s0_test.json',
             'v2_E1_rgbonly_test.json', 'v2_C4_deform_test.json'):
    p = P / name
    if not p.exists():
        print(f'[MISS] {name}')
        continue
    d = json.load(open(p))
    print(f'=== {name} ===')
    for k, v in d.items():
        if isinstance(v, dict):
            print('  ', k, {kk: round(vv, 6) if isinstance(vv, float) else vv
                            for kk, vv in list(v.items())[:8]})
        elif not isinstance(v, list):
            print('  ', k, v)
