#!/bin/bash
# 打印 test 评测 json 的 metrics 字段（只读）
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
/opt/conda/envs/rivermind/bin/python - <<'PY'
import json
for name in ("v2c_R1_reg_early_s0", "v2c_E1_rgbonly_s0", "v2c_E3_ironly_lblir_s0"):
    p = f"ResTest/_test_eval/{name}_test.json"
    try:
        d = json.load(open(p))
    except Exception as e:
        print(name, "->", e); continue
    print(name, sorted(d.keys()))
    m = d.get("metrics")
    if isinstance(m, dict):
        print("   ", json.dumps(m, ensure_ascii=False)[:300])
PY
