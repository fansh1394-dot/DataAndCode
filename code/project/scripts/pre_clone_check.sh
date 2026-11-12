#!/bin/bash
# 最终确认：无训练进程、s1 ckpt 可续训、显存与产物齐备（克隆前快照）
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python

echo "=========== 进程 ==========="
pgrep -af '[t]rain_full.py' || echo "no train_full"
pgrep -af '[p]hase0_chain.sh' || echo "no chain"
echo "=========== GPU ==========="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=========== s1 冻结态 ==========="
"$PY" - <<'PY' 2>&1 | grep -E '^\[chk\]'
import sys
sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")
import torch
p = "/3250410011/root/rivermind-data/MutilModel_3485988898/ResTest/v2c_E1_rgbonly_s1/weights/last.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
print(f"[chk] epoch(0-based)={ck.get('epoch')}  -> resume 将从 epoch {ck.get('epoch')+1} 继续")
print(f"[chk] optimizer={'OK' if ck.get('optimizer') else 'MISSING'}  ema={'OK' if ck.get('ema') is not None else 'MISSING'}")
print(f"[chk] model inline={'OK' if ck.get('model') is not None else 'MISSING'}  best_fitness={ck.get('best_fitness')}")
print(f"[chk] train_args.data={ck['train_args'].get('data')}")
print(f"[chk] 文件大小={__import__('os').path.getsize(p)/1e6:.1f} MB")
PY
echo "=========== 产物齐备性 ==========="
ls -l --time-style=+%m-%d_%H:%M ResTest/v2c_E1_rgbonly_s1/weights/ | tail -4
echo "results.csv 最后两行:"
tail -2 ResTest/v2c_E1_rgbonly_s1/results.csv | cut -c1-100
echo "args.yaml data 行: $(grep '^data:' ResTest/v2c_E1_rgbonly_s1/args.yaml)"
echo "=========== 已完成的基线 test 结果 ==========="
ls -1 ResTest/_test_eval/*.json 2>/dev/null
