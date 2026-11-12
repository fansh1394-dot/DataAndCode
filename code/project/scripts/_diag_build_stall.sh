#!/bin/bash
# 诊断：定位 YOLOMM(dual-yaml) 构建阶段到底卡在哪一行
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
pkill -f '_probe_step.py' 2>/dev/null
sleep 2

cat > /tmp/_diag_build.py <<'PYEOF'
import os, sys, time, torch
PROJ = '/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)
t0 = time.time()
def mark(s):
    print(f'[T+{time.time()-t0:6.1f}s] {s}', flush=True)

mark('import ultralytics')
from ultralytics import YOLOMM
mark('import done')
from ultralytics.nn.tasks import DualHeadFusionDetectV26
mark('import v26 head done')
cfg = 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
mark('YOLOMM(cfg) 开始')
m = YOLOMM(cfg)
mark(f'YOLOMM(cfg) 完成 params={sum(p.numel() for p in m.model.parameters())/1e6:.3f}M')
mark('cuda available')
print('cuda:', torch.cuda.is_available(), flush=True)
PYEOF

timeout 240 $PY -u /tmp/_diag_build.py 2>&1 | grep -aE '^\[T\+|^cuda:|Error|Traceback' 
echo "[exit] ${PIPESTATUS[0]}"
