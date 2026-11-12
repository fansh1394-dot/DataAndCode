#!/bin/bash
# 诊断 YOLOMM(dual yaml) 构建卡死在哪一层
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
cat > /tmp/_diag_build2.py <<'PYEOF'
import os, sys, time
PROJ = '/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)
t0 = time.time()
def mark(s): print(f'[T+{time.time()-t0:6.1f}s] {s}', flush=True)

mark('import YOLOMM')
from ultralytics import YOLOMM
mark('import DetectionModel / parse')
from ultralytics.nn.tasks import DetectionModel, DualHeadFusionDetectV26
from ultralytics.nn.mm.dual_head_v26 import DualHeadFusionDetectV26 as V26b
mark(f'DualHeadFusionDetectV26 来自 tasks: {DualHeadFusionDetectV26}')
mark(f'DualHeadFusionDetectV26 来自 mm:    {V26b}')
mark(f'同一对象: {DualHeadFusionDetectV26 is V26b}')

cfg = 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
mark('DetectionModel(cfg) 直接构建 ...')
m = DetectionModel(cfg=cfg, ch=6, nc=1, verbose=False)
mark(f'DetectionModel 完成 params={sum(p.numel() for p in m.parameters())/1e6:.3f}M')
PYEOF
timeout 300 $PY -u /tmp/_diag_build2.py 2>&1 | grep -aE '^\[T\+|Error|Traceback|line ' | head -40
echo "[rc] ${PIPESTATUS[0]}"
