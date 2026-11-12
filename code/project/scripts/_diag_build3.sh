#!/bin/bash
# 诊断 YOLOMM 包装层为何在 dual yaml 上卡死（DetectionModel 直构 0.3s 成功）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
cat > /tmp/_diag_build3.py <<'PYEOF'
import os, sys, time, faulthandler
PROJ = '/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)
faulthandler.dump_traceback_later(60, exit=True)   # 60s 后打印栈并退出
t0 = time.time()
def mark(s): print(f'[T+{time.time()-t0:6.1f}s] {s}', flush=True)

mark('import YOLOMM')
from ultralytics import YOLOMM
mark('YOLOMM(cfg) 开始')
cfg = 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
m = YOLOMM(cfg)
mark('YOLOMM(cfg) 完成')
PYEOF
timeout 120 $PY -u /tmp/_diag_build3.py 2>&1 | tail -40
echo "[rc] ${PIPESTATUS[0]}"
