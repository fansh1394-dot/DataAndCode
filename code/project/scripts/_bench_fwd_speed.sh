#!/bin/bash
# 测一次双头前向的墙钟（判断是慢还是死锁）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cat > /tmp/_bench_fwd.py <<'PYEOF'
import os, sys, time, torch
PROJ='/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)
t0=time.time()
def mark(s): print(f'[T+{time.time()-t0:7.1f}s] {s}', flush=True)
from ultralytics.nn.tasks import DetectionModel
mark('import done')
cfg='ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
m = DetectionModel(cfg=cfg, ch=6, nc=1, verbose=False).cuda().eval()
mark('model cuda ok')
x = torch.zeros(32, 6, 640, 640, device='cuda')
torch.cuda.synchronize(); mark('input ready')
with torch.no_grad():
    for i in range(3):
        y = m(x); torch.cuda.synchronize()
        mark(f'forward {i+1} done  peak={torch.cuda.max_memory_allocated()/2**30:.2f}G')
mark('done')
PYEOF
timeout 600 $PY -u /tmp/_bench_fwd.py 2>&1 | grep -aE '^\[T\+|Error|Traceback'
echo "[rc] ${PIPESTATUS[0]}"
