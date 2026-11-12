#!/bin/bash
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
echo "===== s2 权重 ====="
ls -la YOLO26/ResTest/v26_E1_rgbonly_s2/weights/
echo "===== s2 后 5 轮 ====="
cut -d, -f1,8,9 YOLO26/ResTest/v26_E1_rgbonly_s2/results.csv | tail -5
echo
echo "===== 加载 s2 best.pt ====="
cat > /tmp/_load_s2.py <<'PYEOF'
import sys
sys.path.insert(0, '/3250410011/root/rivermind-data/MutilModel_3485988898')
from ultralytics import YOLOMM
m = YOLOMM('/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest/v26_E1_rgbonly_s2/weights/best.pt')
print('load OK:', type(m.model).__name__)
PYEOF
timeout 300 $PY -u /tmp/_load_s2.py 2>&1 | grep -aE 'load OK|Error|Traceback'
echo "[rc] ${PIPESTATUS[0]}"
