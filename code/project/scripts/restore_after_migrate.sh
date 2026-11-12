#!/bin/bash
# 迁移后恢复 criteria.py 自愈钩子（sitecustomize 属容器层，重建会丢）
SITE=/opt/conda/envs/rivermind/lib/python3.10/site-packages
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
TARGET="$PROJ/ultralytics/nn/mm/pruning/criteria.py"
PERSIST=/root/rivermind-data/_persist/criteria.py

echo "=== 1. 确认持久母本 ==="
if [ -s "$PERSIST" ]; then
  echo "  母本 OK: $(wc -c < "$PERSIST") bytes"
else
  echo "  [错误] 母本缺失或为空，终止"
  exit 1
fi

echo ""
echo "=== 2. 写入 sitecustomize 自愈钩子 ==="
cat > "$SITE/sitecustomize.py" <<'PYEOF'
# -*- coding: utf-8 -*-
"""项目自愈钩子（防平台重建容器后基线覆盖）。

背景：本实例的基线镜像里 ultralytics/nn/mm/pruning/criteria.py 是 0 字节，
每次重建容器都会把项目文件回滚成该状态，导致
    ImportError: cannot import name 'IMPORTANCE_REGISTRY'
本钩子由 Python 解释器启动时自动 import（site 模块机制），
检查目标文件是否为 0 字节；若是，从持久母本恢复。开销仅一次 stat()。
"""
import os
import shutil

_PERSIST = "/root/rivermind-data/_persist/criteria.py"
_TARGETS = [
    "/3250410011/root/rivermind-data/MutilModel_3485988898/ultralytics/nn/mm/pruning/criteria.py",
]


def _heal():
    if not os.path.exists(_PERSIST) or os.path.getsize(_PERSIST) == 0:
        return
    for t in _TARGETS:
        try:
            if os.path.exists(t) and os.path.getsize(t) == 0:
                shutil.copyfile(_PERSIST, t)
        except Exception:
            pass


_heal()
PYEOF
echo "  已写入 $SITE/sitecustomize.py"

echo ""
echo "=== 3. 验证：人为清空后 Python 启动能否自愈 ==="
: > "$TARGET"
echo "  已清空 -> $(wc -c < "$TARGET") bytes"
cd "$PROJ" || exit 1
/opt/conda/envs/rivermind/bin/python - <<'PYEOF' 2>&1 | grep -vE "FutureWarning|warnings.warn|check_for_updates|timm|def forward|def backward|albumentations"
from ultralytics.nn.mm.pruning.criteria import IMPORTANCE_REGISTRY
print("  IMPORT OK, registry size =", len(IMPORTANCE_REGISTRY))
PYEOF
echo "  修复后: $(wc -c < "$TARGET") bytes"

echo ""
echo "=== 4. 关键模块导入贯通 ==="
/opt/conda/envs/rivermind/bin/python - <<'PYEOF' 2>&1 | grep -vE "FutureWarning|warnings.warn|check_for_updates|timm|def forward|def backward|albumentations"
from ultralytics.nn.mm.partition_fusion import ChannelPartitionFusion
from ultralytics.nn.mm.dual_loss import DualHeadFusionCriterion
from ultralytics.utils.tal import TaskAlignedAssigner
import inspect
print("  ChannelPartitionFusion OK")
print("  DualHeadFusionCriterion 有 tal_metric:", "tal_metric" in inspect.signature(DualHeadFusionCriterion.__init__).parameters)
print("  TaskAlignedAssigner 有 metric:", "metric" in inspect.signature(TaskAlignedAssigner.__init__).parameters)
PYEOF
