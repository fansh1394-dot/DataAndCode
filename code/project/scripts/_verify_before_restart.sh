#!/bin/bash
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
$PY - <<'EOF'
import ast
ast.parse(open('train_full.py', encoding='utf-8').read())
print('train_full.py syntax OK')
EOF
bash -n YOLO26/scripts/v26_chain2b.sh && echo 'chain syntax OK'
echo "--- --plots 是否进入 parser ---"
grep -n '\-\-plots' train_full.py | head -5
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
