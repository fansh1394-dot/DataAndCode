#!/bin/bash
# 定位 YOLOMM 双头模型第 1 步 forward 卡死原因：看完整栈 + GPU 是否被真占用
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
echo "===== 完整栈（faulthandler 输出）====="
sed -n '/Thread 0x/,/^$/p' /tmp/_probe_final.log | head -45
echo
echo "===== 日志尾部原文 ====="
tail -12 /tmp/_probe_final.log
echo
echo "===== GPU 现状 ====="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo
echo "===== 容器内 GPU 可见性 ====="
/opt/conda/envs/rivermind/bin/python -c "import torch; print('avail', torch.cuda.is_available()); print('cnt', torch.cuda.device_count())" 2>&1 | tail -3
