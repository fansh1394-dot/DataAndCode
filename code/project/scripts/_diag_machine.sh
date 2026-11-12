#!/bin/bash
# 区分 CPU 卡死 vs GPU 卡死 + 测 I/O 带宽
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
echo "===== 磁盘 I/O（读 200 张图）====="
timeout 120 $PY -c "
import time,glob
fs=sorted(glob.glob('/root/rivermind-data/uav_data_v2/images/train/*.jpg'))[:200]
t=time.time()
n=0
for f in fs:
    n+=len(open(f,'rb').read())
dt=time.time()-t
print(f'files={len(fs)} bytes={n/1e6:.1f}MB time={dt:.2f}s -> {n/1e6/dt:.1f} MB/s')
" 2>&1 | tail -3
echo
echo "===== CPU 单核算力（轻量）====="
timeout 60 $PY -c "
import time
t=time.time(); s=0
for i in range(20_000_000): s+=i
print(f'20M loop: {time.time()-t:.2f}s')
" 2>&1 | tail -2
echo
echo "===== GPU 冒烟（纯 GEMM）====="
timeout 120 $PY -c "
import torch,time
a=torch.randn(4096,4096,device='cuda'); b=torch.randn(4096,4096,device='cuda')
torch.cuda.synchronize(); t=time.time()
for _ in range(20): c=a@b
torch.cuda.synchronize(); dt=time.time()-t
print(f'20x4096^3 {dt:.2f}s -> {20*2*4096**3/dt/1e12:.2f} TFLOPS')
" 2>&1 | tail -2
echo
echo "===== 当前负载 ====="
uptime
nproc
