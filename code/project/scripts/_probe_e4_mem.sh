#!/bin/bash
# E4 双头模型真实显存测量（batch 32，不降 batch）
# 走与 train_full.py 完全相同的构建路径：YOLOMM(yaml)，避免探针与真实训练不一致。
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "[gate] free=${FREE} MiB (need >= 20000 for batch32 dual-head)"
if [ "$FREE" -lt 20000 ]; then
  echo "[gate] 显存不足，退出，不启动测量"
  exit 7
fi

$PY - <<'PYEOF' 2>&1 | grep -avE 'FutureWarning|warnings.warn|check_for_updates|^\s*def |^\s*$'
import os, sys, torch, yaml, pathlib, tempfile
PROJ = '/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)

# 与链脚本一致：dual 标签口径
dp = pathlib.Path('/root/rivermind-data/uav_data_v2/data.yaml')
ys = yaml.safe_load(dp.read_text(encoding='utf-8'))
ys['labels'] = {'rgb': 'labels', 'ir': 'labels_ir'}
probe_yaml = pathlib.Path('/tmp/_probe_data_dual.yaml')
probe_yaml.write_text(yaml.safe_dump(ys, allow_unicode=True, sort_keys=False), encoding='utf-8')

from ultralytics import YOLOMM
cfg_yaml = 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
print('[build] YOLOMM(yaml) ...', flush=True)
model = YOLOMM(cfg_yaml)
m = model.model
print(f'[build] params={sum(p.numel() for p in m.parameters())/1e6:.3f}M  end2end={getattr(m,"end2end",None)}')

print('[train] 5 步，batch=32 imgsz=640 amp=False ...', flush=True)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
model.train(
    data=str(probe_yaml), epochs=1, batch=32, imgsz=640, workers=4,
    amp=False, device=0, val=False, plots=False, save=False, verbose=False,
    project='/tmp/_probe_mem', name='e4', exist_ok=True,
    cache=False, deterministic=True,
)
print(f'>>> RESULT peak_alloc = {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
print(f'>>> RESULT peak_resv  = {torch.cuda.max_memory_reserved()/2**30:.2f} GiB')
PYEOF
echo "[exit] $?"
