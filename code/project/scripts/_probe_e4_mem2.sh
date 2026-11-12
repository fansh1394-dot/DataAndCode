#!/bin/bash
# E4 双头模型显存实测：复用 train_full.py 的 --resume/--epochs 之外的最小路径——
# 直接调 YOLOMM.train(..., epochs=1) 并在 trainer 里挂 step 钩子，只跑 5 步就中止。
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
export PYTORCH_CUDA_ALLOC_CONF=extendable_segments:True 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "[gate] free=${FREE} MiB (need >= 20000)"
[ "$FREE" -lt 20000 ] && { echo "[gate] 显存不足，退出"; exit 7; }

cat > /tmp/_probe_e4_step.py <<'PYEOF'
import os, sys, torch, yaml, pathlib
PROJ = '/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)

dp = pathlib.Path('/root/rivermind-data/uav_data_v2/data.yaml')
ys = yaml.safe_load(dp.read_text(encoding='utf-8'))
ys['labels'] = {'rgb': 'labels', 'ir': 'labels_ir'}
probe_yaml = pathlib.Path('/tmp/_probe_data_dual.yaml')
probe_yaml.write_text(yaml.safe_dump(ys, allow_unicode=True, sort_keys=False), encoding='utf-8')

# ---- 挂钩子：每步打印显存，第 5 步抛异常中止 ----
import ultralytics.engine.trainer as T
_orig = T.BaseTrainer._do_train
N_STEP = [0]

class _Stop(Exception):
    pass

def _patched(self, world_size=1):
    """替换 _do_train：只跑 5 个训练步，打印显存后退出。"""
    torch.cuda.reset_peak_memory_stats()
    self._setup_train(world_size)
    nbs = 32
    self.optimizer.zero_grad()
    for i, batch in enumerate(self.train_loader):
        batch = self.preprocess_batch(batch)
        loss, self.loss_items = self.compute_batch_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        N_STEP[0] += 1
        print(f'[step {N_STEP[0]}] alloc={torch.cuda.memory_allocated()/2**30:.2f}G '
              f'peak={torch.cuda.max_memory_allocated()/2**30:.2f}G '
              f'resv={torch.cuda.memory_reserved()/2**30:.2f}G', flush=True)
        if i + 1 >= 5:
            break
    print(f'>>> RESULT peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    print(f'>>> RESULT peak_resv ={torch.cuda.max_memory_reserved()/2**30:.2f} GiB')
    raise _Stop()

T.BaseTrainer._do_train = _patched

from ultralytics import YOLOMM
cfg_yaml = 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
print('[build] YOLOMM(yaml) ...', flush=True)
model = YOLOMM(cfg_yaml)
print(f'[build] params={sum(p.numel() for p in model.model.parameters())/1e6:.3f}M', flush=True)

try:
    model.train(
        data=str(probe_yaml), epochs=1, batch=32, imgsz=640, workers=4,
        amp=False, device=0, val=False, plots=False, save=False, verbose=False,
        project='/tmp/_probe_mem', name='e4', exist_ok=True, deterministic=True,
    )
except _Stop:
    print('[done] 5 步测量完成')
PYEOF

$PY /tmp/_probe_e4_step.py 2>&1 | tr '\r' '\n' | grep -aE '^\[gate|^\[build|^\[step|^>>> RESULT|^\[done|Error|Traceback|OutOfMemory' 
echo "[exit] ${PIPESTATUS[0]}"
