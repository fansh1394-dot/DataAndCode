#!/bin/bash
# E4 双头模型显存实测（最终版）
#   1) 杀死残留 probe 进程
#   2) 复刻 trainer 的 setup 流程，只跑 5 个训练步
#   3) 每步打印 torch 显存；同时打印 nvidia-smi 的真实占用
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f '_probe_e4_step.py' 2>/dev/null
sleep 3

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "[gate] free=${FREE} MiB"
[ "$FREE" -lt 20000 ] && { echo "[gate] 显存不足，退出"; exit 7; }

cat > /tmp/_probe_step.py <<'PYEOF'
import os, sys, torch, yaml, pathlib
PROJ = '/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)

dp = pathlib.Path('/root/rivermind-data/uav_data_v2/data.yaml')
ys = yaml.safe_load(dp.read_text(encoding='utf-8'))
ys['labels'] = {'rgb': 'labels', 'ir': 'labels_ir'}
probe_yaml = pathlib.Path('/tmp/_probe_data_dual.yaml')
probe_yaml.write_text(yaml.safe_dump(ys, allow_unicode=True, sort_keys=False), encoding='utf-8')

import ultralytics
from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer

# 关键：模仿 train_full.py 的调用，但把 _do_train 换成只跑 5 步
_orig_do_train = MultiModalDetectionTrainer._do_train

def _short_do_train(self, world_size=1):
    torch.cuda.reset_peak_memory_stats()
    print('[setup] _setup_train ...', flush=True)
    self._setup_train(world_size)
    print(f'[setup] done, loader={len(self.train_loader)} batches', flush=True)
    self.optimizer.zero_grad()
    for i, batch in enumerate(self.train_loader):
        batch = self.preprocess_batch(batch)
        loss, self.loss_items = self.compute_batch_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        print(f'[step {i+1}] alloc={torch.cuda.memory_allocated()/2**30:.2f}G '
              f'peak={torch.cuda.max_memory_allocated()/2**30:.2f}G '
              f'resv={torch.cuda.memory_reserved()/2**30:.2f}G', flush=True)
        if i + 1 >= 5:
            break
    print(f'>>> RESULT peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    print(f'>>> RESULT peak_resv ={torch.cuda.max_memory_reserved()/2**30:.2f} GiB')
    raise SystemExit(0)

MultiModalDetectionTrainer._do_train = _short_do_train

from ultralytics import YOLOMM
cfg_yaml = 'ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml'
print('[build] YOLOMM(yaml) ...', flush=True)
model = YOLOMM(cfg_yaml)
n = sum(p.numel() for p in model.model.parameters())
print(f'[build] params={n/1e6:.3f}M', flush=True)

model.train(
    data=str(probe_yaml), epochs=1, batch=32, imgsz=640, workers=4,
    amp=False, device=0, val=False, plots=False, save=False, verbose=False,
    project='/tmp/_probe_mem', name='e4', exist_ok=True, deterministic=True,
)
PYEOF

$PY -u /tmp/_probe_step.py > /tmp/_probe_step.log 2>&1 &
PID=$!
echo "[run] pid=$PID"

for t in $(seq 1 40); do
  sleep 15
  if ! kill -0 $PID 2>/dev/null; then break; fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  [t=${t}0s] gpu_used=${USED} MiB  $(grep -a 'step ' /tmp/_probe_step.log | tail -1)"
  if grep -qa 'RESULT peak_alloc' /tmp/_probe_step.log; then break; fi
done

echo "===== 关键行 ====="
grep -aE '^\[build|^\[setup|^\[step|^>>> RESULT' /tmp/_probe_step.log
echo "===== 异常（若有）====="
grep -aE 'OutOfMemory|Traceback' -A3 /tmp/_probe_step.log | head -20
wait $PID 2>/dev/null
echo "[exit] $?"
