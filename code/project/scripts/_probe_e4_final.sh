#!/bin/bash
# E4 显存实测：构建 + 5 步训练，全程 faulthandler 兜底，每步打印 torch/nvidia 显存
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1
PY=/opt/conda/envs/rivermind/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 注意：不能用 pkill -f '_probe'，它会把本脚本自己和刚生成的 /tmp/_probe_*.py 一起杀掉
pkill -f '/tmp/_probe_step.py' 2>/dev/null
pkill -f '/tmp/_probe_e4_step.py' 2>/dev/null
sleep 3

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "[gate] free=${FREE} MiB"
[ "$FREE" -lt 20000 ] && { echo "[gate] 不足 20000，退出"; exit 7; }

cat > /tmp/_probe_final.py <<'PYEOF'
import os, sys, time, torch, yaml, pathlib, faulthandler
PROJ = '/3250410011/root/rivermind-data/MutilModel_3485988898'
sys.path.insert(0, PROJ); os.chdir(PROJ)
faulthandler.dump_traceback_later(900, exit=True)
t0 = time.time()
def mark(s): print(f'[T+{time.time()-t0:6.1f}s] {s}', flush=True)

dp = pathlib.Path('/root/rivermind-data/uav_data_v2/data.yaml')
ys = yaml.safe_load(dp.read_text(encoding='utf-8'))
ys['labels'] = {'rgb': 'labels', 'ir': 'labels_ir'}
py = pathlib.Path('/tmp/_probe_data_dual.yaml')
py.write_text(yaml.safe_dump(ys, allow_unicode=True, sort_keys=False), encoding='utf-8')

from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer

def _short(self, world_size=1):
    torch.cuda.reset_peak_memory_stats()
    mark('_setup_train 开始')
    self._setup_train(world_size)
    mark(f'_setup_train 完成 batches={len(self.train_loader)}')
    self.optimizer.zero_grad()
    for i, batch in enumerate(self.train_loader):
        mark(f'step {i+1} forward')
        batch = self.preprocess_batch(batch)
        loss, self.loss_items = self.compute_batch_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        import subprocess
        gpu = subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                             capture_output=True, text=True).stdout.strip()
        print(f'[step {i+1}] torch_alloc={torch.cuda.memory_allocated()/2**30:.2f}G '
              f'peak={torch.cuda.max_memory_allocated()/2**30:.2f}G '
              f'resv={torch.cuda.memory_reserved()/2**30:.2f}G  smi_used={gpu}MiB', flush=True)
        if i + 1 >= 5:
            break
    print(f'>>> RESULT peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    print(f'>>> RESULT peak_resv ={torch.cuda.max_memory_reserved()/2**30:.2f} GiB')
    raise SystemExit(0)

MultiModalDetectionTrainer._do_train = _short

from ultralytics import YOLOMM
mark('build YOLOMM')
model = YOLOMM('ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-deform.yaml')
mark(f'build done params={sum(p.numel() for p in model.model.parameters())/1e6:.3f}M')
model.train(data=str(py), epochs=1, batch=32, imgsz=640, workers=4, amp=False,
            device=0, val=False, plots=False, save=False, verbose=False,
            project='/tmp/_probe_mem', name='e4', exist_ok=True)
PYEOF

timeout 1000 $PY -u /tmp/_probe_final.py > /tmp/_probe_final.log 2>&1
echo "[rc] $?"
echo "===== 结果 ====="
grep -aE '^\[T\+|^\[step|^>>> RESULT|OutOfMemory|Traceback' /tmp/_probe_final.log | tail -30
echo "===== 栈（若有超时）====="
grep -a 'Thread' -A 12 /tmp/_probe_final.log | head -25
