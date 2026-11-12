import glob, os, subprocess
import yaml
from PIL import Image

base = "/3250410011/root/rivermind-data/mendeley_split"
print("dirs:", os.listdir(base)[:10])
imgs = sorted(glob.glob(base + "/images/train/*.png"))[:2]
print("sample:", imgs)
for p in imgs:
    print("size:", p, Image.open(p).size)

for f in ["/3250410011/root/rivermind-data/MutilModel_3485988898/ultralytics/cfg/default.yaml"]:
    if os.path.isfile(f):
        d = yaml.safe_load(open(f))
        keys = [k for k in d if any(s in k.lower() for s in ("crop", "repaste", "paste", "scale", "translate", "mixup", "copy"))]
        print("aug keys:", {k: d[k] for k in keys})

r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("gpu:", r.stdout.strip())
r = subprocess.run("ps aux | grep -E 'train_full|trainMM|YOLO' | grep -v grep", shell=True, capture_output=True, text=True)
print("procs:", r.stdout.strip()[:500])
