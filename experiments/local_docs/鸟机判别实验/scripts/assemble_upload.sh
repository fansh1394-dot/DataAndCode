#!/bin/bash
# 组装待上传 GitHub 的目录：代码 + 实验记录 + 训练结果（不含数据集）
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
BD=/root/rivermind-fs/birddrone
STG=$BD/upload_repo

rm -rf "$STG"
mkdir -p "$STG"/{code,experiments,results}

# ---------- 1. 项目根代码 ----------
mkdir -p "$STG/code/project"
find "$PROJ" -maxdepth 1 -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' \
  -o -name '*.sh' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \
  -o -name '*.ipynb' -o -name '.gitignore' \) ! -size +5M \
  -exec cp -p {} "$STG/code/project/" \;
for d in docs scripts tools reports; do
  [ -d "$PROJ/$d" ] && cp -rp "$PROJ/$d" "$STG/code/project/"
done

# ---------- 2. ultralytics 框架代码 ----------
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
  "$PROJ/ultralytics" "$STG/code/"

# ---------- 3. 训练结果 ----------
copy_run() {  # src dst
  local src=$1 dst=$2
  [ -d "$src" ] || return 0
  mkdir -p "$dst/weights"
  find "$src" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.yaml' \
    -o -name '*.png' -o -name '*.jpg' -o -name '*.json' \) -exec cp -p {} "$dst/" \;
  for w in best last; do
    [ -f "$src/weights/$w.pt" ] && cp -p "$src/weights/$w.pt" "$dst/weights/"
  done
}

for d in "$PROJ/YOLO26/ResTest"/*/; do
  n=$(basename "$d")
  copy_run "$d" "$STG/results/yolo26_restest/$n"
done
for d in "$PROJ/ResTest"/*/; do
  n=$(basename "$d")
  copy_run "$d" "$STG/results/yolo11_restest/$n"
done

# ---------- 4. 鸟机判别实验（脚本/报告/配置/端侧模型/日志） ----------
mkdir -p "$STG/experiments/birddrone"
for d in scripts reports configs; do
  [ -d "$BD/$d" ] && cp -rp "$BD/$d" "$STG/experiments/birddrone/"
done
[ -f "$BD/SOURCES.md" ] && cp -p "$BD/SOURCES.md" "$STG/experiments/birddrone/"
[ -f "$BD/nc3_init.pt" ] && cp -p "$BD/nc3_init.pt" "$STG/experiments/birddrone/"
mkdir -p "$STG/experiments/birddrone/deploy_nc3"
for f in dataset.txt deploy_config.json model_2in.onnx model_nc3_rk3588.rknn model_raw.onnx; do
  [ -f "$BD/deploy_nc3/$f" ] && cp -p "$BD/deploy_nc3/$f" "$STG/experiments/birddrone/deploy_nc3/"
done
mkdir -p "$STG/experiments/logs"
cp -p "$BD"/batch1/*.log "$STG/experiments/logs/" 2>/dev/null

# ---------- 5. 清理缓存 ----------
find "$STG" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$STG" -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$STG" \( -name '*.pyc' -o -name '*.log' \) -delete 2>/dev/null
find "$STG/experiments/logs" -name '*.log' -size +50M -delete 2>/dev/null

echo "===== 体积 ====="
du -sh "$STG"/* 
du -sh "$STG"
echo "run 数: $(( $(find "$STG/results/yolo26_restest" -maxdepth 1 -mindepth 1 -type d | wc -l) + $(find "$STG/results/yolo11_restest" -maxdepth 1 -mindepth 1 -type d | wc -l) ))"
