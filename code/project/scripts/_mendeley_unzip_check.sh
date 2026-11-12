#!/bin/bash
# 解压 Mendeley 数据集并跑对齐检验
set -uo pipefail

ZIP=/root/rivermind-fs/UAV_dataset_EO_IR-nbsXw2.zip
DEST=/root/rivermind-fs/mendeley_eo_ir
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python

mkdir -p "$DEST"
echo "[$(date '+%F %T')] 解压 $ZIP -> $DEST"
unzip -q -o "$ZIP" -d "$DEST"
rc=$?
echo "[$(date '+%F %T')] unzip rc=$rc"

echo "=== 顶层结构 ==="
ls -l "$DEST"
echo "=== 目录树（深度 3，仅目录）==="
find "$DEST" -maxdepth 3 -type d | sed "s|$DEST|.|" | sort | head -60
echo "=== 文件类型统计 ==="
find "$DEST" -type f | sed 's/.*\.//' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -15
echo "=== 图像总数 ==="
find "$DEST" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l
echo "=== 磁盘占用 ==="
du -sh "$DEST"

echo
echo "[$(date '+%F %T')] 开始对齐检验"
cd "$PROJ" || exit 1
"$PY" -u scripts/_check_mendeley_align.py --root "$DEST" --max-pairs 300
