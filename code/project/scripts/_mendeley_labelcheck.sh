#!/bin/bash
# 核实 Mendeley EO/IR 标签是否逐字节相同，并抽样人工核对图像内容
set -uo pipefail
D="/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/Labelled_UAV_Images/Labelled EO_IR Dataset"
PY=/opt/conda/envs/rivermind/bin/python

echo "=== 1. 标签文件逐字节比对（test split，全部 614 个）==="
same=0; diff=0
for f in "$D/UAV_eo_Dataset/labels/test"/*.txt; do
  b=$(basename "$f")
  if cmp -s "$f" "$D/UAV_ir_Dataset/labels/test/$b"; then same=$((same+1)); else diff=$((diff+1)); fi
done
echo "  完全相同: $same   不同: $diff"

echo
echo "=== 2. 全数据集比对（找一个不同的例子）==="
found=0
for f in "$D/UAV_eo_Dataset/labels/train"/*.txt; do
  b=$(basename "$f")
  if ! cmp -s "$f" "$D/UAV_ir_Dataset/labels/train/$b"; then
    echo "  差异样例: $b"; echo "    EO: $(cat "$f")"; echo "    IR: $(cat "$D/UAV_ir_Dataset/labels/train/$b")"
    found=1; break
  fi
done
[ "$found" = 0 ] && echo "  train 全部逐字节相同"

echo
echo "=== 3. 标签内容分布（test，前 10 个非空）==="
n=0
for f in "$D/UAV_eo_Dataset/labels/test"/*.txt; do
  [ -s "$f" ] || continue
  echo "  $(basename "$f" .txt): $(cat "$f")"
  n=$((n+1)); [ "$n" -ge 10 ] && break
done

echo
echo "=== 4. 空标签比例 ==="
for s in train val test; do
  tot=$(ls "$D/UAV_eo_Dataset/labels/$s" | wc -l)
  emp=$(find "$D/UAV_eo_Dataset/labels/$s" -name '*.txt' -empty | wc -l)
  echo "  $s: total=$tot empty=$emp"
done

echo
echo "=== 5. EO 与 IR 图像是否逐字节相同（说明是否是同一张图）==="
n_same=0; n_diff=0
for f in $(ls "$D/UAV_eo_Dataset/images/test"/*.png | head -30); do
  b=$(basename "$f")
  h1=$(md5sum "$f" | cut -d' ' -f1)
  h2=$(md5sum "$D/UAV_ir_Dataset/images/test/$b" | cut -d' ' -f1)
  [ "$h1" = "$h2" ] && n_same=$((n_same+1)) || n_diff=$((n_diff+1))
done
echo "  前 30 张: 相同=$n_same 不同=$n_diff"

echo
echo "=== 6. 图像像素统计对比（前 5 张，判断是否真是不同模态）==="
cd /3250410011/root/rivermind-data/MutilModel_3485988898
"$PY" - "$D" <<'PY'
import sys, cv2, numpy as np, glob, os
D = sys.argv[1]
files = sorted(glob.glob(os.path.join(D, 'UAV_eo_Dataset/images/test/*.png')))[:5]
for f in files:
    b = os.path.basename(f)
    e = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    i = cv2.imread(os.path.join(D, 'UAV_ir_Dataset/images/test', b), cv2.IMREAD_GRAYSCALE)
    if e is None or i is None: continue
    print(f"  {b}: EO mean={e.mean():.1f} std={e.std():.1f} | IR mean={i.mean():.1f} std={i.std():.1f} "
          f"| 像素相关系数={np.corrcoef(e.ravel(), i.ravel())[0,1]:.3f}")
PY
