#!/bin/bash
# 查看 Mendeley EO/IR 数据集的真实目录与配对结构
set -uo pipefail
D="/root/rivermind-fs/mendeley_eo_ir/Dataset_Mendeley_data/Labelled_UAV_Images/Labelled EO_IR Dataset"

echo "=== 顶层 ==="
ls -l "$D"
echo
echo "=== eo 子结构 ==="
find "$D/UAV_eo_Dataset" -maxdepth 2 -type d | sort
echo
echo "=== ir 子结构 ==="
find "$D/UAV_ir_Dataset" -maxdepth 2 -type d | sort
echo
echo "=== eo images/train 前 5 ==="
ls "$D/UAV_eo_Dataset/images/train" | head -5
echo "=== ir images/train 前 5 ==="
ls "$D/UAV_ir_Dataset/images/train" | head -5
echo
echo "=== 计数 ==="
for s in train val test; do
  e=$(ls "$D/UAV_eo_Dataset/images/$s" 2>/dev/null | wc -l)
  i=$(ls "$D/UAV_ir_Dataset/images/$s" 2>/dev/null | wc -l)
  echo "  $s : eo=$e ir=$i"
done
echo
echo "=== 标签位置 ==="
find "$D" -type d -name labels | sort
echo
echo "=== train 里 eo 与 ir 的文件名交集大小 ==="
comm -12 <(ls "$D/UAV_eo_Dataset/images/train" | sort) \
         <(ls "$D/UAV_ir_Dataset/images/train" | sort) | wc -l
echo "=== data.yaml? ==="
find "$D" -maxdepth 3 -iname '*.yaml' -o -maxdepth 3 -iname '*.yml' | head
