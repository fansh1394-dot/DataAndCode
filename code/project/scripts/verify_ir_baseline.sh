#!/bin/bash
# 验证 IR-only 基线的标注路由是否正确
cd /3250410011/root/rivermind-data/MutilModel_3485988898

for n in _smoke_rgbonly_baseline _smoke_xonly_baseline; do
  echo "########## $n ##########"
  echo "--- 数据加载日志（模态/标签相关） ---"
  tr '\r' '\n' < "_d2p1/$n.log" | grep -aiE "images_ir|labels_ir|secondary|modality|IR labels|RGB labels|single_modal|input=|总样本|完整性" | head -12
  echo ""
  echo "--- args.yaml 中的输入/模态 ---"
  grep -aE "single_modal|modality|use_multimodal_aug|mm_rgb_albu_enable" "ResTest/$n/args.yaml" 2>/dev/null
  echo ""
  echo "--- labels.jpg 是否存在（可视化抽样，可人工核对框） ---"
  ls -la "ResTest/$n/labels.jpg" 2>/dev/null
  echo ""
done

echo "########## 数据集标签目录核对 ##########"
echo "labels/      (RGB) 前 2 个文件:"
ls /root/rivermind-data/uav_data_v2/labels/train/*.txt 2>/dev/null | head -2
echo "labels_ir/   (IR)  前 2 个文件:"
ls /root/rivermind-data/uav_data_v2/labels_ir/train/*.txt 2>/dev/null | head -2
echo "两者文件数:"
echo "  labels/train:    $(ls /root/rivermind-data/uav_data_v2/labels/train/*.txt 2>/dev/null | wc -l)"
echo "  labels_ir/train: $(ls /root/rivermind-data/uav_data_v2/labels_ir/train/*.txt 2>/dev/null | wc -l)"

echo ""
echo "########## 关键：labels_ir 是否真有内容（非空标签数） ##########"
nonempty=0
total=0
for f in $(ls /root/rivermind-data/uav_data_v2/labels_ir/train/*.txt 2>/dev/null | head -300); do
  total=$((total+1))
  if [ -s "$f" ]; then nonempty=$((nonempty+1)); fi
done
echo "  抽样 300 个 labels_ir 文件：非空 $nonempty / $total"
