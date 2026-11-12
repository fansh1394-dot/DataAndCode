#!/bin/bash
# 查清 uav_data_v2 的真实目录结构（标签到底在哪）
D=/root/rivermind-data/uav_data_v2
echo "===== 1. 顶层结构 ====="
ls -la "$D"

echo ""
echo "===== 2. 各子目录的文件数（只数一层） ====="
for sub in "$D"/*/; do
  n=$(find "$sub" -maxdepth 1 -type f 2>/dev/null | wc -l)
  dn=$(find "$sub" -maxdepth 1 -type d 2>/dev/null | wc -l)
  printf "  %-40s 文件 %-8s 子目录 %s\n" "$(basename "$sub")" "$n" "$dn"
done

echo ""
echo "===== 3. 一层层展开寻找 *.txt 标签 ====="
find "$D" -maxdepth 3 -type d 2>/dev/null | head -30

echo ""
echo "===== 4. 任意位置的 .txt 文件数（按三层聚合） ====="
for d in $(find "$D" -maxdepth 2 -type d 2>/dev/null); do
  c=$(find "$d" -maxdepth 1 -name '*.txt' -type f 2>/dev/null | wc -l)
  if [ "$c" -gt 0 ]; then
    printf "  %-55s %s 个 .txt\n" "$d" "$c"
  fi
done

echo ""
echo "===== 5. 随机取一个标签文件看内容 ====="
F=$(find "$D" -name '*.txt' -type f 2>/dev/null | head -1)
echo "  样例: $F"
[ -n "$F" ] && head -3 "$F"

echo ""
echo "===== 6. RGB/IR 图像目录 ====="
ls -d "$D"/images* 2>/dev/null
for im in "$D"/images/train "$D"/images_ir/train; do
  [ -d "$im" ] && echo "  $im : $(ls "$im" 2>/dev/null | wc -l) 个文件"
done
