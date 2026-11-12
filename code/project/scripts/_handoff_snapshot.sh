#!/bin/bash
# 交接用现状快照（只读，不改变任何状态）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
cd "$PROJ" || exit 1

echo "===== 1. 当前训练的 run 与轮次 ====="
for d in ResTest/v2c_E1_rgbonly_s1 ResTest/v2c_E1_rgbonly_s2 ResTest/v2c_E4_deform_s0 \
         ResTest/v2c_E4_deform_s1 ResTest/v2c_E4_deform_s2 ResTest/v2c_R1_reg_early_s0; do
  if [ -f "$d/results.csv" ]; then
    n=$(grep -c '' "$d/results.csv")
    best=$(awk -F, 'NR>1{if($NF>m){m=$NF}}END{printf "%.6f", m}' "$d/results.csv" 2>/dev/null)
    echo "$(basename $d): 行数=$n best_fitness=$best"
  fi
done

echo
echo "===== 2. test 正式评测结果文件 ====="
find ResTest -maxdepth 2 -name "*.json" -newermt "2026-10-01" \
  | grep -i -E "test|eval" | head -20

echo
echo "===== 3. 关键 R1 结果 ====="
[ -f ResTest/v2c_R1_reg_early_s0/final_validation.json ] && \
  grep -E "mAP50-95" ResTest/v2c_R1_reg_early_s0/final_validation.json | head -3

echo
echo "===== 4. GPU / 进程 ====="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
pgrep -af "train_full.py" | head -2
pgrep -af "phase0_chain" | head -2

echo
echo "===== 5. 背景库 ====="
ls -la /root/rivermind-data/uav_data_v2/bg_bank 2>/dev/null | head -3
du -sh /root/rivermind-data/uav_data_v2/bg_bank 2>/dev/null
