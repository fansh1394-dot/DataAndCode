#!/bin/bash
# ============================================================
# test 集四方正式评测（论文级「域内评测」口径）
#
# 背景
# -----
# 之前所有 A/B 数字（E1/E2/E3/C4）都取自 val（= 模型选择集，early-stop/best 都在它上面），
# 组内一致但带乐观偏差。项目硬约束：域内正式评测必须落在完全未参与训练的 test 集
# （16917 帧）。本脚本用 eval-only 入口（scripts/eval_split.py）在 test 上重跑四方。
#
# 标签口径（决定 primary_label_dir，即 bboxes 用的是哪套框）
#   E1 RGB-only          : `--label-from vis`  -> vis   (labels/)
#   E2 IR-only（错配臂） : `--label-from dual` -> dual  (primary 仍取 labels/，复现原口径)
#   E3 IR-only（公平臂） : `--label-from ir`   -> ir    (labels_ir/)
#   C4 dual+fusion       : dual                        (rgb->labels/, ir->labels_ir/)
#
# 管线同口径已在 val 上验收：E3 复现 mAP50-95 = 0.6445714377837141，与训练期
# final_validation.json 逐位一致。
# ============================================================
set -uo pipefail

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
WDIR=$PROJ/_d2p1
LOG=$WDIR/eval_test_all.log

mkdir -p "$WDIR" "$PROJ/ResTest/_test_eval"
cd "$PROJ" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one() {  # $1=实验名  $2=labels-mode  $3=tag
  echo "================ [$(date '+%F %T')] $1 (labels-mode=$2) ================"
  "$PY" -u scripts/eval_split.py \
    --weights "ResTest/$1/weights/best.pt" \
    --split test --labels-mode "$2" --tag "$3" \
    --batch 32 --device 0 --workers 4 2>&1 | grep -avE "^\s+Class|it/s\]"
  echo "---- rc=${PIPESTATUS[0]} ----"
}

{
  echo "############ test 集四方评测开始 [$(date '+%F %T')] ############"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  run_one v2_E1_rgbonly_s0       vis  v2_E1_rgbonly_test
  run_one v2_E2_ironly_s0        dual v2_E2_ironly_test
  run_one v2_E3_ironly_lblir_s0  ir   v2_E3_ironly_lblir_test
  run_one v2_C4_deform_s0        dual v2_C4_deform_test
  echo "############ [$(date '+%F %T')] 全部完成 ############"
  for t in v2_E1_rgbonly_test v2_E2_ironly_test v2_E3_ironly_lblir_test v2_C4_deform_test; do
    echo "--- $t ---"
    cat "$PROJ/ResTest/_test_eval/$t.json" 2>/dev/null || echo "(缺失)"
  done
} >> "$LOG" 2>&1
