#!/bin/bash
# ============================================================
# 公平 IR-only 基线看门狗：等 E2 结束后，用 --label-from ir 重跑
#
# 为什么必须重跑
# --------------
# E2（v2_E2_ironly_s0, 0.2074）用的是 `--label-from dual`，它把单模态 IR 映射到
# `labels`（RGB 坐标系）——即「用 IR 图去学 RGB 坐标系的框」。交叉口径探针实测：
#   ir 头预测 vs RGB 标签 = 0.0038   （同一批预测 vs IR 标签 = 0.6084）
#   rgb 头预测 vs IR 标签 = 0.0055
#   IR-only 预测 vs IR 标签 = 0.0035
# 说明 RGB 与 IR 两套标注几何上近乎互斥（非对齐 + 可见性集合不同）。因此
# 「ir 头 0.6415 vs IR-only 0.2074 = +0.43」是**标签口径错配造成的假性差距**，
# 不是 IR 支路能力问题。必须用 `--label-from ir`（primary → labels_ir）重跑，
# 才是与 C4 的 ir 头同口径的可比基线。
#
# 其他参数与 E1/E2 完全一致。epochs 取 40：mAP 自 ep20 起已平台化（见 E2 曲线），
# 且 40 轮 × 465 iter = 18600 > 10000，仍走 SGD，与 100 轮同优化器族。
# ============================================================
set -uo pipefail

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1
PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/uav_data_v2/data.yaml
YAML=ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml
NAME=v2_E3_ironly_lblir_s0
EPOCHS=40
WLOG=$WDIR/watchdog_ironly_fair.log
PIDFILE=$WDIR/watchdog_ironly_fair.pid
E2CSV=$PROJ/ResTest/v2_E2_ironly_s0/results.csv
CSV=$PROJ/ResTest/$NAME/results.csv
LOG=$WDIR/$NAME.log

mkdir -p "$WDIR"
cd "$PROJ" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE" 2>/dev/null || echo 0)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    log "已在运行（pid=$OLD），本次启动忽略"
    exit 0
  fi
  rm -f "$PIDFILE"
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "################ 公平 IR-only 基线看门狗启动 ################"

# ---- 等 E2 结束，避免与现有训练抢显存 ----
while pgrep -f "train_full.py" >/dev/null 2>&1; do
  n=0
  [ -f "$E2CSV" ] && n=$(($(wc -l < "$E2CSV") - 1))
  log "等待 E2 结束… 当前 $n/100"
  sleep 180
done
log "已无 train_full.py 进程，E2 视为结束"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$WLOG"

# ---- 启动公平 IR-only ----
rm -rf "$PROJ/ResTest/$NAME"
rm -f "$LOG"
log "================ 开始 $NAME（--single-modal x --label-from ir, ${EPOCHS}ep） ================"
nohup "$PY" -u train_full.py \
  --config "$YAML" --single-modal x \
  --label-from ir --data "$DATA" \
  --epochs "$EPOCHS" --batch 16 --imgsz 640 --workers 4 --seed 0 --safe-augment \
  --name "$NAME" --amp false \
  > "$LOG" 2>&1 &
TPID=$!
log "$NAME 已启动，训练 pid=$TPID"

waited=0
last_report=0
while true; do
  sleep 120
  waited=$((waited + 120))
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "$NAME 训练进程已退出（累计等待 $((waited / 60)) 分钟）"
    break
  fi
  n=0
  if [ -f "$CSV" ]; then
    n=$(($(wc -l < "$CSV") - 1))
    if [ "$n" -ge "$EPOCHS" ]; then
      log "$NAME results.csv 已达 $n 轮，等待 final_eval 收尾"
      sleep 240
      break
    fi
  fi
  if [ $((waited - last_report)) -ge 1800 ]; then
    last_report=$waited
    log "$NAME 进度 $n/$EPOCHS（已等 $((waited / 60)) 分钟）"
  fi
done

if [ -f "$CSV" ]; then
  log "$NAME 完成 $(($(wc -l < "$CSV") - 1)) 轮"
  tr '\r' '\n' < "$LOG" | grep -aE "epochs completed" | tail -1 | tee -a "$WLOG"
else
  log "[警告] $NAME 未生成 results.csv，请检查 $LOG"
  tr '\r' '\n' < "$LOG" | grep -aE "Traceback|Error|error" | tail -5 | tee -a "$WLOG"
fi
log "################ 公平 IR-only 基线完成 ################"
