#!/bin/bash
# ============================================================
# IR-only 基线看门狗（100 epoch，单跑）
#
# 为什么只跑 IR-only 而不跑 RGB-only：
#   实例剩余时长 23h26m，只够一个 100 轮（约 11~13h）。
#   两者的信息价值不对等：
#     * RGB-only 的收敛值高度可预测（2 轮即 0.618），且双头模型里的 rgb 头
#       （0.7052）在结构上就是「RGB 骨干 → RGB 标签 → 单检测头」，已是等价代理；
#     * IR-only 在 10 轮时仅 0.173 且损失仍在陡降，收敛值完全未知，而它
#       恰好是判定 fuse_ir(0.6427) 是否真有增益的唯一分母 —— 直接决定
#       A/B 交付架构之争。
#
# 设计要点（沿用既有教训）：
#   1) pidfile 防重复启动（避免连接重置重试导致并发训练）
#   2) 开跑前清空目标目录（results.csv 仅在不存在时写表头，否则追加污染）
#   3) 结束判定双保险：进程消失 或 results.csv 达 101 行
#   4) batch 16：nbs=64 下 accumulate=4，等效批大小仍为 64，与 batch 32
#      优化轨迹一致，但显存更省（本机另有他人进程占 2~7G）
# ============================================================
set -uo pipefail

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1
PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/uav_data_v2/data.yaml
YAML=ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml
NAME=v2_E2_ironly_s0
CSV=$PROJ/ResTest/$NAME/results.csv
LOG=$WDIR/$NAME.log
WLOG=$WDIR/watchdog_ironly.log
PIDFILE=$WDIR/watchdog_ironly.pid

mkdir -p "$WDIR"
cd "$PROJ" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE" 2>/dev/null || echo 0)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    log "看门狗已在运行（pid=$OLD），本次忽略"
    exit 0
  fi
  rm -f "$PIDFILE"
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "################ IR-only 基线看门狗启动 ################"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$WLOG"

rm -rf "$PROJ/ResTest/$NAME"
rm -f "$LOG"

nohup "$PY" -u train_full.py \
  --config "$YAML" --single-modal x \
  --label-from dual --data "$DATA" \
  --epochs 100 --batch 16 --imgsz 640 --workers 4 --seed 0 --safe-augment \
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
    if [ "$n" -ge 100 ]; then
      log "$NAME results.csv 已达 $n 轮，等待 final_eval 收尾"
      sleep 240
      break
    fi
  fi

  if [ $((waited - last_report)) -ge 1800 ]; then
    last_report=$waited
    log "$NAME 进度 $n/100（已等 $((waited / 60)) 分钟）"
  fi
done

if [ -f "$CSV" ]; then
  log "$NAME 完成 $(($(wc -l < "$CSV") - 1)) 轮"
  tr '\r' '\n' < "$LOG" | grep -aE "epochs completed" | tail -1 | tee -a "$WLOG"
else
  log "[警告] $NAME 未生成 results.csv，检查 $LOG"
  tr '\r' '\n' < "$LOG" | grep -aE "Traceback|Error" | tail -5 | tee -a "$WLOG"
fi
log "################ 结束 ################"
