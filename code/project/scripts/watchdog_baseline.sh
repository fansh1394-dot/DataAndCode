#!/bin/bash
# ============================================================
# 单模态基线链式看门狗：RGB-only 100ep → IR-only 100ep
#
# 目的：补齐「融合是否有效」缺失的分母（E1 / E2）。
#       项目历史上从未训练过单模态基线，导致 fuse/fuse_ir 的增益
#       一直没有可比较的上界，A/B 交付架构之争也无法判定。
#
# 设计要点（吸取既往教训）：
#   1) pidfile 防重复启动 —— 连接重置后重试整条命令不会产生并发进程
#      （历史事故：非幂等启动脚本导致两个训练进程同时写同一目录）
#   2) 每个 run 开始前 rm -rf 目标目录 —— Ultralytics 的 results.csv
#      仅在文件不存在时写表头，同目录重跑会追加，产生 1,2,1,2 的脏序列
#   3) 结束判定双保险：进程消失 或 results.csv 达 101 行
#   4) batch 16 而非 32：nbs=64 下 accumulate 分别为 4 与 2，
#      等效批大小恒为 64，优化轨迹完全一致（见项目既有结论）；
#      而 16 已实测约 7.8G，在他人进程占用 2~7G 的机器上更安全。
#   5) 全程 tee 到日志，可审计
# ============================================================
set -uo pipefail

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1
PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/uav_data_v2/data.yaml
YAML=ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml
WLOG=$WDIR/watchdog_baseline.log
PIDFILE=$WDIR/watchdog_baseline.pid

mkdir -p "$WDIR"
cd "$PROJ" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

# ---- 防重复启动 ----
if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE" 2>/dev/null || echo 0)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    log "看门狗已在运行（pid=$OLD），本次启动忽略"
    exit 0
  fi
  log "清理陈旧 pidfile（pid=$OLD 已不存在）"
  rm -f "$PIDFILE"
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# ---- 单次训练 + 轮询 ----
run_one () {
  local MOD="$1"
  local NAME="$2"
  local CSV="$PROJ/ResTest/$NAME/results.csv"
  local LOG="$WDIR/$NAME.log"

  log "================ 开始 $NAME（--single-modal $MOD） ================"
  rm -rf "$PROJ/ResTest/$NAME"
  rm -f "$LOG"

  nohup "$PY" -u train_full.py \
    --config "$YAML" --single-modal "$MOD" \
    --label-from dual --data "$DATA" \
    --epochs 100 --batch 16 --imgsz 640 --workers 4 --seed 0 --safe-augment \
    --name "$NAME" --amp false \
    > "$LOG" 2>&1 &
  local TPID=$!
  log "$NAME 已启动，训练 pid=$TPID"

  local waited=0
  local last_report=0
  while true; do
    sleep 120
    waited=$((waited + 120))

    if ! kill -0 "$TPID" 2>/dev/null; then
      log "$NAME 训练进程已退出（累计等待 $((waited / 60)) 分钟）"
      break
    fi

    local n=0
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

  # ---- 结果摘要 ----
  if [ -f "$CSV" ]; then
    local done_n=$(($(wc -l < "$CSV") - 1))
    log "$NAME 完成 $done_n 轮"
    local best=$(awk -F, 'NR>1{if($0!="")v=$0} END{print v}' "$CSV" 2>/dev/null)
    log "$NAME 末行: ${best:0:160}"
    tr '\r' '\n' < "$LOG" | grep -aE "epochs completed" | tail -1 | tee -a "$WLOG"
  else
    log "[警告] $NAME 未生成 results.csv，请检查 $LOG"
    tr '\r' '\n' < "$LOG" | grep -aE "Traceback|Error|error" | tail -5 | tee -a "$WLOG"
  fi
  log ""
}

# ---- 主流程 ----
log "################ 基线看门狗启动 ################"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$WLOG"

run_one rgb v2_E1_rgbonly_s0
run_one x   v2_E2_ironly_s0

log "################ 两个基线全部完成 ################"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$WLOG"
