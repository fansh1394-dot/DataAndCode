#!/bin/bash
# ============================================================
# E1_s1 收尾守卫：E1_s1 训练 + test 评测都跑完后，抢在 E1_s2 启动前停掉 part2 链
#
# 为什么需要它
#   用户要在 E1_s1 完成后克隆实例，克隆期间必须没有训练占 GPU、且文件不再变动。
#   phase0_chain.sh 2 在 E1_s1 评测结束后会立即启动 E1_s2（中间只隔几秒），
#   故必须在链内部把它拦下来，而不是从外部事后 pkill。
#
# 时序
#   1. 轮询等待：E1_s1 进程退出 且 _test_eval/v2c_E1_rgbonly_s1_test.json 出现
#   2. kill 掉 part2 链（先 TERM 后 KILL）以及任何 train_full.py
#   3. 清掉可能已启动的 ResTest/v2c_E1_rgbonly_s2 半成品，删除 p2 pidfile
#   4. 写 _d2p1/CLONE_READY 标记文件（含时间戳与 E1_s1 最终数字）
#
# 恢复（克隆完成后手动执行）
#   bash _d2p1/phase0_chain.sh 2
# ============================================================
set -uo pipefail

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1
PY=/opt/conda/envs/rivermind/bin/python
LOG=$WDIR/clone_guard.log
P2PID=$WDIR/phase0_chain_p2.pid
NAME=v2c_E1_rgbonly_s1
JSON=$PROJ/ResTest/_test_eval/${NAME}_test.json

cd "$PROJ" || exit 1
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "############ E1_s1 克隆守卫启动 ############"

# ---------- 1. 等 E1_s1 训练进程退出 ----------
E1PID=0
for i in $(seq 1 5); do
  E1PID=$(pgrep -f "name $NAME" | head -1)
  [ -n "$E1PID" ] && break
  sleep 60
done
if [ -z "$E1PID" ]; then
  log "未找到 $NAME 训练进程（可能已结束），继续等评测结果"
else
  log "监视 $NAME 训练 pid=$E1PID"
fi

# ---------- 2. 等训练结束 且 test json 落盘 ----------
for i in $(seq 1 2160); do   # 最长 36 小时（150ep 约 19h + 余量）
  alive=0
  pgrep -f "name $NAME" >/dev/null 2>&1 && alive=1
  if [ "$alive" -eq 0 ]; then
    if [ -f "$JSON" ]; then
      log "✅ $NAME 训练结束且 test 评测完成"
      break
    fi
    # 训练已退但 json 未落：说明链正在评测，继续等
  fi
  if [ $((i % 10)) -eq 0 ]; then
    HAS=no; [ -f "$JSON" ] && HAS=yes
    log "  等待中 第 $i 分钟: 训练alive=$alive test_json=$HAS"
  fi
  sleep 60
done

if [ ! -f "$JSON" ]; then
  log "⛔ 等待超时：$NAME 训练未完成或 test 评测未产出，仍执行停链以避免 GPU 被后续 run 占用"
fi

# ---------- 3. 停 part2 链 ----------
if [ -f "$P2PID" ]; then
  PG=$(cat "$P2PID" 2>/dev/null || echo 0)
  log "停止 part2 链 pid=$PG"
  kill -TERM "$PG" 2>/dev/null || true
fi
pkill -f "phase0_chain.sh 2" 2>/dev/null || true
sleep 3
pkill -KILL -f "phase0_chain.sh 2" 2>/dev/null || true

# 若 E1_s2 已抢跑，一并停掉
for i in $(seq 1 60); do
  pkill -f "name v2c_E1_rgbonly_s2" 2>/dev/null || true
  pgrep -f "train_full.py" >/dev/null 2>&1 || break
  sleep 5
done
if pgrep -f "train_full.py" >/dev/null 2>&1; then
  log "⛔ 仍有 train_full.py 进程未退出，请人工检查："
  pgrep -af "train_full.py" | cut -c1-120 | tee -a "$LOG"
else
  log "GPU 上已无训练进程"
fi

# ---------- 4. 清理半成品 + pidfile ----------
rm -rf "$PROJ/ResTest/v2c_E1_rgbonly_s2"
rm -f "$P2PID" "$WDIR/run_r1_train.pid"
log "已清理 ResTest/v2c_E1_rgbonly_s2 与 p2 pidfile"

# ---------- 5. 标记就绪 ----------
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"
{
  echo "ready_at=$(date '+%F %T')"
  echo "name=$NAME"
  if [ -f "$JSON" ]; then
    "$PY" -c "import json;m=json.load(open('$JSON'))['metrics'];print('test_mAP50-95=%.6f'%m['metrics/mAP50-95(B)']);print('test_mAP50=%.6f'%m['metrics/mAP50(B)'])"
  fi
} > "$WDIR/CLONE_READY"
log "已写 $WDIR/CLONE_READY"
log "############ 可以克隆实例 ############"
log "克隆完成后恢复：bash $WDIR/phase0_chain.sh 2"
