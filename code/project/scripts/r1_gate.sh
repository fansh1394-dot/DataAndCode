#!/bin/bash
# ============================================================
# R1 插队门控：phase0 part1（E1/E3）收尾后，抢在 part2 前跑 R1.1，跑完再恢复 part2
#
# 为什么需要它
#   phase0_chain.sh 1 在全流程末尾会自动启动 part2（E4_s0 → E1_s1 → … 共 5 run ≈ 75h）。
#   G1 闸门（R1.1 − E1 与 σ 的关系）决定选题一/三/四是否立项，在关键路径上，
#   故按用户决策让 R1 插在 part2 之前；part2 只用于补 σ，可延后。
#
# 时序
#   1. 等 phase0_chain_p2.pid 出现（说明 part1 的 test 评测已跑完、part2 刚启动）
#   2. 杀掉 part2 链及其训练进程，删掉半成品目录（chain 每次启动都会 rm -rf，安全）
#   3. 前台跑 run_r1.sh train（R1_DETACHED=1 走内联分支，含训练 + test 评测）
#   4. R1 结束（无论成败）→ 重启 part2
#
# 用法（detach 执行）
#   nohup setsid bash _d2p1/r1_gate.sh > _d2p1/r1_gate.log 2>&1 < /dev/null &
# ============================================================
set -uo pipefail

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
WDIR=$PROJ/_d2p1
P2PID=$WDIR/phase0_chain_p2.pid
NAME_E4=v2c_E4_deform_s0
LOG=$WDIR/r1_gate.log

cd "$PROJ" || exit 1
mkdir -p "$WDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "############ R1 门控启动 ############"
log "等待 part1 收尾 / part2 启动（pidfile 出现）"

for i in $(seq 1 1440); do   # 最长 12 小时
  if [ -f "$P2PID" ] && kill -0 "$(cat "$P2PID" 2>/dev/null || echo 0)" 2>/dev/null; then
    log "检测到 part2 已启动 pid=$(cat "$P2PID")"
    break
  fi
  # part1 链已退出但 part2 没起来的兜底
  if [ ! -f "$WDIR/phase0_chain_p1.pid" ] && [ "$i" -gt 60 ]; then
    log "part1 已结束且未见 part2 pidfile，直接进入 R1"
    break
  fi
  sleep 30
done

# ---------- 停掉 part2 ----------
if [ -f "$P2PID" ]; then
  PG=$(cat "$P2PID" 2>/dev/null || echo 0)
  log "停止 part2 链 pid=$PG"
  kill -TERM "$PG" 2>/dev/null || true
  sleep 3
  kill -KILL "$PG" 2>/dev/null || true
fi
pkill -f "phase0_chain.sh 2" 2>/dev/null || true
pkill -f "name $NAME_E4" 2>/dev/null || true

log "等待 GPU 上的 phase0 训练进程退出"
for i in $(seq 1 240); do
  pgrep -f "train_full.py" >/dev/null 2>&1 || break
  sleep 5
done
if pgrep -f "train_full.py" >/dev/null 2>&1; then
  log "⛔ 仍有 train_full.py 进程，放弃插队（不改动 part2 之外的状态）"
  exit 3
fi

rm -rf "$PROJ/ResTest/$NAME_E4"
rm -f "$P2PID"
log "已清理半成品 ResTest/$NAME_E4，GPU 现状："
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

# ---------- 跑 R1 ----------
log "---- 启动 R1（内联，含 R1.1 训练 + test 评测）----"
R1_DETACHED=1 bash scripts/run_r1.sh train r11 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
log "R1 结束 rc=$RC"

# ---------- 恢复 part2 ----------
log "---- 恢复 phase0 part2（5 run）----"
nohup setsid bash "$WDIR/phase0_chain.sh" 2 >> "$WDIR/phase0_chain_p2.gate.log" 2>&1 < /dev/null &
sleep 20
if [ -f "$P2PID" ]; then
  log "✅ part2 已恢复 pid=$(cat "$P2PID")"
else
  log "⛔ part2 未存活，请手动执行：bash $WDIR/phase0_chain.sh 2"
fi

log "############ R1 门控结束 ############"
exit "$RC"
