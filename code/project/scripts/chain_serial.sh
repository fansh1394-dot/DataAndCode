#!/usr/bin/env bash
# 串行总链：CGAFusion 解绑版 -> 几何增强
#
# 为什么串行
# ----------
# 两条链各自挂"等显存"会互相抢占：谁先拿到 20 GB 谁跑，另一个无限等待，
# 甚至可能出现两边同时判定通过、双双 OOM 的情况。
# 串行能保证：任一时刻只有一个训练在跑，显存判定不会误判。
#
# 顺序理由
# --------
# CGA 解绑版验证的是「代码级改动能否让算子追平基线」，结论直接影响
# 第 5 章怎么写；几何增强验证的是「数据侧能否提升鲁棒性」，是并列议题。
# 先跑 CGA，能更早拿到可写进论文的结论。
set -u

# 容器分配了 256 核，但模型构建期的 stride 推断前向在 CPU 上跑，
# 线程超订会导致构建阶段长时间无输出（表现为「卡死」）。子进程继承此设置。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
LOG=logs/serial_chain.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== 串行总链启动：CGA解绑 -> 几何增强 ==="

log "--- 第 1 段：CGAFusion 解绑版 ---"
bash scripts/queue_cga_decoupled.sh
rc1=$?
log "第 1 段结束 exit=${rc1}"

# 让 final_eval 释放显存
sleep 60

log "--- 第 2 段：几何增强 ---"
bash scripts/queue_geo_aug.sh
rc2=$?
log "第 2 段结束 exit=${rc2}"

sleep 60

log "--- 第 3 段：P2 检测头 ---"
bash scripts/queue_p2_head.sh
rc3=$?
log "第 3 段结束 exit=${rc3}"

log "=== 串行总链结束（CGA=${rc1}, GEO=${rc2}, P2=${rc3}）==="
