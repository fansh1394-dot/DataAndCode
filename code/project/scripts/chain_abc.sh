#!/usr/bin/env bash
# A -> B -> C 串联执行链
#
# 为什么必须串联
# --------------
# A/B 是评测（要显存），C 是训练（要显存）。三者若各自挂守候，CMCE 一结束
# 会同时醒来抢显存，评测与训练互相拖慢甚至 OOM。因此统一由本脚本串行调度：
#   0) 等 mdp150_M0CMCE_s0 训练结束
#   1) A：补评 concat / CGAFusion / PIM / CMCE
#   2) B：IR/RGB 置零敏感度
#   3) C：M0_concat seed=1/2 种子方差标定
# 评测之间显存释放有延迟，每一步之间留 60s 缓冲。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
LOG=logs/abc_chain.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== A/B/C 串联链启动 ==="

# 0) 等 CMCE 训练结束
log "等待 mdp150_M0CMCE_s0 训练结束..."
while pgrep -f "train_v26.py.*--name mdp150_M0CMCE_s0" >/dev/null 2>&1; do
    sleep 120
done
log "CMCE 已结束，缓冲 60s 让 final_eval 释放显存"
sleep 60

# 1) A 阶段
log "--- A 阶段：融合算子组 test 评测 ---"
bash scripts/eval_fusion_ops_group.sh
sleep 60

# 2) B 阶段
log "--- B 阶段：模态置零敏感度 ---"
bash scripts/eval_zero_modality.sh
sleep 60

# 3) C 阶段
log "--- C 阶段：种子方差标定（两组 × 150 轮）---"
bash scripts/queue_seed_std.sh

log "=== A/B/C 串联链结束 ==="
