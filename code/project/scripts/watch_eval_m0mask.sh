#!/usr/bin/env bash
# M0mask 对照组「训完即评测」守护 + 续接实验组
#
# 背景
# ----
# 正在跑的蒸馏链（queue_distill_ablation.sh）是旧版，启动时已固定脚本内容，
# 其内部没有评测步骤。因此单独起本守护，与训练链并行：
#   1) 轮询 M0mask 的 results.csv，直到末轮 >= 149（训练自然结束）；
#   2) 再确认 train_v26 进程已退出（避免 final_eval 与评测抢显存）；
#   3) 跑整 test 集评测，产出 JSON；
#   4) 评测完成后交给链脚本继续跑实验组（链本身会自动接上）。
#
# 为什么不改原链脚本：它已在运行中，替换文件对当前进程无效，反而制造版本歧义。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/mendeley_split/data.yaml
NAME=mdp150_M0mask_s0
RUN=YOLO26/ResTest/${NAME}
OUT_DIR=YOLO26/ResTest/_test_eval
LOG=logs/distill_auto_eval.log
POLL=${POLL:-120}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 等训练自然结束
while true; do
    if [ -f "${RUN}/results.csv" ]; then
        last=$(tail -1 "${RUN}/results.csv" | cut -d, -f1 | tr -d ' ')
        # 注意：pgrep -c 在「0 个匹配」时退出码为 1，若写成 `pgrep -c ... || echo 0`
        # 会把输出拼成 "0\n0"，变量里带换行，`-eq 0` 直接比较失败，守护会永远卡在轮询。
        # 这里显式吞掉退出码，只取 stdout。
        running=$(pgrep -fc "train_v26.py --config .* --name ${NAME}" 2>/dev/null)
        running=${running:-0}
        log "轮询：末轮=${last:-NA} 训练进程=${running}"
        if [ "${last:-0}" -ge 149 ] 2>/dev/null && [ "${running}" -eq 0 ]; then
            log "训练已结束（末轮 ep${last}）"
            break
        fi
    fi
    sleep "$POLL"
done

# 给 final_eval 与显存释放留出余量
sleep 60

if [ ! -f "${RUN}/weights/best.pt" ]; then
    log "致命：找不到 ${RUN}/weights/best.pt，放弃评测"
    exit 2
fi

out="${OUT_DIR}/${NAME}_test.json"
if [ -f "$out" ]; then
    log "评测结果已存在，跳过：$out"
    exit 0
fi

# 抢占策略：评测要赶在链脚本启动实验组之前完成。
# 链脚本自身也有显存守候（NEED=20000），若它先起跑，本评测会排队等待。
# 因此这里直接尝试；若显存不足则等待，直到拿到窗口。
EVAL_NEED=${EVAL_NEED:-16000}
while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ -n "${f}" ] && [ "${f}" -ge "${EVAL_NEED}" ]; then break; fi
    log "等待评测显存：空闲 ${f:-NA} MiB < ${EVAL_NEED} MiB"
    sleep 60
done

log "开始整 test 集评测 ${NAME}（空闲 ${f} MiB）"
$PY -u scripts/eval_split.py \
    --weights "${RUN}/weights/best.pt" \
    --data "$DATA" \
    --split test \
    --labels-mode dual \
    --imgsz 640 \
    --batch 32 \
    --workers 4 \
    --tag "${NAME}_test" \
    --out "$out" >> logs/eval_${NAME}.log 2>&1
code=$?

if [ "$code" -eq 0 ] && [ -f "$out" ]; then
    log "评测完成 exit=0 -> $out"
else
    log "评测失败 exit=${code}，见 logs/eval_${NAME}.log"
fi
exit "$code"
