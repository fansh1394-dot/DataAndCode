#!/usr/bin/env bash
# 守候式续训：part3h 在 ep20 的 final_eval 阶段被外部租户挤爆显存而中断，
# 等空闲显存达标后从 last.pt 续训剩余轮次。
#
# 与 queue_part3h.sh 的关键差异（切勿混用）：
#   1. 绝不再把 YOLO26/ResTest/${NAME} 挪到 _aborted —— 那里有 ep0~20 的权重与
#      results.csv，挪走就等于丢训练成果；续训必须原地复用该目录；
#   2. 起手先用 --resume 从 last.pt 恢复优化器/EMA/轮次，而不是从头训；
#   3. OOM 重试只重试「续训」这一步，且重试前先校验 last.pt 仍在。
#
# 用法：setsid nohup bash scripts/queue_part3h_resume.sh < /dev/null > /dev/null 2>&1 &
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

NEED=${NEED:-20000}          # 起跑所需空闲显存（MiB）：实测 batch 32 峰值约 19 GB，
                             # 21000 会在「只剩一个 3.3 GB 残留进程」时白白空转
NAME=${NAME:-mdp150_M3part3h_s0}
CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-part3h.yaml
PY=/opt/conda/envs/rivermind/bin/python
RUN=YOLO26/ResTest/${NAME}
LAST=${RUN}/weights/last.pt
LOG=logs/${NAME}.resume.log
TRAINLOG=logs/${NAME}.resume_train.log
MAX_TRY=${MAX_TRY:-20}
mkdir -p logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

# 续训前置校验：目录/权重/已完成轮次必须都在，否则直接放弃（不擅自重训覆盖）
if [ ! -f "$LAST" ]; then
    log "致命：找不到续训权重 ${LAST}，放弃（不做任何覆盖操作）"
    exit 2
fi
start_ep=$(tail -1 "${RUN}/results.csv" 2>/dev/null | cut -d, -f1 | tr -d ' ')
log "续训目标：${NAME}；last.pt 存在；results.csv 最后轮次=${start_ep:-未知}"

attempt=0
while [ "$attempt" -lt "$MAX_TRY" ]; do
    attempt=$((attempt + 1))
    log "第 ${attempt}/${MAX_TRY} 次尝试：等待空闲显存 >= ${NEED} MiB"
    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then break; fi
        log "空闲 ${f:-NA} MiB < ${NEED} MiB，120s 后重试"
        sleep 120
    done

    # 每次重试前复核权重仍在（防止上一轮崩溃写坏 last.pt）
    if [ ! -f "$LAST" ]; then
        log "致命：重试前发现 ${LAST} 不存在，停止守候"
        exit 2
    fi

    log "显存达标（空闲 ${f} MiB），从 last.pt 续训"
    # train_v26.py 的参数校验是「先 config 再 resume」：
    #   * --config 是前置必填项，即使续训也要给（校验顺序见 L76-81）；
    #   * --resume 只认 --name（内部固定读 YOLO26/ResTest/<name>/weights/last.pt），
    #     不接受 ckpt 路径参数；
    #   * 其余训练参数（epochs/patience/batch/amp/seed）由 check_resume 从 ckpt 的
    #     train_args 自动恢复，这里重复传反而会覆盖，故一律不传。
    $PY -u train_v26.py \
        --config "$CFG" \
        --resume \
        --name "$NAME" >> "$TRAINLOG" 2>&1
    code=$?

    if [ "$code" -eq 0 ]; then
        log "续训正常结束，exit=0；日志 ${TRAINLOG}"
        exit 0
    fi

    if grep -q "CUDA out of memory" "$TRAINLOG"; then
        log "本次因 CUDA OOM 退出（exit=${code}），300s 后重试"
        sleep 300
        continue
    fi

    log "续训异常退出（exit=${code}），非 OOM，放弃重试；日志 ${TRAINLOG}"
    exit "$code"
done

log "重试次数用尽（${MAX_TRY}），放弃"
exit 1
