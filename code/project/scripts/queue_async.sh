#!/usr/bin/env bash
# 错位鲁棒训练（第三阶段，排在 queue_dfl.sh 之后；端口 30453）
#
# 动机：test 集错位退化曲线（mdp150_M0_concat_s0）实测
#   平移 1px -2.2% / 2px -6.0% / 4px -15.5% / 6px -23.2% / 8px -29.5%（融合增益转负）
#   旋转 0.5° -4.7% / 1° -12.7% / 2° -24.3%
#   -> 对齐残差 1px 即掉出 2σ 判定带，纯靠标定无法兜住。
# 做法：训练期单侧注入 1~4px 平移 + ±1° 旋转（p=0.5），把容忍度从标定精度搬到数据分布。
#
# 纪律（沿用 queue_dfl.sh）
#   - batch 固定 32，与基线 mdp150_M0_concat_s0/s1/s2 同口径；显存不足只等待，不降 batch
#   - 等 queue_dfl.sh 整条链退出后才起跑（显式串行，避免两条链抢卡互相判“被抢占”）
#   - s/it > 5 判被抢占 -> 杀掉重来；OOM -> 原 batch 等待重试
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
ADATA=$PWD/_d2p1/async_yamls/data_train_async.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/async_chain.log
NEED=${NEED:-14000}
MAX_TRY=${MAX_TRY:-40}
WARM_WAIT=${WARM_WAIT:-360}
SLOW_IT=${SLOW_IT:-5}
BATCH=${BATCH:-32}
NAME=${NAME:-mdp150_M0Async_s0}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

recent_it_s() {
    local f="$1"
    [ -f "$f" ] || { echo 0; return; }
    local v
    v=$(grep -aoE '[ <][0-9]+\.[0-9]+s/it' "$f" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    echo "${v:-0}"
}

wait_prev_chain() {
    while pgrep -f 'scripts/queue_dfl.sh' > /dev/null 2>&1; do
        log "等待 queue_dfl.sh 结束（避免两条链抢卡）"
        sleep 300
    done
}

wait_gpu() {
    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then return 0; fi
        log "等待显存：空闲 ${f:-NA} MiB < ${NEED} MiB（batch 保持 ${BATCH}，不降级）"
        sleep 180
    done
}

[ -f "$CFG" ] || { log "致命：缺配置 $CFG"; exit 2; }
[ -f "$ADATA" ] || { log "致命：缺派生 yaml $ADATA（align 必须为 false）"; exit 3; }

wait_prev_chain

attempt=0
while [ "$attempt" -lt "$MAX_TRY" ]; do
    attempt=$((attempt + 1))

    run_dir=$(ls -d YOLO26/ResTest/${NAME}* 2>/dev/null | head -1)
    if [ -n "$run_dir" ] && [ -f "$run_dir/results.csv" ]; then
        last_ep=$(tail -1 "$run_dir/results.csv" | cut -d, -f1 | tr -d ' ')
        if [ "${last_ep:-0}" -ge 149 ] 2>/dev/null; then
            log "${NAME} 已完成（末轮 ep${last_ep}），跳过"
            exit 0
        fi
        mv "$run_dir" "YOLO26/ResTest/_aborted/${NAME}_$(date +%H%M%S)" 2>/dev/null
        log "${NAME} 残留目录已挪至 _aborted"
    fi

    wait_gpu
    log "第 ${attempt}/${MAX_TRY} 次起跑 ${NAME}  batch=${BATCH}  空闲=$(free_mib) MiB"

    traylog="logs/${NAME}.log"
    : > "$traylog"
    $PY -u train_v26.py \
        --config "$CFG" \
        --label-from dual \
        --data "$ADATA" \
        --name "$NAME" \
        --seed 0 \
        --epochs 150 \
        --patience 30 \
        --batch "$BATCH" \
        --workers 4 \
        --amp false \
        --mm-async-geo-p 0.5 \
        --mm-async-geo-translate-px 1 4 \
        --mm-async-geo-rotate-deg -1.0 1.0 \
        >> "$traylog" 2>&1 &
    pid=$!

    sleep "$WARM_WAIT"
    if ! kill -0 "$pid" 2>/dev/null; then
        log "  ${NAME} 启动即退出，见 ${traylog}（若报 align 守卫/参数错误则需修脚本）"
        tail -n 5 "$traylog" | tee -a "$LOG"
        exit 4
    fi
    it=$(recent_it_s "$traylog")
    log "  速度检查：s/it=${it}"
    if awk "BEGIN{exit !(${it:-0} > ${SLOW_IT})}"; then
        log "  s/it=${it} > ${SLOW_IT}，判定被抢占，杀掉重来"
        kill -9 "$pid" 2>/dev/null
        pkill -9 -f "train_v26.py.*--name ${NAME}" 2>/dev/null
        sleep 30
        continue
    fi

    wait "$pid"
    code=$?
    if [ "$code" -eq 0 ]; then
        log "${NAME} 完成 exit=0（batch=${BATCH}）"
        exit 0
    fi
    if grep -aq "CUDA out of memory" "$traylog"; then
        log "${NAME} OOM（batch=${BATCH}，保持不降），300s 后重试"
        sleep 300
        continue
    fi
    log "${NAME} 异常退出 exit=${code}（非 OOM）；见 ${traylog}"
    exit "$code"
done
log "${NAME} 达到 MAX_TRY=${MAX_TRY} 仍未完成"
exit 9
