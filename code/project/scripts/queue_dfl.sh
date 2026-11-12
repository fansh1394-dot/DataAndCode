#!/usr/bin/env bash
# 回归精度实验链（两阶段串行，端口 30453）
#
# 阶段 1: mdp150_M0DFL_s0     box 7.5->12.0, dfl 1.5->3.0
# 阶段 2: mdp150_M0GShape_s0  标签分配相似度 iou -> geoshape（COXNet GeoShape）
#
# 纪律
#   - batch 固定 32，与基线 mdp150_M0_concat_s0/s1/s2 严格同口径；显存不足只等待，
#     不降 batch（2026-11-02 用户明确要求）。OOM 仅原 batch 重试。
#   - 判定带：test μ=0.7551 σ=0.0094 -> 有效 >0.7739 (+2σ)，有害 <0.7363 (-2σ)
#   - 小目标瓶颈依据：<16px 桶 R50=0.837、IoU>=0.5 占 97%、medIoU 0.795，
#     但 mAP50-95 仅 0.50（1px 误差即 IoU 0.784），故两个阶段都只动定位/分配侧。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/dfl_chain.log
NEED=${NEED:-14000}
MAX_TRY=${MAX_TRY:-40}
WARM_WAIT=${WARM_WAIT:-360}
SLOW_IT=${SLOW_IT:-5}
BATCH=${BATCH:-32}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

recent_it_s() {
    local f="$1"
    [ -f "$f" ] || { echo 0; return; }
    local v
    v=$(grep -aoE '[ <][0-9]+\.[0-9]+s/it' "$f" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    echo "${v:-0}"
}

# 只等待，永不降 batch（用户 2026-11-02 要求）
wait_gpu() {
    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then return 0; fi
        log "等待显存：空闲 ${f:-NA} MiB < ${NEED} MiB（batch 保持 ${BATCH}，不降级）"
        sleep 180
    done
}

run_one() {
    local name="$1"
    shift
    local extra=("$@")
    local attempt=0 pid code traylog run_dir last_ep it

    while [ "$attempt" -lt "$MAX_TRY" ]; do
        attempt=$((attempt + 1))

        run_dir=$(ls -d YOLO26/ResTest/${name}* 2>/dev/null | head -1)
        if [ -n "$run_dir" ] && [ -f "$run_dir/results.csv" ]; then
            last_ep=$(tail -1 "$run_dir/results.csv" | cut -d, -f1 | tr -d ' ')
            if [ "${last_ep:-0}" -ge 149 ] 2>/dev/null; then
                log "${name} 已完成（末轮 ep${last_ep}），跳过"
                return 0
            fi
            mv "$run_dir" "YOLO26/ResTest/_aborted/${name}_$(date +%H%M%S)" 2>/dev/null
            log "${name} 残留目录已挪至 _aborted"
        fi

        wait_gpu
        log "第 ${attempt}/${MAX_TRY} 次起跑 ${name}  batch=${BATCH}  extra=(${extra[*]:-})  空闲=$(free_mib) MiB"

        traylog="logs/${name}.log"
        : > "$traylog"
        $PY -u train_v26.py \
            --config "$CFG" \
            --label-from dual \
            --data "$DATA" \
            --name "$name" \
            --seed 0 \
            --epochs 150 \
            --patience 30 \
            --batch "$BATCH" \
            --workers 4 \
            --amp false \
            "${extra[@]}" \
            >> "$traylog" 2>&1 &
        pid=$!

        sleep "$WARM_WAIT"
        if kill -0 "$pid" 2>/dev/null; then
            it=$(recent_it_s "$traylog")
            log "  速度检查：s/it=${it}"
            if awk "BEGIN{exit !(${it:-0} > ${SLOW_IT})}"; then
                log "  s/it=${it} > ${SLOW_IT}，判定被抢占，杀掉重来"
                kill -9 "$pid" 2>/dev/null
                pkill -9 -f "train_v26.py.*--name ${name}" 2>/dev/null
                sleep 30
                continue
            fi
        fi

        wait "$pid"
        code=$?
        if [ "$code" -eq 0 ]; then
            log "${name} 完成 exit=0（batch=${BATCH}）"
            return 0
        fi
        if grep -aq "CUDA out of memory" "$traylog"; then
            log "${name} OOM（batch=${BATCH}，保持不降），300s 后重试"
            sleep 300
            continue
        fi
        log "${name} 异常退出 exit=${code}（非 OOM）；见 ${traylog}"
        return "$code"
    done
    log "${name} 达到 MAX_TRY=${MAX_TRY} 仍未完成"
    return 9
}

[ -f "$CFG" ] || { log "致命：缺配置 $CFG"; exit 2; }

log "=== 阶段1/2 启动：box/DFL ==="
run_one mdp150_M0DFL_s0 --box 12.0 --dfl 3.0 || { log "阶段1失败，链终止"; exit 7; }

log "=== 阶段2/2 启动：GeoShape 分配 ==="
run_one mdp150_M0GShape_s0 --loss-assign-metric geoshape || { log "阶段2失败"; exit 8; }

log "=== 两阶段全部完成 ==="
