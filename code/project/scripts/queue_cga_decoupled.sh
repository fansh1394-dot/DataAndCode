#!/usr/bin/env bash
# CGAFusion 解绑版实验链（改法 A）
#
# 实验设计
# --------
# 对照组（已有结果，不重跑）：
#   mdp150_M0_concat_s0/s1/s2   基线，μ=0.7551  σ=0.0094  2σ=0.0188
#   mdp150_M0CGAFusion_s0       旧 CGA（系数绑定），0.7435
#
# 实验组：
#   mdp150_M0CGADec_s0          新 CGA（系数解绑 + 残差），YAML 与旧 CGA 相同
#
# 判定：
#   超过 0.7739（μ+2σ）-> 显著优于基线
#   落在 [0.7363, 0.7739] -> 与基线无显著差异
#   低于 0.7363 -> 显著劣于基线
#
# 注意：cgafusion.py 已改为默认 decouple=True，因此本组用的是解绑版。
#       若要复现旧 CGA，需显式构造 decouple=False（当前注册处不传该参数）。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-cga3h.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/cga_dec_chain.log
NEED=${NEED:-20000}
MAX_TRY=${MAX_TRY:-40}
WARM_WAIT=${WARM_WAIT:-360}
SLOW_IT=${SLOW_IT:-5}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

recent_it_s() {
    local f="$1"
    [ -f "$f" ] || { echo 0; return; }
    local v
    v=$(grep -aoE '[ <][0-9]+\.[0-9]+s/it' "$f" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    echo "${v:-0}"
}

wait_gpu() {
    while true; do
        f=$(free_mib)
        if [ -n "${f}" ] && [ "${f}" -ge "${NEED}" ]; then break; fi
        log "等待显存：空闲 ${f:-NA} MiB < ${NEED} MiB"
        sleep 120
    done
}

run_one() {
    local name="$1"
    local traylog="logs/${name}.log"

    [ -f "$CFG" ] || { log "致命：缺配置 $CFG"; return 2; }

    local attempt=0
    while [ "$attempt" -lt "$MAX_TRY" ]; do
        attempt=$((attempt + 1))

        local run_dir
        run_dir=$(ls -d YOLO26/ResTest/${name}* 2>/dev/null | head -1)
        if [ -n "$run_dir" ] && [ -f "$run_dir/results.csv" ]; then
            local last_ep
            last_ep=$(tail -1 "$run_dir/results.csv" | cut -d, -f1 | tr -d ' ')
            if [ "${last_ep:-0}" -ge 149 ] 2>/dev/null; then
                log "${name} 已完成（末轮 ep${last_ep}），跳过"
                return 0
            fi
            mv "$run_dir" "YOLO26/ResTest/_aborted/${name}_$(date +%H%M%S)" 2>/dev/null
            log "${name} 残留目录已挪至 _aborted"
        fi

        wait_gpu
        log "第 ${attempt}/${MAX_TRY} 次：起跑 ${name}（CGAFusion 解绑版）  空闲=$(free_mib) MiB"

        : > "$traylog"
        $PY -u train_v26.py \
            --config "$CFG" \
            --label-from dual \
            --data "$DATA" \
            --name "$name" \
            --seed 0 \
            --epochs 150 \
            --patience 30 \
            --batch 32 \
            --workers 4 \
            --amp false \
            >> "$traylog" 2>&1 &
        local pid=$!

        sleep "$WARM_WAIT"
        if kill -0 "$pid" 2>/dev/null; then
            local it
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
        local code=$?
        if [ "$code" -eq 0 ]; then
            log "${name} 完成 exit=0"
            return 0
        fi
        if grep -aq "CUDA out of memory" "$traylog"; then
            log "${name} OOM（exit=${code}），300s 后重试"
            sleep 300
            continue
        fi
        log "${name} 异常退出 exit=${code}（非 OOM）；见 ${traylog}"
        return "$code"
    done
    log "${name} 尝试次数用尽，放弃"
    return 1
}

log "=== CGAFusion 解绑版实验链启动 ==="
run_one mdp150_M0CGADec_s0
log "=== 链结束 ==="
