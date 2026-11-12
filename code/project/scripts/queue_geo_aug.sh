#!/usr/bin/env bash
# 几何增强实验链（新实例 端口 30453）
#
# 唯一变量：几何增强强度。其余与 mdp150_M0_concat_s0 完全一致。
#   基线: degrees=0,   shear=0,  perspective=0     <- 已有结果，不再重跑
#   mild: degrees=10,  shear=5,  perspective=5e-4
#   strong（可选）: degrees=20, shear=10, perspective=1e-3
#
# 判定门槛：基线三种子 μ=0.7551，2σ=0.0188 -> 必须 > 0.7739 才算显著提升。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

BASE=ultralytics/cfg/models/mm/detect/yolo26/base
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/geo_chain.log
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

# $1=组名 $2=配置文件名 $3=degrees $4=shear $5=perspective
run_one() {
    local name="$1" cfgname="$2" deg="$3" shr="$4" per="$5"
    local cfg="$BASE/$cfgname"
    local traylog="logs/${name}.log"

    [ -f "$cfg" ] || { log "致命：缺配置 $cfg"; return 2; }

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
        log "第 ${attempt}/${MAX_TRY} 次：起跑 ${name}  geo(deg=${deg},shear=${shr},persp=${per})  空闲=$(free_mib) MiB"

        : > "$traylog"
        $PY -u train_v26.py \
            --config "$cfg" \
            --label-from dual \
            --data "$DATA" \
            --name "$name" \
            --seed 0 \
            --epochs 150 \
            --patience 30 \
            --batch 32 \
            --workers 4 \
            --amp false \
            --degrees "$deg" \
            --shear "$shr" \
            --perspective "$per" \
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

log "=== 几何增强实验链启动 ==="
run_one mdp150_M0GeoMild_s0   yolo26n-mm-dual-fusion-geo_mild.yaml   10 5 0.0005
if [ "${WITH_STRONG:-0}" = "1" ]; then
    run_one mdp150_M0GeoStrong_s0 yolo26n-mm-dual-fusion-geo_strong.yaml 20 10 0.001
fi
log "=== 链结束 ==="
