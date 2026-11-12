#!/usr/bin/env bash
# C 阶段：种子方差标定
#
# 目的
# ----
# 各组间差值只有 0.02~0.05。若不标定「单种子噪声带 σ」，就无法判断
# 「CGAFusion 比 concat 低 0.0206」是真实退化还是种子波动。审稿人必问。
#
# 设计
# ----
# 对当前最优方案 M0_concat 再跑 seed=1、seed=2，与已有 seed=0 构成 3 种子。
# 三组共用同一份 YAML/数据/超参，唯一差异是 seed，因此三者离散度
# 就是该方案的种子噪声 σ。
# 判据：|A组均值差| < 2σ 视为不显著。
#
# 为什么只标定 M0_concat：
#   标定 σ 需要一个「重复的同一配置」，不需要每个算子都跑 3 次。
#   得到 σ 后可直接用于所有组间比较（假设各组噪声同量级）。
#   若后续要严格化，再对 CGAFusion 补 2 个种子。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/seed_std_chain.log
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
    local name="$1"; local seed="$2"
    local traylog="logs/${name}.log"

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
        log "第 ${attempt}/${MAX_TRY} 次：起跑 ${name}  seed=${seed}  空闲=$(free_mib) MiB"

        : > "$traylog"
        $PY -u train_v26.py \
            --config "$CFG" \
            --label-from dual \
            --data "$DATA" \
            --name "$name" \
            --seed "$seed" \
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
        log "${name} 异常退出 exit=${code}（非 OOM），放弃；见 ${traylog}"
        return "$code"
    done
    log "${name} 尝试次数用尽，放弃"
    return 1
}

log "=== C 阶段：M0_concat 种子方差标定（seed=1, seed=2）==="
run_one mdp150_M0_concat_s1 1
run_one mdp150_M0_concat_s2 2
log "=== C 阶段结束 ==="
