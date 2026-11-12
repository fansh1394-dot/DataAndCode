#!/usr/bin/env bash
# 融合算子替换实验链：CGAFusion -> PIM -> CMCE（显存守候 + 速度哨兵）
#
# 实验设计
# --------
# 与 mdp150_M0_concat_s0 完全同一份配方（150 轮 / batch 32 / seed 0 / 无屏蔽 / 无蒸馏），
# 唯一差异是把 M0 融合颈的三处 Concat 换成对应融合算子：
#   L46(P3) / L49(P4) / L52(P5)，保留其后的 1x1 Conv，层数与层号不变。
# 因此这三组可以直接和 M0_concat 比，用于回答「换融合算子能否超过朴素拼接」。
#
# 为什么不开 mm-fusion-rgb-mask / 蒸馏
# ----------------------------------
# 那两项是为了让 IR 旁路获得梯度而加的「训练技巧」。若本组也开，就同时改了
# 两个变量，无法把增益归因到融合算子本身。算子层面对照应保持干净。
# （若要跑「算子 + 屏蔽」的加强臂，需要先把 tasks.py 的 _is_fusion_node 扩到
#   识别这六类算子，否则屏蔽会静默失效。）
#
# 参数量（n 尺度，实测）
#   M0 concat 5.9020 M | CGAFusion 5.9691 M | PIM 6.1618 M | CMCE 5.8160 M
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

BASE=ultralytics/cfg/models/mm/detect/yolo26/base
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/fusion_ops_chain.log
NEED=${NEED:-20000}          # 起跑所需空闲显存（MiB）
MAX_TRY=${MAX_TRY:-40}
WARM_WAIT=${WARM_WAIT:-360}  # 起跑后观察多久再判速度（秒）
SLOW_IT=${SLOW_IT:-5}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

# 进度条同时存在 "1.20it/s" 与 "1.34s/it"，前者也含 "s/it" 子串，
# 必须用前导分隔符约束，否则取到空值。
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
    local cfg="$2"
    local traylog="logs/${name}.log"

    if [ ! -f "$cfg" ]; then
        log "致命：缺少配置 $cfg，跳过 ${name}"
        return 2
    fi

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
            log "${name} 残留未完成目录已挪至 _aborted"
        fi

        wait_gpu
        log "第 ${attempt}/${MAX_TRY} 次：起跑 ${name}  配置=$(basename "$cfg")  空闲=$(free_mib) MiB"

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
            >> "$traylog" 2>&1 &
        local pid=$!

        sleep "$WARM_WAIT"
        if kill -0 "$pid" 2>/dev/null; then
            local it
            it=$(recent_it_s "$traylog")
            log "  速度检查：s/it=${it}"
            if awk "BEGIN{exit !(${it:-0} > ${SLOW_IT})}"; then
                log "  s/it=${it} > ${SLOW_IT}，判定显存被抢占，杀掉重来"
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
        log "${name} 异常退出 exit=${code}（非 OOM），放弃本组；见 ${traylog}"
        return "$code"
    done
    log "${name} 尝试次数用尽（${MAX_TRY}），放弃"
    return 1
}

log "=== 融合算子替换链启动：CGAFusion -> PIM -> CMCE ==="
run_one mdp150_M0CGAFusion_s0 "${BASE}/yolo26n-mm-dual-fusion-cga3h.yaml"
run_one mdp150_M0PIM_s0        "${BASE}/yolo26n-mm-dual-fusion-pim3h.yaml"
run_one mdp150_M0CMCE_s0       "${BASE}/yolo26n-mm-dual-fusion-cmce3h.yaml"
log "=== 融合算子链结束 ==="
