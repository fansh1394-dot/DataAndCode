#!/usr/bin/env bash
# M0 + B3 蒸馏实验链（显存守候版）
#
# 设计要点
# --------
# 1) 两组对照，唯一差异是蒸馏权重：
#      M0mask      蒸馏 0.0  -> 对照
#      M0maskDist  蒸馏 1.0  -> 实验
#    两组都必须开 mm-fusion-rgb-mask>0：B3 蒸馏的触发条件是
#    head._mm_fusion_drop_active=True，不开屏蔽则蒸馏永不生效。
#
# 2) 为什么需要守候 + 速度哨兵：
#    本机 GPU 与他方共享。实测出现过「启动时空闲 22 GB，起跑后被外部进程挤到只剩
#    463 MiB」的情况，单 batch 从 0.7 s 劣化到 27~53 s（慢 40-70 倍），150 轮要跑
#    100 小时以上。因此在 (a) 起跑前要求空闲显存达标；(b) 起跑后 6 分钟内检查首轮
#    速度，过慢则主动杀掉并回到守候，避免白烧。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

CFG=ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml
DATA=/3250410011/root/rivermind-data/mendeley_split/data.yaml
PY=/opt/conda/envs/rivermind/bin/python
LOG=logs/distill_chain.log
NEED=${NEED:-20000}      # 起跑所需空闲显存（MiB）
MAX_TRY=${MAX_TRY:-40}   # 单组最大起跑尝试次数
WARM_WAIT=${WARM_WAIT:-360}  # 起跑后观察多久再判速度（秒）
SLOW_IT=${SLOW_IT:-5}    # 首轮平均 s/it 超过该值即判为「被抢占」，杀掉重来

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

# 取训练日志里最近一个 "x/144 [mm:ss<..., N.NNs/it]" 的 s/it 值
recent_it_s() {
    local f="$1"
    [ -f "$f" ] || { echo 0; return; }
    grep -aoE '[0-9.]+s/it' "$f" | tail -1 | tr -d 's/it' || echo 0
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
    local name="$1"; local distill="$2"
    local traylog="logs/${name}.log"

    local attempt=0
    while [ "$attempt" -lt "$MAX_TRY" ]; do
        attempt=$((attempt + 1))

        # 已完成则跳过（results.csv 末轮到达 epochs 视为完成）
        local run_dir
        run_dir=$(ls -d YOLO26/ResTest/${name}* 2>/dev/null | head -1)
        if [ -n "$run_dir" ] && [ -f "$run_dir/results.csv" ]; then
            local last_ep
            last_ep=$(tail -1 "$run_dir/results.csv" | cut -d, -f1 | tr -d ' ')
            if [ "${last_ep:-0}" -ge 149 ] 2>/dev/null; then
                log "${name} 已完成（末轮 ep${last_ep}），跳过"
                return 0
            fi
            # 未完成的残留目录：保留权重，用 --resume 续训不在本脚本范围，先挪走
            mv "$run_dir" "YOLO26/ResTest/_aborted/${name}_$(date +%H%M%S)" 2>/dev/null
            log "${name} 残留未完成目录已挪至 _aborted"
        fi

        wait_gpu
        log "第 ${attempt}/${MAX_TRY} 次：起跑 ${name}  distill=${distill}  空闲=$(free_mib) MiB"

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
            --mm-fusion-rgb-mask 0.3 \
            --mm-fusion-rgb-mask-scope sample \
            --mm-dual-w-fuse-ir-distill "$distill" \
            >> "$traylog" 2>&1 &
        local pid=$!

        # 速度哨兵：等 WARM_WAIT 秒后看首轮 s/it
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

log "=== B3 蒸馏消融链（守候版）启动：M0mask(对照) -> M0maskDist(实验) ==="
run_one mdp150_M0mask_s0 0.0
run_one mdp150_M0maskDist_s0 1.0
log "=== 链结束 ==="
