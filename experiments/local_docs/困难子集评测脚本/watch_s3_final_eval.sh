#!/bin/bash
# 等 S3 训练结束后，自动在最终 best.pt 上跑困难子集评测，并生成 N0/S3 同口径对比。
#
# 为什么必须后台常驻：run_remote.ps1 走 plink bash -s，命令会阻塞到远端返回；
# S3 还剩约 70 轮（实测单轮 ~780s，≈15h），不可能阻塞等待。
# 故本脚本由 launch_watch_eval.sh 用 setsid 脱离会话启动，启动后立即返回。
#
# 幂等：pidfile 记录自身 pid；run_remote.ps1 重试导致重复启动时，第二个实例直接退出。
#       评测阶段本身也靠 eval_hard.sh 的 pgrep 守卫防重复。
#
# 用法: watch_s3_final_eval.sh [TAG] [TOTAL_EPOCHS] [POLL_SEC] [MAX_WAIT_H]
set -u

ROOT=/root/rivermind-data
RES=$ROOT/MutilModel_3485988898/ResTest
L=$ROOT/logs
RUN=$RES/v2_S3_s0
TAG="${1:-S3}"
TOTAL="${2:-100}"
POLL="${3:-120}"
MAXH="${4:-24}"

mkdir -p "$L"
PIDF="$L/.watch_${TAG}.pid"
GL="$L/watch_${TAG}.log"

if [ -f "$PIDF" ]; then
    OLD=$(cat "$PIDF" 2>/dev/null || echo "")
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "ALREADY_WATCHING pid=$OLD"
        exit 0
    fi
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

# 同时落盘到 $GL：launch_watch_eval.sh 把 stdout 收进 boot log，
# 而 $GL 才是长期滚动日志（便于 tail -f / 事后取证），两边都留。
log() { echo "[$(date +%F_%T)] $*" | tee -a "$GL"; }

log "守护启动 pid=$$ TAG=$TAG 目标=$RUN 总轮数=$TOTAL 轮询=${POLL}s 上限=${MAXH}h"

rows_now() {
    if [ -f "$RUN/results.csv" ]; then
        echo $(( $(wc -l < "$RUN/results.csv") - 1 ))
    else
        echo 0
    fi
}
# 实测该模式能匹配到 13 个训练相关 pid（已验证），训练真正消失时会全部归零。
tr_alive() { pgrep -f "train_full.py.*--name v2_S3_s0" >/dev/null 2>&1; }

DEADLINE=$(( $(date +%s) + MAXH * 3600 ))
STATE="unknown"

# ---------- 1) 等训练 ----------
while :; do
    rows=$(rows_now)

    if ! tr_alive; then
        if [ "$rows" -ge "$TOTAL" ]; then
            log "训练结束，完成 $rows/$TOTAL 轮"
            STATE="full"
            break
        fi
        log "训练进程消失，仅 $rows/$TOTAL 轮，疑似中断；等 $((POLL*2))s 确认"
        sleep $((POLL * 2))
        if tr_alive; then
            log "进程已恢复，继续等待"
            continue
        fi
        log "确认中断（$rows/$TOTAL 轮）。仍用现有 best.pt 评测，结果标注为 PARTIAL。"
        STATE="partial"
        break
    fi

    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log "等待超时（${MAXH}h，当前 $rows/$TOTAL 轮）。不评测，退出。"
        exit 2
    fi

    log "等待中：$rows/$TOTAL 轮，训练存活"
    sleep "$POLL"
done

# ---------- 2) 评测 ----------
W="$RUN/weights/best.pt"
JSON="$L/hard_eval_${TAG}.json"

if [ ! -f "$W" ]; then
    log "FATAL: 找不到权重 $W"
    exit 3
fi
log "权重: $(stat -c 'size=%s mtime=%y' "$W")"
CKPT_EP=$(tail -1 "$RUN/results.csv" 2>/dev/null | cut -d, -f1)
log "best.pt 对应训练轮次(末轮)=${CKPT_EP}  训练态=${STATE}"

if [ -f "$JSON" ]; then
    BK="${JSON}.prev_$(date +%m%d_%H%M%S)"
    mv "$JSON" "$BK" && log "旧 JSON 已备份 -> $BK"
fi

peak_wait() {
    n=0
    while pgrep -f "eval_hard_subset.py" >/dev/null 2>&1; do
        n=$((n + 1))
        if [ "$n" -ge 180 ]; then
            log "评测等待超过 30min，放弃等待"
            return 1
        fi
        sleep 10
    done
    return 0
}

# JSON 完整性校验：光有文件不够，必须能解析且含 fuse 分支与全集键，
# 否则是写坏/写一半的文件，不能当成评测成功。
json_ok() {
    [ -f "$JSON" ] || return 1
    PJ="$JSON" TOK="$TAG" /opt/conda/bin/python - <<'PY' >/dev/null 2>&1
import json, os, sys
p = os.environ["PJ"]
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception:
    sys.exit(1)
if not isinstance(d, dict):
    sys.exit(1)
if "branch_map50_95_ALL" not in d or "fuse" not in d.get("branch_map50_95_ALL", {}):
    sys.exit(1)
if d.get("n_samples", 0) < 1000:
    sys.exit(1)
sys.exit(0)
PY
}

ok=0
for bs in 4 2 1; do
    log "评测启动 batch=$bs"
    bash "$ROOT/scripts/eval_hard.sh" "$W" "$TAG" "$bs" 2>&1 | sed 's/^/    /'
    sleep 20
    if pgrep -f "eval_hard_subset.py" >/dev/null 2>&1; then
        log "评测运行中，等待收尾"
        peak_wait
    fi
    if json_ok; then
        log "batch=$bs 产出有效 JSON"
        ok=1; break
    fi
    if [ -f "$JSON" ]; then
        log "batch=$bs 的 JSON 校验不通过（可能写坏），改名留证后降档重试"
        mv "$JSON" "${JSON}.bad_$(date +%m%d_%H%M%S)" 2>/dev/null
    fi
    log "batch=$bs 未产出有效 JSON，检查日志尾部"
    tail -20 "$(ls -t "$L"/eval_hard_${TAG}_*.log 2>/dev/null | head -1)" 2>/dev/null | sed 's/^/    /'
done

if [ "$ok" -ne 1 ]; then
    log "FATAL: 三次评测均未产出有效 ${JSON}"
    exit 4
fi

log "评测完成 -> $JSON"
EVA=$(PJ="$JSON" /opt/conda/bin/python -c 'import json,os;print(json.load(open(os.environ["PJ"],encoding="utf-8")).get("n_samples"))' 2>/dev/null)
log "评测样本数=$EVA"

# ---------- 3) 对比 ----------
log "生成 N0/${TAG} 同口径对比"
bash "$ROOT/scripts/compare_hard_final.sh" N0 "$TAG" 2>&1 | tee "$L/hard_compare_N0_${TAG}.txt"

# ---------- 4) 收尾摘要 ----------
{
    echo "==================== 评测收尾摘要 ===================="
    echo "TAG=${TAG}  训练状态=${STATE}  完成轮数=$(rows_now)/${TOTAL}"
    echo "评测样本数=${EVA}"
    echo "权重=$W"
    echo "JSON=$JSON"
    echo "日志=$GL"
    echo "对比=$L/hard_compare_N0_${TAG}.txt"
    echo
    echo "--- 核心判定（only_ir 组 recall，S3 内部 fuse_ir vs ir）---"
    grep -aE "核心判定|融合是否可用" "$L/hard_compare_N0_${TAG}.txt" 2>/dev/null | head -20
    echo
    echo "--- 训练最终轮（epoch,time 前两列）---"
    tail -1 "$RUN/results.csv" | cut -d, -f1,2
} > "$L/watch_${TAG}_summary.txt"

log "摘要 -> $L/watch_${TAG}_summary.txt"
log "守护结束"
