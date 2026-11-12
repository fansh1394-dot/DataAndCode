#!/usr/bin/env bash
# B 阶段：IR 置零敏感度测试（判断融合增益是否真的来自 IR）
#
# 判据
# ----
# 对每个权重跑四种口径的 test 评测：
#   1) 原样（基准，已有则跳过）
#   2) IR 置 0      —— 彻底抹掉 IR 信息
#   3) IR 置中灰    —— 排除「分布外黑块」的混淆：黑块可能让 BN 统计爆炸，
#                      导致下降来自分布偏移而非信息缺失
#   4) RGB 置 0     —— 反向对照，验证 RGB 通路是否正常
#
# 结论口径（论文可直接引用）：
#   ΔIR  = fuse(原样) - fuse(IR置中灰)
#   ΔRGB = fuse(原样) - fuse(RGB置0)
#   若 ΔIR ≈ 0 而 ΔRGB 显著 > 0，说明模型实际只用了 RGB，融合增益是伪增益。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/mendeley_split/data.yaml
OUT_DIR=YOLO26/ResTest/_test_eval
LOG=logs/zero_modality.log
BATCH=${BATCH:-16}

mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_eval() {
    local name="$1"; local tag="$2"; shift 2
    local W="YOLO26/ResTest/${name}/weights/best.pt"
    local OUT="${OUT_DIR}/${tag}.json"

    if [ ! -f "$W" ]; then log "  跳过 $tag：缺权重"; return; fi
    if [ -f "$OUT" ]; then log "  跳过 $tag：已有结果"; return; fi

    $PY -u scripts/eval_split.py \
        --weights "$W" \
        --data "$DATA" \
        --split test \
        --labels-mode dual \
        --imgsz 640 \
        --batch "$BATCH" \
        --workers 4 \
        --tag "$tag" \
        --out "$OUT" "$@" >> "logs/eval_${tag}.log" 2>&1
    local code=$?
    if [ "$code" -eq 0 ] && [ -f "$OUT" ]; then
        log "  $tag 完成"
    else
        log "  $tag 失败 exit=${code}，见 logs/eval_${tag}.log"
    fi
}

log "=== B 阶段：IR/RGB 置零敏感度测试 ==="

# 选 M0_concat 作主对象（当前最优、无屏蔽干扰），再补 CGAFusion 作对照
for NAME in mdp150_M0_concat_s0 mdp150_M0CGAFusion_s0; do
    log "-- $NAME"
    run_eval "$NAME" "${NAME}_test"
    run_eval "$NAME" "${NAME}_test_zir"   --zero-modality ir  --zero-fill 0.0
    run_eval "$NAME" "${NAME}_test_zirg"  --zero-modality ir  --zero-fill 128
    run_eval "$NAME" "${NAME}_test_zrgb"  --zero-modality rgb --zero-fill 0.0
done

log "=== B 阶段结束 ==="
