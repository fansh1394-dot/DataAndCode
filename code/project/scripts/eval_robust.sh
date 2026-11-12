#!/usr/bin/env bash
# 双评测：标准 test + 旋转鲁棒性 test
#
# 用途：比较基线与几何增强模型在「正常」和「几何扰动」两种条件下的掉点幅度。
#   标准 test : 原始 mendeley_split
#   旋转 test : mendeley_rot15 / rot30（由 _make_rot_testset.py 生成）
#
# 为什么两个都要跑：
#   若几何增强只提高标准 test 的 mAP，说明它只是普通正则；
#   若标准 test 持平但旋转 test 掉得更少，说明它提升了几何鲁棒性——
#   这才是「数据丰富化」真正想证明的东西。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
OUT_DIR=YOLO26/ResTest/_test_eval
LOG=logs/eval_robust.log
ANGLES=${ANGLES:-15 30}

mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run() {
    local weights="$1" tag="$2" data="$3"
    local out="${OUT_DIR}/${tag}.json"
    [ -f "$weights" ] || { log "  跳过 $tag：缺权重"; return; }
    [ -f "$out" ] && { log "  跳过 $tag：已有结果"; return; }

    log "  评测 $tag"
    $PY -u scripts/eval_split.py \
        --weights "$weights" --data "$data" --split test --labels-mode dual \
        --imgsz 640 --batch 16 --workers 4 \
        --tag "$tag" --out "$out" >> "logs/eval_${tag}.log" 2>&1
    [ -f "$out" ] && log "    $tag 完成" || log "    $tag 失败，见 logs/eval_${tag}.log"
}

log "=== 标准 + 旋转鲁棒性评测 ==="

# 待评测的模型：基线三种子 + 几何增强组
MODELS=(
  "mdp150_M0_concat_s0"
  "mdp150_M0_concat_s1"
  "mdp150_M0_concat_s2"
  "mdp150_M0GeoMild_s0"
)
[ "${WITH_STRONG:-0}" = "1" ] && MODELS+=("mdp150_M0GeoStrong_s0")

for NAME in "${MODELS[@]}"; do
    W="YOLO26/ResTest/${NAME}/weights/best.pt"
    log "-- $NAME"

    # 1) 标准 test
    run "$W" "${NAME}_test" /3250410011/root/rivermind-data/mendeley_split/data.yaml

    # 2) 各角度旋转 test
    for A in $ANGLES; do
        D="/3250410011/root/rivermind-data/mendeley_rot${A}/data.yaml"
        if [ -f "$D" ]; then
            run "$W" "${NAME}_rot${A}" "$D"
        else
            log "  跳过 rot${A}：缺数据集 $D（先跑 _make_rot_testset.py --angle $A）"
        fi
    done
done

log "=== 结束 ==="
