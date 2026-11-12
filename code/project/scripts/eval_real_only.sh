#!/usr/bin/env bash
# 纯真实帧（is_synthetic=False）test 评测，与全量 test 对照
#
# 目的：量化合成数据带来的指标虚高。
#   Δ = 全量指标 − 纯真实指标
#   若 Δ 明显为正，说明论文里报的全量数字被合成帧抬高了。
set -u

cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1

PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/mendeley_split/data.yaml
# 列表必须放在数据集根目录下：check_det_dataset 会把 val 的相对路径
# 拼到 data.yaml 的 path 上，放到工程目录下会报 images not found。
LIST=lists/test_real_only.txt
DS=/3250410011/root/rivermind-data/mendeley_split
OUT_DIR=YOLO26/ResTest/_test_eval
LOG=logs/eval_real_only.log

[ -f "$DS/$LIST" ] || { echo "缺 $DS/$LIST，先跑 _make_real_only_list.py"; exit 2; }
mkdir -p "$OUT_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# M0_concat 三种子 + CGAFusion，覆盖基线与最优算子
for NAME in mdp150_M0_concat_s0 mdp150_M0_concat_s1 mdp150_M0_concat_s2 mdp150_M0CGAFusion_s0; do
    W="YOLO26/ResTest/${NAME}/weights/best.pt"
    OUT="${OUT_DIR}/${NAME}_test_real.json"
    [ -f "$W" ] || { log "跳过 $NAME：缺权重"; continue; }
    [ -f "$OUT" ] && { log "跳过 $NAME：已有"; continue; }

    log "评测（纯真实帧）$NAME"
    $PY -u scripts/eval_split.py \
        --weights "$W" --data "$DATA" --split test --labels-mode dual \
        --val-list "$LIST" --imgsz 640 --batch 16 --workers 4 \
        --tag "${NAME}_test_real" --out "$OUT" >> "logs/eval_${NAME}_real.log" 2>&1
    code=$?
    [ "$code" -eq 0 ] && [ -f "$OUT" ] && log "  $NAME 完成" || log "  $NAME 失败 exit=${code}"
done

log "=== 结束 ==="
