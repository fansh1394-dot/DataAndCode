#!/bin/bash
# ============================================================
# R1 · 已配准早期融合上界（选题四 B5，全项目 thesis 的 paper-killer）
#
# 阶段（严禁跳步）
#   fit       逐序列仿射拟合，复现「静态刚性配准不可达」的负结果（预期 rc 非 0）
#   fit-frame 逐帧平移（R1.3 oracle）天花板质检，构造性 ≤2px 应接近 100%
#   export    导出 uav_data_v2_reg（per-frame warp IR + 变换标签 + data.yaml）
#   train     等 GPU 空闲后训练 R1.1（early 6ch），可选 R1.2（dual-fusion）；detach 后台
#   eval      评 test 并打印 JSON
#   status    查看闸门状态
#
# 注：已配准上界采用 per-frame oracle —— 实测两模态位移逐帧变化（序列内 dx std 20~50px），
# 逐序列单一仿射 ≤2px 仅 11.8%，故不存在可实现的静态配准。per-frame 结果为上界参考，
# 论文中必须标注 oracle，不得与方法表并列。
#
# 用法
#   bash scripts/run_r1.sh fit
#   bash scripts/run_r1.sh fit-frame
#   bash scripts/run_r1.sh export
#   bash scripts/run_r1.sh train          # 默认只训 R1.1
#   bash scripts/run_r1.sh train all      # R1.1 + R1.2
#   bash scripts/run_r1.sh eval
#   bash scripts/run_r1.sh status
#
# 为什么 train 必须等 GPU：phase0 的 G0 重基线仍在跑（E4 峰值 18.1~18.5G），
# 与 R1 并行会 OOM 或互相拖慢，故默认排队等待，而不是抢占。
# ============================================================
set -uo pipefail

PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
ROOT=/root/rivermind-data/uav_data_v2
REG=/root/rivermind-data/uav_data_v2_reg
STAGE=${1:-fit}
WHAT=${2:-r11}
LOG=$PROJ/_d2p1/run_r1.log
PIDF=$PROJ/_d2p1/run_r1_train.pid

cd "$PROJ" || exit 1
mkdir -p "$PROJ/_d2p1" "$PROJ/ResTest/_test_eval"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MM_PROJ="$PROJ"

say() { echo "[$(date '+%F %T')] $*"; }

case "$STAGE" in
# ---------------------------------------------------------------- fit
fit)
  # 逐序列仿射（原 R1.0 规格）实测 ≤2px 仅 11.8%，作为负结果复现
  say "R1.0 per-seq fit-only（负结果复现）"
  "$PY" -u scripts/reg_export_v2.py --root "$ROOT" --out "$REG" \
        --splits train,val,test --mode per-seq --fit l2 --ransac 3.0 --min-pairs 20 --fit-only \
        2>&1 | tee "$PROJ/_d2p1/r1_fit_perseq.log"
  rc=${PIPESTATUS[0]}
  say "per-seq fit rc=$rc（预期非 0：验收线不可达）"
  exit 0
  ;;

# ------------------------------------------------------- fit-frame（天花板）
fit-frame)
  # R1.3 oracle：逐帧平移，构造性对齐，只作已配准上界参考，不得入方法表
  say "R1.3 per-frame fit-only 天花板质检"
  "$PY" -u scripts/reg_export_v2.py --root "$ROOT" --out "$REG" \
        --splits train,val,test --mode per-frame --fit l2 --min-pairs 20 --fit-only \
        2>&1 | tee "$PROJ/_d2p1/r1_fit_frame.log"
  rc=${PIPESTATUS[0]}
  say "per-frame fit rc=$rc"
  exit "$rc"
  ;;

# ------------------------------------------------------------- export
export)
  say "R1 导出 uav_data_v2_reg（per-frame oracle）"
  "$PY" -u scripts/reg_export_v2.py --root "$ROOT" --out "$REG" \
        --splits train,val,test --mode per-frame --fit l2 --min-pairs 20 --workers 48 \
        2>&1 | tee "$PROJ/_d2p1/r1_export.log"
  rc=${PIPESTATUS[0]}
  say "export rc=$rc"
  [ -f "$REG/data.yaml" ] && say "data.yaml OK" || say "缺 data.yaml，导出失败"
  exit "$rc"
  ;;

# -------------------------------------------------------------- train
train)
  if [ "${R1_DETACHED:-0}" != "1" ]; then
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      say "已有 R1 训练排队进程 pid=$(cat "$PIDF")，先看 status"
      exit 0
    fi
    R1_DETACHED=1 WHAT="$WHAT" nohup setsid bash "$0" train "$WHAT" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    say "已后台排队 pid=$(cat "$PIDF")，日志 $LOG"
    exit 0
  fi

  say "R1 训练排队中：等待 GPU 空闲（无 train_full.py 进程）"
  for i in $(seq 1 5760); do   # 最长 48 小时
    if ! pgrep -f "train_full.py" > /dev/null 2>&1; then break; fi
    sleep 30
  done
  if pgrep -f "train_full.py" > /dev/null 2>&1; then
    say "等待超时，仍有训练进程；退出"
    exit 3
  fi
  sleep 60
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  say "GPU 空闲 ${free_mb} MiB"
  if [ "${free_mb:-0}" -lt 20000 ]; then
    say "显存不足 20000 MiB，退出（请先清理外部进程）"
    exit 4
  fi

  if [ ! -f "$REG/data.yaml" ]; then
    say "缺 $REG/data.yaml，先跑 export"
    exit 5
  fi

  # ---------- R1.1 已配准早期融合（B5）
  say "开始 R1.1：early 6ch on registered data"
  rm -rf "$PROJ/ResTest/v2c_R1_reg_early_s0"
  "$PY" -u train_full.py \
    --config ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-early.yaml \
    --data "$REG/data.yaml" --label-from dual \
    --epochs 150 --batch 32 --imgsz 640 --workers 4 --safe-augment --seed 0 \
    --amp false --device 0 --route A --name v2c_R1_reg_early_s0 \
    > "$PROJ/_d2p1/r1_early.log" 2>&1
  say "R1.1 训练结束 rc=$?"

  W="$PROJ/ResTest/v2c_R1_reg_early_s0/weights/best.pt"
  if [ -f "$W" ]; then
    say "R1.1 test 评测"
    "$PY" -u scripts/eval_split.py --weights "$W" --data "$REG/data.yaml" \
      --split test --labels-mode vis --tag v2c_R1_reg_early_s0_test \
      --batch 32 --device 0 --workers 4 > "$PROJ/_d2p1/r1_early_eval.log" 2>&1
    say "R1.1 评测 rc=$?"
    cat "$PROJ/ResTest/_test_eval/v2c_R1_reg_early_s0_test.json" 2>/dev/null
  else
    say "R1.1 缺 best.pt，跳过评测"
  fi

  [ "$WHAT" = "all" ] || { say "R1 训练流程结束（未含 R1.2）"; exit 0; }

  # ---------- R1.2 已配准双头融合
  say "开始 R1.2：dual-fusion on registered data"
  rm -rf "$PROJ/ResTest/v2c_R1b_reg_dual_s0"
  "$PY" -u train_full.py \
    --config ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-dual-fusion.yaml \
    --data "$REG/data.yaml" --label-from dual \
    --epochs 150 --batch 32 --imgsz 640 --workers 4 --safe-augment --seed 0 \
    --amp false --device 0 --route A --name v2c_R1b_reg_dual_s0 \
    > "$PROJ/_d2p1/r1_dual.log" 2>&1
  say "R1.2 训练结束 rc=$?"

  W2="$PROJ/ResTest/v2c_R1b_reg_dual_s0/weights/best.pt"
  if [ -f "$W2" ]; then
    "$PY" -u scripts/eval_split.py --weights "$W2" --data "$REG/data.yaml" \
      --split test --labels-mode vis --tag v2c_R1b_reg_dual_s0_test \
      --batch 32 --device 0 --workers 4 > "$PROJ/_d2p1/r1_dual_eval.log" 2>&1
    cat "$PROJ/ResTest/_test_eval/v2c_R1b_reg_dual_s0_test.json" 2>/dev/null
  fi
  say "R1 全流程结束"
  ;;

# --------------------------------------------------------------- eval
eval)
  for tag in v2c_R1_reg_early_s0_test v2c_R1b_reg_dual_s0_test; do
    echo "--- $tag ---"
    cat "$PROJ/ResTest/_test_eval/$tag.json" 2>/dev/null || echo "(缺失)"
  done
  ;;

# ------------------------------------------------------------- status
status)
  echo "=== R1 排队/训练进程 ==="
  [ -f "$PIDF" ] && { echo -n "pidfile: "; cat "$PIDF"; } || echo "pidfile: (无)"
  pgrep -af "run_r1.sh|train_full.py" || echo "(无相关进程)"
  echo
  echo "=== 配准产物 ==="
  ls -ld "$REG" 2>/dev/null || echo "(未导出)"
  [ -f "$REG/_reg_meta.json" ] && "$PY" - "$REG/_reg_meta.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get('summary', {})
a = d.get('acceptance', {})
print(f"  both帧={s.get('n_both_frames')} 配准后均值={s.get('frame_err_mean_px'):.3f}px "
      f"identity={s.get('identity_err_mean_px'):.2f}px")
print(f"  ≤2px 帧占比={a.get('frame_ok_frac_2px', 0)*100:.2f}%  pass={a.get('pass')}")
EOF
  echo
  echo "=== test 结果 ==="
  for t in v2c_R1_reg_early_s0_test v2c_R1b_reg_dual_s0_test; do
    [ -f "$PROJ/ResTest/_test_eval/$t.json" ] && {
      echo -n "  $t: ";
      "$PY" -c "import json;d=json.load(open('$PROJ/ResTest/_test_eval/$t.json'));print(d['metrics'].get('metrics/mAP50-95(B)'))"
    } || echo "  $t: (缺失)"
  done
  ;;
*)
  echo "用法: bash scripts/run_r1.sh {fit|fit-frame|export|train [r11|all]|eval|status}"
  exit 1
  ;;
esac
