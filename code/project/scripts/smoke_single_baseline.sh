#!/bin/bash
# E1/E2 单模态基线 —— 构建 + 2 轮冒烟（先验证贯通性，再决定是否跑正式 100 轮）
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
RUN=$PROJ/_d2p1
PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/uav_data_v2/data.yaml
YAML=ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml

mkdir -p "$RUN"; cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_smoke () {
  local MOD=$1
  local NAME="_smoke_${MOD}only_baseline"
  echo ""
  echo "############################################################"
  echo "### 单模态基线冒烟：--single-modal $MOD"
  echo "############################################################"
  rm -rf "$PROJ/ResTest/$NAME"; rm -f "$RUN/$NAME.log"
  nohup "$PY" -u train_full.py \
    --config "$YAML" --single-modal "$MOD" \
    --label-from dual --data "$DATA" \
    --epochs 2 --batch 16 --imgsz 640 --workers 4 --seed 0 --safe-augment \
    --name "$NAME" --amp false \
    > "$RUN/$NAME.log" 2>&1 &
  local PID=$!
  # 最长等 20 分钟
  for i in $(seq 1 40); do
    sleep 30
    if ! kill -0 $PID 2>/dev/null; then break; fi
    if grep -qaE "epochs completed|Error|Traceback|AssertionError" "$RUN/$NAME.log" 2>/dev/null; then
      sleep 10; break
    fi
  done
  echo "--- 关键日志 ---"
  tr '\r' '\n' < "$RUN/$NAME.log" | grep -aE "single_modal|nc 对齐|Transferred|Starting training|epochs completed|all +[0-9]|Error|Traceback|ValueError|AssertionError" | head -20
  echo "--- results.csv ---"
  if [ -f "$PROJ/ResTest/$NAME/results.csv" ]; then
    wc -l "$PROJ/ResTest/$NAME/results.csv"
    tail -2 "$PROJ/ResTest/$NAME/results.csv" | cut -c1-120
  else
    echo "[无 results.csv]"
  fi
}

run_smoke rgb
# 注意：--single-modal 的 choices 中 x 即红外（见 train_full.py 的 --single-modal 定义）
run_smoke x
