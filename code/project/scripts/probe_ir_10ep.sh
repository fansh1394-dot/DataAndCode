#!/bin/bash
# IR-only 基线：延长到 10 轮，确认是在正常学习还是真的学不动
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
RUN=$PROJ/_d2p1
PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/uav_data_v2/data.yaml
YAML=ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml
NAME=_probe_xonly_10ep

cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for pid in $(pgrep -f "train_full.py"); do
  [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null
done
sleep 3

rm -rf "$PROJ/ResTest/$NAME"; rm -f "$RUN/$NAME.log"
nohup "$PY" -u train_full.py \
  --config "$YAML" --single-modal x \
  --label-from dual --data "$DATA" \
  --epochs 10 --batch 16 --imgsz 640 --workers 4 --seed 0 --safe-augment \
  --name "$NAME" --amp false \
  > "$RUN/$NAME.log" 2>&1 &
echo "launched $NAME"
