#!/bin/bash
# =============================================================================
# 迁移前冻结：停掉 s1 / 停掉第二批链 / 冻结 s1 可续训 ckpt / 启动 s2 冻结式续训
# 所有步骤幂等，可重复执行（链已停则跳过，s1 已冻结则重复冻结无副作用）
# =============================================================================
set -u
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
PY=/opt/conda/envs/rivermind/bin/python
cd "$PROJ" || exit 1

echo "=========== 1. 停止 phase0 第二批链 ==========="
pkill -f 'phase0_chain.sh 2' && echo "已终止链 ctrl 进程" || echo "链未运行"
echo "--- 链进程（应无输出）---"
pgrep -af '[p]hase0_chain.sh' || echo clean

echo "=========== 2. 停止 s1 训练进程 ==========="
"$PY" scripts/force_stop.py v2c_E1_rgbonly_s1 2>&1 | grep -E '^\[kill\]'
echo "--- train_full 进程（应无输出）---"
pgrep -af '[t]rain_full.py' || echo clean

echo "=========== 3. 冻结 s1 ckpt ==========="
"$PY" scripts/freeze_run.py --name v2c_E1_rgbonly_s1 --only-ckpt 2>&1 | grep -E '^\['

echo "=========== 4. 用冻结 ckpt 续训 s2 ==========="
RUN=v2c_E1_rgbonly_s2
rm -rf "ResTest/$RUN"
LOG=YOLO26/logs/${RUN}_resume.log
mkdir -p YOLO26/logs
nohup setsid "$PY" -u train_full.py --route A \
  --config ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-single-baseline.yaml \
  --single-modal rgb --label-from dual \
  --data /root/rivermind-data/uav_data_v2/data.yaml \
  --epochs 150 --batch 32 --imgsz 640 --workers 4 \
  --seed 2 --safe-augment --name "$RUN" --amp false \
  > "$LOG" 2>&1 &
echo "  已启动 pid=$! log=$LOG"

echo "=========== 5. 收尾状态 ==========="
sleep 90
echo "--- 进程 ---"
pgrep -af '[t]rain_full.py' | cut -c1-160 || echo none
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "--- ckpt ---"
ls -l --time-style=+%H:%M:%S ResTest/v2c_E1_rgbonly_s1/weights/ | tail -3
echo "--- 续训日志尾部 ---"
tr '\r' '\n' < "$LOG" | grep -vE 'Warning|warn' | tail -12
