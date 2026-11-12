#!/bin/bash
# D3-P2 判别实验：冻结骨干+融合颈，只训 fuse 头，且强制融合节点输出全部来自 IR。
#
# 设计要点（与前六条路线本质不同）：
#   - 融合节输出被整体替换为 ir_attended（原位填充，幅值/语义保持 IR 分布），
#     下游 3x3 卷积无从绕过 —— 这解决了 D3-P1「通道硬划分被下游卷积抹平」的问题。
#   - 冻结所有非 fuse 头参数（--mm-freeze-probe-scope head），使「特征不可用」与
#     「头学不动」彻底分离。
#   - 训练与验证均生效（不像 D2-P1b 只在训练时扰动）。
#
# 判读：
#   fuse mAP50-95 >> 地板（>0.15）-> IR 特征可用，瓶颈在训练信号分配
#   fuse mAP50-95 ≈ 0.001        -> IR 特征对 fuse 头不可用
PROJ=/3250410011/root/rivermind-data/MutilModel_3485988898
RUN=$PROJ/_d2p1
PY=/opt/conda/envs/rivermind/bin/python
DATA=/root/rivermind-data/uav_data_v2/data.yaml
NAME=v2_D3P2_forceir_headonly_s0

mkdir -p "$RUN"; cd "$PROJ"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========== 清理残留 ==========="
for pid in $(pgrep -f "train_full.py"); do
  if [ "$pid" != "$$" ]; then kill -9 "$pid" 2>/dev/null || true; fi
done
sleep 5
echo "残留训练进程: $(pgrep -f train_full.py | wc -l)"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo ""
echo "=========== 启动 $NAME ==========="
rm -rf "$PROJ/ResTest/$NAME"; rm -f "$RUN/$NAME.log"
nohup "$PY" -u train_full.py \
  --route C --pretrained ResTest/v2_C4_deform_s0/weights/best.pt \
  --label-from dual --data "$DATA" \
  --epochs 40 --batch 16 --imgsz 640 --workers 4 --seed 0 --safe-augment \
  --config dual-fusion-partition --name "$NAME" \
  --amp false \
  --mm-force-ir-fusion \
  --mm-freeze-for-fusion-probe --mm-freeze-probe-scope head \
  > "$RUN/$NAME.log" 2>&1 &
echo "launched $NAME"
sleep 120
echo ""
echo "=========== 启动日志 ==========="
tr '\r' '\n' < "$RUN/$NAME.log" | grep -aE "nc 对齐|Transferred|融合颈|ChannelPartition|D3-P2|冻结|Starting training|Error|Traceback|assert" | head -25
