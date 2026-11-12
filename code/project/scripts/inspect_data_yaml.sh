#!/bin/bash
D=/root/rivermind-data/uav_data_v2
echo "=== 当前 data.yaml（mtime: $(stat -c %y "$D/data.yaml")） ==="
cat "$D/data.yaml"
echo ""
echo "=== 与 bak_label 的差异 ==="
diff "$D/data.yaml" "$D/data.yaml.bak_label"
echo ""
echo "=== labels_ir 抽样：真非空数（用 find -size +0） ==="
echo "  labels_ir/train 非空: $(find "$D/labels_ir/train" -name '*.txt' -size +0c 2>/dev/null | wc -l) / 29716"
echo "  labels/train    非空: $(find "$D/labels/train" -name '*.txt' -size +0c 2>/dev/null | wc -l) / 29716"
echo ""
echo "=== 同一样本 stem 在两套标签下的内容对比 ==="
S=20190926_095902_1_4_00460
echo "--- labels/$S.txt ---"; cat "$D/labels/train/$S.txt" 2>/dev/null || echo "[缺]"
echo "--- labels_ir/$S.txt ---"; cat "$D/labels_ir/train/$S.txt" 2>/dev/null || echo "[缺]"
