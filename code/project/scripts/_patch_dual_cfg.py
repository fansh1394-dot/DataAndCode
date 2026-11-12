# -*- coding: utf-8 -*-
"""补齐 DUAL_HEAD_CONFIGS：把 yolo26 的三个真实双头配置纳入白名单。

背景：yolo26n-mm-dual-fusion-concat / align3h / part3h 都用 DualHeadFusionDetectV26
聚合三头，属"双头 + 融合分支"架构。但 DUAL_HEAD_CONFIGS 只收录了 YOLO11 时代的别名，
导致给 M0 传 --mm-dual-* 参数时被误判为"单头融合"并 return 2。
此前的 M0/M3/L4 训练都没有传这类参数，所以该缺口一直未被触发。
"""
import io

PATH = "train_full.py"

OLD = (
    'DUAL_HEAD_CONFIGS = ("dual-fusion", "dual-fusion-b1", "dual-fusion-reliable", '
    '"dual-fusion-gated", "dual-fusion-deform", "dual-fusion-partition", "v26-dual-fusion-deform")'
)

NEW = '''DUAL_HEAD_CONFIGS = (
    "dual-fusion", "dual-fusion-b1", "dual-fusion-reliable", "dual-fusion-gated",
    "dual-fusion-deform", "dual-fusion-partition", "v26-dual-fusion-deform",
    # YOLO26 下同样使用 DualHeadFusionDetectV26 聚合层的真实配置文件。
    # 原先未收录：这些配置的历史训练都没有传 --mm-dual-* 参数，校验分支从未执行到，
    # 缺口未被发现；一旦传入屏蔽/蒸馏参数就会被误判为"单头融合"而拒绝。
    "yolo26n-mm-dual-fusion-concat",
    "yolo26n-mm-dual-fusion-align3h",
    "yolo26n-mm-dual-fusion-part3h",
)'''

src = io.open(PATH, encoding="utf-8").read()
assert OLD in src, "未找到 DUAL_HEAD_CONFIGS 原定义，请检查文件版本"
io.open(PATH, "w", encoding="utf-8").write(src.replace(OLD, NEW))
print("patched DUAL_HEAD_CONFIGS")
