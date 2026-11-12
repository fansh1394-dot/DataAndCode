# -*- coding: utf-8 -*-
"""逐步复现 M0 融合节点定位过程，打印每一步的判定，定位失败原因。"""
import sys

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

from ultralytics.nn.tasks import DetectionModel, yaml_model_load

CFG = "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml"

# 直接解析 YAML 层定义，复现 ch[] 推断，不依赖已有模型
from ultralytics.nn.tasks import parse_model  # noqa: E402

m = DetectionModel(yaml_model_load(CFG), ch=3, nc=1, verbose=False)
inner = m.model
agg = inner[-1]
ft = agg.fuse_detect

print("fuse_detect.f =", ft.f)
for src in ft.f:
    print(f"\n=== 从 fuse 头输入 {src} 回溯 ===")
    cur = src
    for hop in range(4):
        layer = inner[cur] if 0 <= cur < len(inner) else None
        if layer is None:
            print(f"  hop{hop}: 层 {cur} 越界，停止")
            break
        pf = getattr(layer, "f", None)
        print(f"  hop{hop}: 层 {cur} = {type(layer).__name__} f={pf}")
        if isinstance(pf, int):
            nxt = (cur - 1) if pf == -1 else pf
        elif isinstance(pf, (list, tuple)) and len(pf) == 1:
            nxt = pf[0]
        else:
            print("    -> 多输入层，无法沿单链继续")
            break
        print(f"    -> 下一层 {nxt} = {type(inner[nxt]).__name__}  f={getattr(inner[nxt], 'f', None)}")

        # 检查融合节点判定要素
        c = inner[nxt]
        if type(c).__name__ == "Concat":
            print(f"    ** 命中 Concat，其 f={c.f} **")
            f = c.f
            if isinstance(f, (list, tuple)) and len(f) == 2:
                print(f"       两路输入层: {f[0]}={type(inner[f[0]]).__name__}, {f[1]}={type(inner[f[1]]).__name__}")
        cur = nxt
