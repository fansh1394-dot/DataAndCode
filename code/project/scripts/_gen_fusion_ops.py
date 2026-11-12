# -*- coding: utf-8 -*-
"""生成并校验三个融合算子替换 YAML：CGAFusion / PIM / CMCE。

替换策略（与 M0 保持结构完全一致，只换融合算子）：
  M0 融合颈是「Concat -> 1x1 Conv 压回原通道 -> C3k2」。
  本脚本只把 Concat 换成融合算子，保留后置 1x1 Conv。
  这样层数不变、层号不变、neck 宽度不变，唯一的差异就是融合算子本身。

参数口径（依据 tasks.py parse_model 的两条注册分支，已在服务器核实）：
  * 跨模态分支 {CIDAF, CGAFusion, DAF, DynamicAlignFusion, WDAF}
      ouc = args[0]（缺省 c_left），然后 ouc = make_divisible(min(ouc, max_channels) * width, 8)
      由于 ch[] 已经是缩放后的通道（n 尺度 P3=64/P4=128/P5=256），
      必须传「未缩放的」256/512/1024，才能被 width=0.25 缩放回 64/128/256。
      CGAFusion 的 args[1] 是 reduction。
  * 纯特征融合分支 {PIM, CMCE, WCMF, GFM, ...}
      len(args)==0 时自动 args[0]=c_left，且 c2=c_left；
      该分支不再乘 width，所以传空列表 [] 才是正确口径。
"""
import sys
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

BASE = ROOT / "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml"

# (算子名, 每个尺度对应的 args, 输出 YAML 文件名)
CANDS = [
    ("CGAFusion", [[256, 8], [512, 8], [1024, 8]], "yolo26n-mm-dual-fusion-cga3h.yaml"),
    ("PIM", [[], [], []], "yolo26n-mm-dual-fusion-pim3h.yaml"),
    ("CMCE", [[], [], []], "yolo26n-mm-dual-fusion-cmce3h.yaml"),
]

# 绝对层号 46/49/52，但 YAML 的 head 从绝对层 22 开始，因此 head 下标 = 绝对层号 - 22。
# 不硬编码 head 下标，改为按「两个非负前驱」的 Concat 自动定位，更稳。
HEAD_OFFSET = 22


def _fusion_indices(head) -> list:
    idx = [
        i for i, ln in enumerate(head)
        if len(ln) >= 3 and ln[2] == "Concat" and isinstance(ln[0], list)
        and len(ln[0]) == 2 and all(isinstance(x, int) and x >= 0 for x in ln[0])
    ]
    assert len(idx) == 3, f"融合 Concat 定位异常: {idx}"
    for i in idx:
        nxt = head[i + 1]
        assert nxt[2] == "Conv", f"融合节点 {i + HEAD_OFFSET} 之后不是 1x1 Conv: {nxt[2]}"
        # Conv 的 args 是 [out_ch, k, s]，故以 args[1:3] 判断是否为 1x1
        assert nxt[3][1:3] == [1, 1], f"融合节点之后不是 1x1 Conv: {nxt[3]}"
    return idx


def gen(op: str, arg_per_scale: list, out_name: str) -> Path:
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    head = cfg["head"]
    for k, hi in enumerate(_fusion_indices(head)):
        cur = head[hi]
        head[hi] = [cur[0], 1, op, arg_per_scale[k]]
    out = BASE.parent / out_name
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def check(op, yaml_path):
    print(f"\n===== {op} =====", flush=True)
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        m = DetectionModel(cfg, ch=6, nc=1, verbose=False)
        n_par = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  构建 OK  layers={len(m.model)}  参数量={n_par:.4f} M", flush=True)

        idx = [i for i, lay in enumerate(m.model) if isinstance(lay, torch.nn.Module)
               and type(lay).__name__ in (op,)]
        print(f"  算子层号={idx}", flush=True)

        # 逐尺度通道核对：融合算子输出必须等于其后 1x1 Conv 的输入
        for i in (46, 49, 52):
            nxt = m.model[i + 1]
            cin = getattr(getattr(nxt, "conv", None), "in_channels", None)
            print(f"  L{i}({type(m.model[i]).__name__}) -> L{i+1}({type(nxt).__name__}) in_ch={cin}", flush=True)

        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 6, 256, 256))
        print(f"  前向 OK  type={type(out).__name__}", flush=True)
        return True, n_par
    except Exception:
        traceback.print_exc()
        return False, 0.0


if __name__ == "__main__":
    print("[1] 生成 YAML", flush=True)
    paths = []
    for op, args, name in CANDS:
        p = gen(op, args, name)
        print(f"  {p.name}  <- {op}", flush=True)
        paths.append((op, p))
    print("[2] 构建校验", flush=True)
    for op, p in paths:
        ok, n_par = check(op, p)
        print(f"  => {op}: {'OK' if ok else 'FAIL'} ({n_par:.4f} M)", flush=True)
