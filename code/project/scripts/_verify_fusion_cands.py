# -*- coding: utf-8 -*-
"""六个候选融合算子的构建校验（修正版）。

上一版失败原因：把「Concat + 1x1 Conv」两行合并成一行后，后续所有层的层号前移，
但 YAML 里的前向引用索引（f 字段）没有同步改写，导致 IndexError / 越界。
本版做法：重建整份 head，逐层改写索引。

注意：不改变层数更安全 —— 只把 Concat 换成融合算子，保留其后的 1x1 Conv 不动。
这样层号完全不变，只验证「算子本身能否构建与运行」。
真正训练时若要省掉 1x1，再单独处理索引重映射。
"""
import sys
import tempfile
import traceback

sys.path.insert(0, "/3250410011/root/rivermind-data/MutilModel_3485988898")

import torch
import yaml

from ultralytics.nn.tasks import DetectionModel, yaml_model_load

BASE = "ultralytics/cfg/models/mm/detect/yolo26/base/yolo26n-mm-dual-fusion-concat.yaml"

# 融合颈三段 Concat 分别在 head 内的位置。
#
# 参数口径（依据 tasks.py 注册分支，务必按此写，否则会像上一版一样报错）：
#   * 「跨模态双输入」分支 (CIDAF/CGAFusion/DAF/DynamicAlignFusion/WDAF)
#       ouc = args[0]（缺省取 c_left），随后 ouc = make_divisible(min(ouc, max_channels) * width, 8)
#       => 直接传 c_left 的原始值即可，不要手工乘 width。
#       CGAFusion 的 args[1] 是 reduction，不是 out_dim。
#   * 「纯特征融合」分支 (WCMF/PIM/CMCE/GFM/...)
#       args[0] 缺省取 c_left，输出 c2 = c_left。
#       => 传空列表 [] 让注册逻辑自动推断，最安全。
CANDS = [
    ("CGAFusion", "CGAFusion", [256, 8]),   # [ouc, reduction]
    ("WCMF", "WCMF", []),
    ("WDAF", "WDAF", [256]),                # 跨模态分支：args[0]=ouc
    ("PIM", "PIM", []),
    ("CMCE", "CMCE", []),
    ("GFM", "GFM", []),
]


def build_yaml(op: str, arg_tmpl: list, oucs=(256, 512, 1024)) -> str:
    cfg = yaml.safe_load(open(BASE, encoding="utf-8"))
    head = cfg["head"]

    fus = [i for i, ln in enumerate(head)
           if len(ln) >= 3 and ln[2] == "Concat"
           and isinstance(ln[0], list) and all(isinstance(x, int) and x >= 0 for x in ln[0])]
    assert len(fus) == 3, f"融合 Concat 定位异常: {fus}"

    for k, i in enumerate(fus):
        ouc = oucs[k]
        args = [ouc if a == 256 else a for a in arg_tmpl]
        head[i] = [head[i][0], 1, op, args]

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, tmp, allow_unicode=True, sort_keys=False)
    tmp.close()
    return tmp.name


def check(label, op, arg_tmpl):
    print(f"\n===== {label} =====", flush=True)
    try:
        path = build_yaml(op, arg_tmpl)
        m = DetectionModel(yaml_model_load(path), ch=3, nc=1, verbose=False)
        agg = m.model[-1]
        n_param = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  构建 OK  layers={len(m.model)}  branches={agg.branch_order}", flush=True)
        print(f"  参数量 {n_param:.4f} M  (M0 基线 5.90 M, Δ={n_param - 5.90:+.4f})", flush=True)

        idx = [i for i, lay in enumerate(m.model) if getattr(lay, "_mm_fusion_concat", False)]
        print(f"  融合节点定位 {idx}" if idx else "  ⚠ 融合节点未定位（屏蔽/蒸馏会失效）", flush=True)

        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 6, 256, 256))
        print(f"  前向 OK  keys={list(out.keys()) if isinstance(out, dict) else type(out).__name__}", flush=True)
        return True
    except Exception as e:
        print(f"  ✗ 失败 {type(e).__name__}: {str(e)[:180]}", flush=True)
        return False


if __name__ == "__main__":
    import signal

    def _timeout(signum, frame):
        raise TimeoutError("单个算子校验超过 120s，判定为卡死")

    ok, fail = [], []
    for label, op, arg in CANDS:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(120)
        try:
            r = check(label, op, arg)
        except TimeoutError as e:
            print(f"  ✗ {label} 超时: {e}", flush=True)
            r = False
        finally:
            signal.alarm(0)
        (ok if r else fail).append(label)
    print(f"\n=== 汇总 ===")
    print(f"  通过 {len(ok)}: {ok}")
    print(f"  失败 {len(fail)}: {fail}")
