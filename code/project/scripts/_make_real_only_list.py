# -*- coding: utf-8 -*-
"""生成 test 集的「纯真实帧」图像列表，供 eval_split.py --val-list 使用。

背景：test 614 帧中有 118 帧是合成数据（is_synthetic=True，19.2%）。
混入合成帧会把指标抬高，且审稿人可能质疑。本脚本按 metadata 里的
is_synthetic 字段筛出真实帧，写成图像列表。

注意：--val-list 的条目是「相对 data.yaml 的 path」的路径，由
base.py 的 _resolve_list_entry 解析（绝对路径优先，其次列表目录，再次数据集根）。
"""
import json
from pathlib import Path

ROOT = Path("/3250410011/root/rivermind-data/mendeley_split")
OUT = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/_d2p1/eval_yamls")


def main():
    meta = json.loads((ROOT / "metadata" / "test.json").read_text(encoding="utf-8"))
    frames = meta["frames"]

    real, syn = [], []
    for name, info in frames.items():
        (syn if info.get("is_synthetic") else real).append(name)

    OUT.mkdir(parents=True, exist_ok=True)

    # 图像列表：相对 dataset path（data.yaml 里 path 指向 mendeley_split）
    # 帧名可能是 "xxx.png" 或 "xxx"，统一补 images/test/ 前缀并保证扩展名
    def to_entry(n):
        base = n if "." in Path(n).name else f"{n}.png"
        return f"images/test/{base}"

    real_list = OUT / "test_real_only.txt"
    real_list.write_text("\n".join(to_entry(n) for n in sorted(real)) + "\n", encoding="utf-8")

    print(f"test 总帧数      = {len(frames)}")
    print(f"  真实帧         = {len(real)}  ({len(real) / len(frames) * 100:.1f}%)")
    print(f"  合成帧         = {len(syn)}  ({len(syn) / len(frames) * 100:.1f}%)")
    print(f"\n已写出: {real_list}  共 {len(real)} 行")

    # 抽 3 行核对路径是否真实存在
    print("\n路径抽查：")
    for n in sorted(real)[:3]:
        p = ROOT / to_entry(n)
        print(f"  {'OK ' if p.exists() else 'MISS'} {to_entry(n)}")


if __name__ == "__main__":
    main()
