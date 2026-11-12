# -*- coding: utf-8 -*-
"""生成几何增强实验的 YAML 与训练参数（在服务器上执行）。

设计
----
基线 mdp150_M0_concat_s0 的融合颈是「Concat + 1x1 Conv」，其余增强参数为
  mosaic=1.0, fliplr=0.5, scale=0.5, translate=0.1, hsv h/s/v=0.015/0.7/0.4
  degrees=0, shear=0, perspective=0
本实验**只打开几何项**，YAML 不动（同一份 concat 配置），
因此差异可完全归因到几何增强。

已验证（scripts/_verify_sync_geo.py）：
  原生 RandomPerspective 用同一仿射矩阵 M 同时变换 6 通道图与 instances_ir，
  开启后 RGB/IR 框逐框坐标差 = 0.00 像素，对齐未被破坏。
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
BASE = ROOT / "ultralytics/cfg/models/mm/detect/yolo26/base"
OUT = ROOT / "_d2p1/geo_exp"

# 两档几何强度。温和档先跑，激进档作为对照。
CONFIGS = {
    "geo_mild": {"degrees": 10.0, "shear": 5.0, "perspective": 0.0005},
    "geo_strong": {"degrees": 20.0, "shear": 10.0, "perspective": 0.001},
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_yaml = BASE / "yolo26n-mm-dual-fusion-concat.yaml"
    if not base_yaml.exists():
        print(f"缺基础配置: {base_yaml}")
        return 1
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))

    manifest = {}
    for tag, geo in CONFIGS.items():
        p = OUT / f"yolo26n-mm-dual-fusion-{tag}.yaml"
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        manifest[tag] = {"yaml": str(p), "geo": geo}
        print(f"[YAML] {p.name}  geo={geo}")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n清单: {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
