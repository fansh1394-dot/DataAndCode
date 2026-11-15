#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

for v in ("reg", "unreg"):
    p = f"/root/rivermind-fs/birddrone/batch1/eval_swap/A0_valIR_{v}.json"
    d = json.load(open(p))
    m = d.get("metrics") or d
    print(f"{v:6s} 整体={m['metrics/mAP50-95(B)']:.4f} "
          f"RGB={m['metrics/rgb/metrics/mAP50-95(B)']:.4f} "
          f"IR={m['metrics/ir/metrics/mAP50-95(B)']:.4f} "
          f"fuse={m['metrics/fuse/metrics/mAP50-95(B)']:.4f}")
