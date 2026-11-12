# -*- coding: utf-8 -*-
"""E33：双输入 ONNX -> RKNN int8，并核对模拟器 cos 相似度。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/rivermind-fs/birddrone/deploy_nc3/model_2in.onnx")
    ap.add_argument("--dataset", default="/root/rivermind-fs/birddrone/deploy_nc3/dataset.txt")
    ap.add_argument("--calib-img", default="/root/rivermind-fs/birddrone/deploy_nc3/calib_img")
    ap.add_argument("--out", default="/root/rivermind-fs/birddrone/deploy_nc3/model_nc3_rk3588.rknn")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    import onnx
    from rknn.api import RKNN

    g = onnx.load(a.onnx)
    print("inputs", [i.name for i in g.graph.input])
    assert len(g.graph.input) == 2

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0], [0, 0, 0]],
                std_values=[[255, 255, 255], [255, 255, 255]],
                target_platform="rk3588",
                quantized_dtype="asymmetric_quantized-8",
                optimization_level=3)
    rknn.load_onnx(model=a.onnx, inputs=["rgb", "ir"],
                   input_size_list=[[1, 3, a.imgsz, a.imgsz], [1, 3, a.imgsz, a.imgsz]])
    rknn.build(do_quantization=True, dataset=a.dataset)
    rknn.export_rknn(a.out)
    print("exported", a.out)

    rknn.init_runtime()
    import cv2
    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    pairs = [l.split() for l in Path(a.dataset).read_text().splitlines() if l.strip()][:5]
    cos_all = []
    for rgb_p, ir_p in pairs:
        rgb = cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB)
        ir = cv2.cvtColor(cv2.imread(ir_p), cv2.COLOR_BGR2RGB)
        ref = sess.run(None, {"rgb": (rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0),
                              "ir": (ir.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)})
        got = rknn.inference(inputs=[rgb, ir])
        cs = [cosine(r, x) for r, x in zip(ref, got)]
        cos_all.append(cs)
        print("cos", [round(c, 5) for c in cs])
    print("min_cos", round(float(np.min(cos_all)), 5))
    rknn.release()


if __name__ == "__main__":
    main()
