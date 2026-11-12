# -*- coding: utf-8 -*-
"""E33：ONNX -> RKNN int8（RK3588），并用模拟器核对 cos 相似度。"""
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
    ap.add_argument("--onnx", default="/root/rivermind-fs/birddrone/deploy_nc3/model_raw.onnx")
    ap.add_argument("--calib", default="/root/rivermind-fs/birddrone/deploy_nc3/calib")
    ap.add_argument("--out", default="/root/rivermind-fs/birddrone/deploy_nc3/model_nc3_rk3588.rknn")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ncalib", type=int, default=300)
    a = ap.parse_args()

    import onnx
    from rknn.api import RKNN

    m = onnx.load(a.onnx)
    in_name = m.graph.input[0].name
    print("onnx input", in_name, "outputs", [o.name for o in m.graph.output])

    calib = sorted(Path(a.calib).glob("calib_*.npy"))[: a.ncalib]
    print("calib", len(calib))

    def gen():
        for p in calib:
            yield [np.load(p)]

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0] * 6], std_values=[[255] * 6],
                target_platform="rk3588",
                quantized_dtype="asymmetric_quantized-8",
                optimization_level=3)
    rknn.load_onnx(model=a.onnx, inputs=[in_name],
                   input_size_list=[[1, 6, a.imgsz, a.imgsz]])
    rknn.build(do_quantization=True, dataset=gen)
    rknn.export_rknn(a.out)
    print("exported", a.out)

    rknn.init_runtime()
    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    cos_all = []
    for p in calib[:5]:
        img = np.load(p)
        ref = sess.run(None, {in_name: (img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)})
        got = rknn.inference(inputs=[img])
        cs = [cosine(r, g) for r, g in zip(ref, got)]
        cos_all.append(cs)
        print("cos", [round(c, 5) for c in cs])
    print("min_cos", round(float(np.min(cos_all)), 5))
    rknn.release()


if __name__ == "__main__":
    main()
