# -*- coding: utf-8 -*-
"""CPU 干跑：验证教师按 800 前向 + 特征 hook 是否可用（不依赖 GPU 空闲）。"""
import torch

from ultralytics.nn.mm.distill.runtime import _TeacherRuntime
from ultralytics.nn.mm.distill.schema import TeacherSpec
from ultralytics.nn.tasks import attempt_load_one_weight

W = (
    "/3250410011/root/rivermind-data/MutilModel_3485988898/"
    "YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt"
)


def main():
    spec = TeacherSpec(name="t", role="dual", weights=W, imgsz=800)
    model, ckpt = attempt_load_one_weight(W)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("teacher loaded, last layer:", type(model.model[-1]).__name__)

    rt = _TeacherRuntime(spec, model, family="yolomm")
    rt.register_feature_hooks([48, 51, 54])

    batch = {"img": torch.rand(2, 6, 640, 640)}
    preds = rt.forward(batch)
    feats = rt.get_features()
    print("preds type:", type(preds).__name__)
    print("features:", {k: tuple(v.shape) for k, v in feats.items()})

    assert tuple(feats[48].shape[-2:]) == (100, 100), "fuse P3 教师特征应为 800/8=100"
    assert tuple(feats[51].shape[-2:]) == (50, 50), "fuse P4 教师特征应为 800/16=50"
    assert tuple(feats[54].shape[-2:]) == (25, 25), "fuse P5 教师特征应为 800/32=25"
    print("OK")


if __name__ == "__main__":
    main()
