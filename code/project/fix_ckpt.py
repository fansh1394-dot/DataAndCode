# -*- coding: utf-8 -*-
# 一次性修复脚本：把服务器(Linux)上训练的 best.pt 转成本地(Windows)可加载的 best_local.pt
# 原因：checkpoint 里序列化了 pathlib.PosixPath，Windows 上无此类，直接 torch.load 会崩
import pathlib

import torch

SRC = r"C:\Users\不如喝茶去\Desktop\双光谱毕设\ResTest\yolomm-c3k2-pkimodule\weights\best.pt"
DST = r"C:\Users\不如喝茶去\Desktop\双光谱毕设\ResTest\yolomm-c3k2-pkimodule\weights\best_local.pt"


def convert(obj):
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(convert(v) for v in obj)
    return obj


def main():
    orig_posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = orig_posix

    ckpt = convert(ckpt)

    torch.save(ckpt, DST)
    print("完成，已保存: " + DST)
    print("epoch =", ckpt.get("epoch"))


if __name__ == "__main__":
    main()
