# -*- coding: utf-8 -*-
"""生成最终对比结果：predict_img_dual_fixed / predict_video_dual_fixed

在 ResTest 下生成两套修复后结果目录，供论文做前后对比。
- predict_img_dual_fixed   验证集图片（10 对）
- predict_video_dual_fixed 视频抽帧（100 对）

配准补偿通过环境变量开启：
    ULTRALYTICS_MM_REGISTRATION=1
    ULTRALYTICS_MM_REGISTRATION_MODE=shift|affine

用法：
    python scripts/calib/run_fixed_predict.py            # 不启用配准补偿（仅修根因）
    python scripts/calib/run_fixed_predict.py --reg      # 启用配准补偿
"""
import argparse
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

WEIGHTS = r"C:\Users\不如喝茶去\Desktop\双光谱毕设\ResTest\yolomm-c3k2-pkimodule\weights\best_local.pt"
DATA_YAML = r"D:\uav_data\data.yaml"
FRAME_RGB = r"D:\uav_data\demo_frames\rgb"
FRAME_IR = r"D:\uav_data\demo_frames\ir"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", action="store_true", help="启用跨模态配准补偿")
    ap.add_argument("--mode", default="affine", choices=["shift", "affine"], help="补偿模式")
    ap.add_argument("--tag", default="", help="输出目录后缀（如 _reg）")
    args = ap.parse_args()

    if args.reg:
        os.environ["ULTRALYTICS_MM_REGISTRATION"] = "1"
        os.environ["ULTRALYTICS_MM_REGISTRATION_MODE"] = args.mode
        print(f"[配准] 已启用，模式={args.mode}")
    else:
        os.environ["ULTRALYTICS_MM_REGISTRATION"] = "0"
        print("[配准] 未启用（仅根因修复）")

    from ultralytics import YOLOMM
    from ultralytics.tools import MultiModalSampler

    model = YOLOMM(WEIGHTS)

    # ---- 方案 A：验证集图片 ----
    sampler = MultiModalSampler(DATA_YAML, split="val")
    rgb_list, x_list = sampler.sample_source_list(n=10)
    print("=== 双模态预测（RGB + IR 图片）===")
    model.predict(
        rgb_source=rgb_list,
        x_source=x_list,
        project=str(PROJ / "ResTest"),
        name=f"predict_img_dual_fixed{args.tag}",
        save=True,
        exist_ok=True,
    )

    # ---- 方案 B：视频抽帧 ----
    names = sorted(os.listdir(FRAME_RGB))
    rgb_v = [os.path.join(FRAME_RGB, n) for n in names if n.lower().endswith(".jpg")]
    ir_v = [os.path.join(FRAME_IR, n) for n in names if n.lower().endswith(".jpg")]
    print(f"=== 双模态预测（视频抽帧 {len(rgb_v)} 对）===")
    model.predict(
        rgb_source=rgb_v,
        x_source=ir_v,
        project=str(PROJ / "ResTest"),
        name=f"predict_video_dual_fixed{args.tag}",
        save=True,
        exist_ok=True,
    )
    print("完成")


if __name__ == "__main__":
    main()
