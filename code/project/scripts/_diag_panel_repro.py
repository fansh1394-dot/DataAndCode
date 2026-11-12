"""复现训练时 plot_training_samples 的可视化链路，输出 IR 面板单帧放大图用于定位偏移。

用法：python scripts/_diag_panel_repro.py
产物：YOLO26/logs/_diag_panel/<stem>.jpg
"""
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.data.augment import LetterBox, Compose, Format  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402

DATA_YAML = "/root/rivermind-data/uav_data_v2/data.yaml"
OUT = ROOT / "YOLO26/logs/_diag_panel"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(DATA_YAML)
    hyp = IterableSimpleNamespace(**{
        "imgsz": 640, "rect": False, "cache": False, "single_cls": False,
        "classes": None, "fraction": 1.0, "task": "detect", "timedebug": False,
        "hsv_h": 0, "hsv_s": 0, "hsv_v": 0, "degrees": 0.0, "translate": 0.0,
        "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0,
        "use_multimodal_aug": True, "mm_target_repaste_p": 0.0, "mm_bg_swap_p": 0.0,
        "mm_async_geo_p": 0.0, "mm_modal_dropout_p": 0.0, "mm_rgb_albu_enable": False,
        "mm_x_non_geo_enable": False, "mask_ratio": 4, "overlap_mask": True,
        "auto_augment": "none",
    })
    ds = build_yolo_dataset(
        hyp, data["train"], 1, data, mode="train", rect=False, stride=32,
        multi_modal_image=True, x_modality="ir", x_modality_dir="images_ir",
    )
    # 与训练一致的增强链（关闭随机项，只保留 letterbox）
    ds.transforms = Compose([
        LetterBox(new_shape=(640, 640), scaleup=False),
        Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
    ])
    print("nc images:", ds.ni, "dual_labels_configured:", getattr(ds, "dual_labels_configured", None))
    print("label_files_ir[0]:", ds.label_files_ir[0] if getattr(ds, "label_files_ir", None) else None)
    print("x_paths[0]:", ds._x_paths[0])

    picks = [0, 1, 2, 5, 10, 20]
    for i in picks:
        if i >= ds.ni:
            continue
        s = ds[i]
        stem = Path(s["im_file"]).stem
        img = s["img"]  # [6,640,640] float
        ir = img[3:6].float().mean(0).numpy()
        rgbch = img[0:3].float().mean(0).numpy()
        print("  stats ir min/max/mean", ir.min(), ir.max(), round(float(ir.mean()), 4),
              "| rgb min/max/mean", rgbch.min(), rgbch.max(), round(float(rgbch.mean()), 4))
        ir_u8 = np.clip(ir * 255, 0, 255).astype(np.uint8)
        # 左右并排：左=IR 通道，右=RGB 通道，便于确认第 4-6 通道确实是红外
        vis_l = cv2.cvtColor(ir_u8, cv2.COLOR_GRAY2BGR)
        rgb_u8 = np.clip(rgbch * 255, 0, 255).astype(np.uint8)
        vis_r = cv2.cvtColor(rgb_u8, cv2.COLOR_GRAY2BGR)
        vis = np.concatenate([vis_l, vis_r], axis=1)
        H, W = vis.shape[:2]
        W1 = W // 2  # 单侧宽度（640）

        def draw(bb, cls, color, tag, col):
            for j in range(len(bb)):
                x, y, w, h = bb[j].tolist()
                x1, y1 = int((x - w / 2) * W1) + col, int((y - h / 2) * H)
                x2, y2 = int((x + w / 2) * W1) + col, int((y + h / 2) * H)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                cv2.putText(vis, f"{tag}{int(cls[j])}", (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        draw(s["bboxes"].numpy().reshape(-1, 4), s["cls"].numpy().reshape(-1), (255, 128, 0), "rgb", 0)
        draw(s["bboxes"].numpy().reshape(-1, 4), s["cls"].numpy().reshape(-1), (255, 128, 0), "rgb", W1)
        if "bboxes_ir" in s and s["bboxes_ir"].numel():
            draw(s["bboxes_ir"].numpy().reshape(-1, 4), s["cls_ir"].numpy().reshape(-1), (255, 0, 0), "ir", 0)
        cv2.imwrite(str(OUT / f"{stem}.jpg"), vis)
        print("saved", stem, "n_rgb", len(s["bboxes"]), "n_ir", len(s.get("bboxes_ir", [])))


if __name__ == "__main__":
    main()
