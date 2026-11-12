# 这是多模态实例分割的 COCO 指标计算示例（纯本地实现，不依赖 pycocotools / faster-coco-eval）。
# 你可以参考该脚本按需调整参数（data/split/device/imgsz/conf 等）。

from ultralytics import YOLOMM

# 示例：加载分割权重
model = YOLOMM("/path/to/your/mm-seg/weights/best.pt")

# 直接调用 cocoval（会自动按 task=segment 分发到分割版 COCO 验证器）
results = model.cocoval(
    data="/path/to/your/data_seg.yaml",
    split="val",
    device="0",
    imgsz=640,
    conf=0.05,
    save_json=True,
    plots=False,
    # modality="x",  # 可选：单模态消融，如 'rgb' / 'x' / 具体模态名
    project="ResTest",
    name="ValCOCOSeg",
)

# 关键结果（Mask 为主指标）
print("COCO Mask AP:", results.get("metrics/coco_mask/AP"))
print("COCO Box  AP:", results.get("metrics/coco_box/AP"))

