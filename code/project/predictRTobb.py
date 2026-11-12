"""RTDETRMM OBB 推理脚本 — 双模态旋转框检测"""

from ultralytics import RTDETRMM

DATA_DIR = "/mnt/SSD/work/multimodel/ultralyticmm/datasets/obb"  #数据集地址
RGB_IMG = f"{DATA_DIR}/images/test/1001.png" #主模态图片
SEN_IMG = f"{DATA_DIR}/images_sar/test/1001.png" # 副模态图片
WEIGHTS = "ResTest/RTDETRMM-obb-Test/weights/best.pt" #权重地址

model = RTDETRMM(WEIGHTS)

results = model.predict(
    rgb_source=RGB_IMG,
    x_source=SEN_IMG,
    save=True,
    project="ResTest",
    name="RTDETRMM-obb-predict",
    exist_ok=True,
    debug=True,
)

for r in results:
    print(f"检测结果类型: {type(r).__name__}")
    boxes = getattr(r, "boxes", None)
    obb = getattr(r, "obb", None)
    if boxes is not None:
        print(f"boxes.data: {boxes.data.shape if boxes.data is not None else 'None'}")
    if obb is not None:
        print(f"obb.data: {obb.data.shape if obb.data is not None else 'None'}")
    print(f"检测结果数量: {len(boxes) if boxes is not None else 0}")
