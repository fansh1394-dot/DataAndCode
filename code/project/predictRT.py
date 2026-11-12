import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ultralytics.tools import MultiModalSampler
from ultralytics import RTDETRMM

# 使用可用的RT-DETR权重
model = RTDETRMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/RTDETRMM-LST/weights/best.pt')
dataset_yaml = "/home/zhizi/work/multimodel/ultralyticmm/data.yaml"
sampler = MultiModalSampler(dataset_yaml, split="val", seed=42)
rgb_img, x_img = sampler.sample_source_list(n=10)
# rgb_img, x_img = sampler.get_source_dirs()
# 测试双模态预测（新API - 显式rgb_source/x_source）
print("=== 测试双模态预测 ===")
model.predict(rgb_source=rgb_img,
              x_source=x_img,
              project='ResTest',
              name='rtdetrmm_dual_modal',
              save=True,
              exist_ok=True,
            #   debug=True
              )

# 测试单模态RGB预测（使用零填充X模态）
print("\n=== 测试单模态RGB预测 ===")
model.predict(rgb_source='/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',
              x_source=None,
              project='ResTest',
              name='rtdetrmm_single_rgb',
              save=True,
              exist_ok=True
              )

# 测试单模态X模态预测（使用零填充RGB模态）
print("\n=== 测试单模态X模态预测 ===")
model.predict(rgb_source=None,
              x_source='/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png',
              project='ResTest',
              name='rtdetrmm_single_x',
              save=True,
              exist_ok=True
              )