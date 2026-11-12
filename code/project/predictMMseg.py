import sys
from pathlib import Path
from ultralytics.tools import MultiModalSampler
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLOMM

# 使用可用的YOLOMM权重
model = YOLOMM('//home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM-seg-LST/weights/best.pt')
dataset_yaml = "/home/zhizi/work/multimodel/ultralyticmm/data_SEG.yaml"
# rgb_img = '/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png'
# x_img = '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png'
sampler = MultiModalSampler(dataset_yaml, split="val", seed=42)
# rgb_img, x_img = sampler.sample_one()
rgb_img, x_img = sampler.sample_source_list(n=10)
# 测试双模态预测
print("=== 测试双模态预测 ===")
model.predict(rgb_source=rgb_img,
               x_source=x_img,
              project='ResTest',
              name='yolomm_seg_dual_modal',
              save=True,
              exist_ok=True,
              debug=True,
              )

# 测试单模态RGB预测
print("\n=== 测试单模态RGB预测 ===")
model.predict(rgb_source=rgb_img,
              x_source=None,
              project='ResTest',
              name='yolomm_seg_single_rgb',
              save=True,
              modality='rgb',
              exist_ok=True,
              debug=True,
              )

# 测试单模态X模态预测  
print("\n=== 测试单模态X模态预测 ===")
model.predict(x_source=x_img,
              rgb_source=None,
              project='ResTest',
              name='yolomm_seg_single_x',
              save=True,
              modality='x',
              exist_ok=True,
              debug=True,
              )