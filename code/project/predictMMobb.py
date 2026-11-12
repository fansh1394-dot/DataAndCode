import sys
from pathlib import Path
from ultralytics.tools import MultiModalSampler
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLOMM

# 使用可用的YOLOMM OBB权重
model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/OBB-Test/weights/best.pt')
dataset_yaml = "/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ultralytics/cfg/datasets/mmdata/obbmm.yaml"

sampler = MultiModalSampler(dataset_yaml, split="val")
# rgb_img, x_img = sampler.sample_source_list(n=10)
rgb_img, x_img = sampler.sample_one()

# 推理阈值可根据需要调整
common_kwargs = dict(project='ResTest', save=True, save_json=True, exist_ok=True, conf=0.3, iou=0.35)

# 测试双模态OBB预测
print("=== 测试双模态 OBB 预测 ===")
model.predict(rgb_source=rgb_img,
              x_source=x_img,
              name='OBB_MM_dual_modal',
              **common_kwargs,
              )

# 测试单模态RGB OBB预测
print("\n=== 测试单模态 RGB OBB 预测 ===")
model.predict(rgb_source=rgb_img,
              x_source=None,
              name='OBB_MM_single_rgb',
              modality='rgb',
              **common_kwargs,
              )

# 测试单模态X模态 OBB预测
print("\n=== 测试单模态 X OBB 预测 ===")
model.predict(x_source=x_img,
              rgb_source=None,
              name='OBB_MM_single_x',
              modality='x',
              **common_kwargs,
              )
