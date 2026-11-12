#这是提供 参考示例的 YOLOMM稠密深度推理器使用方法
#按必须按照你的需求来更改而不是盲目去使用


from ultralytics import YOLOMM
from ultralytics.tools import MultiModalSampler

#模型权重
model = YOLOMM('ResTest/depth-test/weights/best.pt')

sampler = MultiModalSampler('/datasets/depthmm/data.yaml', split='val')
rgb_list, x_list = sampler.sample_source_list(n=8)
for r, x in zip(rgb_list, x_list):
    print(f"  RGB: {r}")
    print(f"  X: {x}")

# 测试双模态预测（保存 uint16 米制 png + turbo 可视化）
print("=== 测试双模态预测 ===")
model.predict(rgb_source=rgb_list,
              x_source=x_list,
              project='ResTest',
              name='predict-depth-dual',
              save=True,
              exist_ok=True,
              debug=True,
              )

# 测试单模态RGB预测（使用零填充X模态）
print("\n=== 测试单模态RGB预测 ===")
model.predict(rgb_source=rgb_list,
              x_source=None,
              project='ResTest',
              name='predict-depth-rgb',
              save=True,
              exist_ok=True,
              debug=True,
              )

# 测试单模态X预测（使用零填充RGB模态）
print("\n=== 测试单模态X预测 ===")
model.predict(rgb_source=None,
              x_source=x_list,
              project='ResTest',
              name='predict-depth-x',
              save=True,
              exist_ok=True,
              debug=True,
              )
