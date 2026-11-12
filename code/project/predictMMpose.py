# YOLOMM Pose 预测脚本参考示例
# 请根据实际需求修改配置参数

from ultralytics import YOLOMM

# 加载训练好的模型权重
model = YOLOMM('ResTest/yolomm-pose-tiger/weights/best.pt')

# 图像路径
rgb = '/path/to/rgb_image.jpg'
edge = '/path/to/edge_image.jpg'

# 双模态预测
print("=== 双模态预测 ===")
model.predict(
    rgb_source=rgb,
    x_source=edge,
    project='ResTest',
    name='yolomm-pose-dual',
    save=True,
    exist_ok=True,
)

# 单模态RGB预测（X模态零填充）
print("\n=== 单模态RGB预测 ===")
model.predict(
    rgb_source=rgb,
    x_source=None,
    project='ResTest',
    name='yolomm-pose-rgb-only',
    save=True,
    exist_ok=True,
)

# 单模态Edge预测（RGB模态零填充）
print("\n=== 单模态Edge预测 ===")
model.predict(
    rgb_source=None,
    x_source=edge,
    project='ResTest',
    name='yolomm-pose-edge-only',
    save=True,
    exist_ok=True,
)
