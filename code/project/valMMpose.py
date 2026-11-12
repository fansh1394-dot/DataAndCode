# YOLOMM Pose 验证脚本参考示例
# 请根据实际需求修改配置参数

from ultralytics import YOLOMM

# 加载训练好的模型权重
model = YOLOMM('ResTest/yolomm-pose-tiger/weights/best.pt')

# 验证配置
model.val(
    data='/home/zhizi/work/multimodel/ultralyticmm/datasets/tiger/data.yaml',
    split='val',
    device='0',
    project='ResTest',
    name='yolomm-pose-tiger-val',
    # modality='edge',  # 模态消融参数，非必要不开启
)
