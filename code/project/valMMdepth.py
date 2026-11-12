#这是提供 参考示例的 YOLOMM稠密深度验证器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

model = YOLOMM('ResTest/depth-test/weights/best.pt')
model.val(data='/datasets/depthmm/data.yaml',  # GT=gt_depth/；X=images_ir/
          split='val', device='0',
        #   modality='x',  # 模态消融参数 非必要不得开启
          project='ResTest',
          name='depth-val')
