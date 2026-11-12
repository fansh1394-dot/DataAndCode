from ultralytics import YOLOMM

model = YOLOMM('/root/ultralyticsmm/ResTest/yolomm-m3fd-200ep/weights/best.pt')
model.val(data='/root/datasets/M3FD_split/data.yaml',
          split='test',device='0',
        #   modality='x',模态消融参数 非必要不得开启
          # coco=True,# 打开之后输出为 coco 指标，COCO 指标和 YOLO 指标大概率不一致
          project='ResTest',
          name='test/yolomm-lst-val')
