#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')  # GPU控制: 0/1/2/3 或 "0,1" 多卡
from ultralytics import YOLOMM

if __name__ == '__main__':
    
    model = YOLOMM('yolo11n-mm-mid-seg.yaml')
    model.train(data='/root/datasets/tree_seg/tree_seg/data.yaml',
                epochs=50,batch=8*2,
                # modality='X', #模态消融参数，非必要不用开启
                # loss_cls='bce',      # 分类损失类型: bce | focal (默认: bce)
                # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou (默认: ciou)
                # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
                cache=True,
                # amp = False,
                # exist_ok=True,
                project='ResTest',name='YOLOMM-seg-test')
