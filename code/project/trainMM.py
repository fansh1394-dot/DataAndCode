#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('ultralytics/cfg/models/mm/detect/yolo11/extraction/yolo11n-mm-c3k2_pkimodule.yaml')
    # model = YOLOMM('yolo11n-mm-mid.yaml')
    model.train(
                # data='/root/datasets/M3FD_split/data.yaml',
                # 数据集配置：本项目 data.yaml 里的 path 也要同步指向服务器上的数据根目录
                data='/root/autodl-tmp/uav_data/data.yaml',
                # data='D:/uav_data/data.yaml',  # ← 本地 Windows 调试时换回这行
                epochs=2, batch=16, imgsz=640, close_mosaic=0,
                scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
                # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
                # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
                # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
                # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
                # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
                # amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭
                device='0',
                cache=False,  #内存超过32G 的可以考虑使用此方法加速训练，硬盘空间足够大的也可以考虑 设置为 disk 获取加速，都不满足的就关闭
                              #注意：本项目 5.5 万张双模态图（RGB+IR）开 True 约需 110G 内存，服务器一般不够，故设为 False
                exist_ok=True,
                workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试（服务器 CPU 核少时可降到 4）
                # afss_enabled=True, #启动afss训练采样机制
                project='ResTest', name='yolomm-c3k2-pkimodule')
