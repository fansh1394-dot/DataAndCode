#这是提供 参考示例的 RTDETRMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import RTDETRMM

if __name__ == '__main__':
    model = RTDETRMM('rtdetr-r18-mm-mid.yaml')
    model.train(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
                epochs=100, batch=16, imgsz=640, close_mosaic=0,
                scale='s',  # 选择模型 YAML `scales`（n/s/m/l/x）；
                # modality='RGB', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
                # single_modal='x', #单模态参数，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
                # loss_cls='varifocal',  # 分类损失类型: focal | varifocal | bce (默认: varifocal)
                # loss_bbox='l1',        # L1 定位损失: l1 | none (默认: l1)
                # loss_giou='giou',      # GIoU 损失: giou | none (默认: giou)
                # loss_aux=True,         # 辅助损失开关: True | False (默认: True)
                # amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭
                optimizer='AdamW' ,# 推荐使用 AdamW 用于收敛稳定
                lr0= 1e-4, #初始学习率
                lrf= 1e-3, #最终学习率
                device='0',
                cache=True,  #内存超过32G 的可以考虑使用此方法加速训练，硬盘空间足够大的也可以考虑 设置为 disk 获取加速，都不满足的就关闭
                exist_ok=True,
                workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
                # afss_enabled=True, #启动afss训练采样机制
                project='ResTest', name='RTDETRMM-Test')