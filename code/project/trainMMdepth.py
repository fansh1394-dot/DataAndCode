#这是提供 参考示例的 YOLOMM稠密深度训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('yolo11n-mm-mid-depth.yaml')
    # model = YOLOMM('yolo11n-mm-mid-depth.yaml', scale='s')  # 可选 scales: n/s/m/l/x
    model.train(
                data='/datasets/depthmm/data.yaml',  # GT=gt_depth/；X=images_ir/（M3FD 小集）
                epochs=5, batch=8, imgsz=640,
                scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
                mosaic=0.0,  # depth 不做 Mosaic/MixUp
                # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
                # single_modal='x', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
                # dlog=1.0,   # SILog gain
                # dgrad=0.5,  # 梯度金字塔 gain
                # dlam=1.0,   # 1.0=纯尺度不变 SILog；0.0=log-RMSE
                # amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭
                device='0',
                cache=True,  #内存超过32G 的可以考虑使用此方法加速训练，硬盘空间足够大的也可以考虑 设置为 disk 获取加速，都不满足的就关闭
                exist_ok=False,
                workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
                # afss_enabled 不要打开：depth 任务不支持 AFSS
                project='ResTest', name='depth-test')
