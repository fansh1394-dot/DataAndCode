# 这是提供参考示例的 YOLOMM Output 级蒸馏训练脚本
# 必须按照你的需求来更改，而不是盲目直接使用
#
# Output 蒸馏: 对齐教师-学生的检测头输出 logits
# 需配合蒸馏配置 yaml 使用，配置中 mappings.output 定义教师参与声明

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('yolo11n-mm-mid.yaml')

    model.train(
        data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
        epochs=10,
        batch=16,
        scale='n',
        device='0',
        cache=True,
        exist_ok=True,
        # 蒸馏配置: [配置文件路径, 蒸馏模式]
        # 模式: 'output' (检测头输出对齐)
        distill=['/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/distill.yaml', 'output'],
        project='ResTest',
        name='DistillExp/output-dual-teacher-to-yolomm-n',
    )
