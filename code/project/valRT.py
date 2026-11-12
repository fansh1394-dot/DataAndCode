from ultralytics import RTDETRMM

model = RTDETRMM('/mnt/SSD/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/RTDETRMM-LST/weights/best.pt')
model.val(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
            split='test',device=0,
            # modality='rgb',  # 双模态验证
            coco=True,
            project='ResTest',name='RTDETRMM-LST-dual-val')