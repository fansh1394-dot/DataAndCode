#这是RTDETRMM热力图可视化的方法，请你参考我的使用
#使用方法查看 wiki目录中的vis_usage.md文件
from ultralytics import RTDETRMM

model = RTDETRMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/RTDETRMM-LST/weights/best.pt')
model.vis(
          rgb_source = '/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',  # rgb_source：RGB 输入
          x_source = '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png',  # x_source：X 模态输入
          method='heat',  # 热力图：heat|heatmap
          layers=[10,15,20,25],  # 使用yaml层（RTDETR层数可能与YOLO不同，请根据模型调整）
          overlay='dual',  # 叠加底图：'rgb'|'x'|'dual'；双模态建议 dual 一次看全
          # modality='X',  # 模态消融：'rgb' 或 'x'；双模态输入时也可强制消融
          alg='gradcam',
          colormap='turbo',   # 推荐 turbo（默认），更舒适
          layout='panel',     # 输出三联图：原图｜热图｜叠加（默认）
          panel_scale=1.0,    # 放大输出，避免看起来过小
        #   split=True,
          save=True,
          project='ResTest/vis',
          name='VisRT',

          # ========== 可视化分辨率控制（提升CAM细腻度）==========
          # imgsz=1280,           # 可视化前处理输入尺寸（整数），提高可显著提升CAM空间分辨率（代价：更慢/更占显存）
          # auto_imgsz=True,      # 根据原图大小自动上调可视化imgsz（默认True），False则使用模型默认imgsz
          # imgsz_cap=1280,       # auto_imgsz的上限（默认1280），防止显存溢出

          # ========== CAM质量优化（减少"块状"观感）==========
          # cam_smooth=True,      # 对CAM上采样后做高斯平滑（默认True），减少锯齿/块状效应
          # cam_smooth_sigma=1.2, # 平滑强度（默认1.2，非负数），越大越柔和/细腻
          # cam_resize_interp='cubic',  # CAM上采样插值方法（默认cubic），可选：nearest|linear|cubic|lanczos|area

          # ========== 论文级素材导出（每层独立文件夹）==========
          # export_components=True,     # 导出每层的原图/热图/叠加/panel素材（默认True），便于论文自由组合
          # export_panel=True,          # export_components=True时，是否额外导出各路三联图panel（默认True）
          # export_scale=1.0,           # 导出素材整体缩放倍数（默认1.0），生成高质量论文图时可设2.0
          # save_scale=1.0,             # export_scale的别名，二者等效
          )
