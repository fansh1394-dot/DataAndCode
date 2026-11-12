# -*- coding: utf-8 -*-
# 单元自检：registration.py 的映射正确性
import sys
sys.path.insert(0, r'c:\Users\不如喝茶去\Desktop\双光谱毕设\MutilModel_3485988898')
import numpy as np
from ultralytics.engine.multimodal.registration import DualRegistration, get_registration

ok = True

# 1) 关闭时恒等
reg_off = DualRegistration(mode='shift', shift=(0.0136, 0.0289), enabled=False)
b = np.array([[960., 540., 1200., 700., 0.9, 0.]])
r = reg_off.apply_boxes(b, (1080, 1920), (512, 640))
print('1) 关闭 -> 恒等:', np.allclose(r, b)); ok &= np.allclose(r, b)

# 2) 画幅一致 + 关闭 -> 恒等（回归保护）
assert np.allclose(r, b)

# 3) shift 模式：把手算值比对
reg_on = DualRegistration(mode='shift', shift=(0.0136, 0.0289), enabled=True)
r2 = reg_on.apply_boxes(b, (1080, 1920), (512, 640))
# 期望： 归一化 x = 960/1920 = 0.5 -> 0.5136 -> *640 = 328.70
#        y = 540/1080 = 0.5 -> 0.5289 -> *512 = 270.80
exp = np.array([(0.5 + 0.0136) * 640, (0.5 + 0.0289) * 512])
print(f'3) shift x1,y1 = ({r2[0,0]:.2f},{r2[0,1]:.2f})  期望 ({exp[0]:.2f},{exp[1]:.2f})')
ok &= np.allclose(r2[0, :2], exp, atol=1e-6)

# 4) 画幅一致（src==dst==IR画幅）时 shift 模式应等价于 IR 图幅下的像素平移
b4 = np.array([[320., 256., 400., 332., 0.9, 0.]])  # 已在 IR 画幅归一化 0.5/0.5
r3 = reg_on.apply_boxes(b4, (512, 640), (512, 640))
d = r3[0, :2] - b4[0, :2]
print(f'4) 同画幅像素位移 = ({d[0]:+.2f},{d[1]:+.2f})  期望 (+8.70,+14.80)')
ok &= np.allclose(d, [0.0136 * 640, 0.0289 * 512], atol=1e-6)

# 5) 空数组 / 无框
r4 = reg_on.apply_boxes(np.zeros((0, 6)), (1080, 1920), (512, 640))
print('5) 空框安全:', r4.shape == (0, 6)); ok &= r4.shape == (0, 6)

# 6) 从标定文件加载
reg_f = DualRegistration.from_calib(r'D:\uav_data\_vis\calib.npz', mode='shift', enabled=True)
print(f'6) 加载标定: {reg_f.describe()}')

# 7) 从标定文件加载仿射
reg_a = DualRegistration.from_calib(r'D:\uav_data\_vis\calib.npz', mode='affine', enabled=True)
print(f'7) 仿射: {reg_a.describe()}')

# 8) 文件不存在时安全降级
reg_m = DualRegistration.from_calib(r'D:\no_such\file.npz', enabled=True)
print(f'8) 缺失文件降级: mode={reg_m.mode} enabled={reg_m.enabled}')
ok &= (reg_m.mode == 'none' and not reg_m.enabled)

# 9) 仿射恒等性：给单位仿射应还原为原坐标
reg_id = DualRegistration(mode='affine',
                          affine_x=np.array([1.0, 0.0, 0.0]),
                          affine_y=np.array([0.0, 1.0, 0.0]), enabled=True)
r9 = reg_id.apply_boxes(b, (1080, 1920), (512, 640))
exp9 = np.array([0.5 * 640, 0.5 * 512, 1200 / 1920 * 640, 700 / 1080 * 512])
print(f'9) 单位仿射 -> 纯画幅换算: {np.round(r9[0,:4],2)}  期望 {np.round(exp9,2)}')
ok &= np.allclose(r9[0, :4], exp9, atol=1e-4)

# 10) 裁剪边界：框跑出画幅应被裁
b10 = np.array([[1850., 1050., 1900., 1075., 0.9, 0.]])
r10 = reg_on.apply_boxes(b10, (1080, 1920), (512, 640))
inb = (r10[:, [0, 2]] <= 640).all() and (r10[:, [1, 3]] <= 512).all()
print(f'10) 越界裁剪: {np.round(r10[0,:4],2)} 在界内={inb}'); ok &= inb

# 11) 真实仿射：用标定系数验证框中心映射符合预期
reg_real = DualRegistration.from_calib(r'D:\uav_data\_vis\calib.npz', mode='affine', enabled=True)
bc = np.array([[480., 270., 480., 270., 0.9, 0.]])       # RGB 画幅归一化中心 (0.25,0.25)
rc = reg_real.apply_boxes(bc, (1080, 1920), (512, 640))
nx, ny = rc[0, 0] / 640, rc[0, 1] / 512
ax, ay = reg_real.affine_x, reg_real.affine_y
enx = ax[0] * 0.25 + ax[1] * 0.25 + ax[2]
eny = ay[0] * 0.25 + ay[1] * 0.25 + ay[2]
print(f'11) 仿射中心映射 ({nx:.5f},{ny:.5f})  期望 ({enx:.5f},{eny:.5f})')
ok &= np.allclose([nx, ny], [enx, eny], atol=1e-6)

# 12) 默认模式应为 affine（若标定文件含 affine 系数）
reg_def = DualRegistration.from_calib(r'D:\uav_data\_vis\calib.npz', enabled=True)
print(f'12) from_calib 默认模式 = {reg_def.mode}'); ok &= (reg_def.mode == 'affine')

print('\n' + ('全部通过' if ok else '存在失败项'))
