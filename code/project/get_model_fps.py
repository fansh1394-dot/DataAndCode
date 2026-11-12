# FPS 性能测试脚本 — YOLOMM / RTDETRMM
# 文件: get_model_fps.py
# 请根据实际需求修改下方配置参数后运行
#
# 测量内容:
#   1. 前处理+推理+后处理 全管线 FPS
#   2. 仅模型前向推理 FPS
# 两组数据使用同一批抽样图像

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch

# ==================== 用户配置区 ====================
#模型路径
MODEL_PATH  = '/mnt/SSD/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/rgbt-tiny-11n-mid/weights/best.pt'
#数据yaml 路径
DATA_YAML   = '/mnt/SSD/work/multimodel/ultralyticmm/datasets/rgbt_tiny/rgbt_tiny_yolo/dataset.yaml'
MODEL_TYPE  = 'yolomm'                    # 模型类型: 'yolomm' 或 'rtdetrmm'
N_SAMPLES   = 50                          # 抽样数量
SPLIT       = 'val'                       # 数据集分割: 'train' / 'val' / 'test'
DEVICE      = '0'                         # 推理设备: '0' / 'cpu'
WARMUP      = 3                           # 预热轮数 (排除冷启动)
HALF        = False                       # 是否使用 FP16 推理
SEED        = 42                          # 随机种子
IMGSZ       = 640                         # 推理图像尺寸
# ===================================================


# ─────────────────── ASCII 框线输出 ───────────────────

def _pad(text, width, align='left'):
    """中英混合字符串对齐填充"""
    text = str(text)
    display_w = sum(2 if ord(c) > 127 else 1 for c in text)
    pad = max(0, width - display_w)
    if align == 'right':
        return ' ' * pad + text
    elif align == 'center':
        return ' ' * (pad // 2) + text + ' ' * (pad - pad // 2)
    return text + ' ' * pad


def _display_width(text):
    """计算中英混合字符串显示宽度"""
    return sum(2 if ord(c) > 127 else 1 for c in str(text))


def _make_table(headers, rows, col_widths=None):
    """生成 Unicode 盒线表格字符串列表

    col_widths 基于字符数 (len)，不含 display width 偏差。
    调用者负责将结果嵌入外框时使用一致的宽度计算。
    """
    # 用 display_width 计算实际渲染宽度
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = _display_width(h)
            for row in rows:
                w = max(w, _display_width(str(row[i])))
            col_widths.append(w + 2)  # 两侧各1空格

    lines = []
    lines.append('┌' + '┬'.join('─' * w for w in col_widths) + '┐')
    header_cells = [_pad(h, w, 'center') for h, w in zip(headers, col_widths)]
    lines.append('│' + '│'.join(header_cells) + '│')
    lines.append('├' + '┼'.join('─' * w for w in col_widths) + '┤')
    for row in rows:
        cells = [_pad(str(v), w) for v, w in zip(row, col_widths)]
        lines.append('│' + '│'.join(cells) + '│')
    lines.append('└' + '┴'.join('─' * w for w in col_widths) + '┘')
    return lines


def print_ascii_report(cfg, pipeline_stats, infer_stats, latency_stats):
    """打印完整的 ASCII 框线报告"""
    W = 62  # 外框宽度

    def section_line(text):
        pad = W - 2 - _display_width(text)
        return '║ ' + text + ' ' * max(0, pad) + '║'

    def empty_line():
        return '║' + ' ' * W + '║'

    def embed_line(table_line):
        """将内嵌表格行嵌入外框"""
        pad = W - 2 - _display_width(table_line)
        return '║ ' + table_line + ' ' * max(0, pad) + '║'

    lines = []
    lines.append('╔' + '═' * W + '╗')
    title = 'FPS 性能测试报告'
    tw = _display_width(title)
    pad = W - tw
    lines.append('║' + ' ' * (pad // 2) + title + ' ' * (pad - pad // 2) + '║')
    lines.append('╠' + '═' * W + '╣')

    # 模型信息区
    lines.append(section_line('模型信息'))
    lines.append(section_line(f'  模型类型:  {cfg["model_type"].upper()}'))
    lines.append(section_line(f'  权重路径:  {cfg["model_path"]}'))
    lines.append(section_line(f'  数据集:    {cfg["dataset_name"]}'))
    lines.append(section_line(f'  抽样数量:  {cfg["n_samples"]}'))
    lines.append(section_line(f'  图像尺寸:  {cfg["imgsz"]}'))
    lines.append(section_line(f'  推理设备:  {cfg["device_name"]}'))
    lines.append(section_line(f'  显存:      {cfg["device_mem"]}'))
    lines.append(section_line(f'  精度:      {"FP16" if cfg["half"] else "FP32"}'))
    lines.append(section_line(f'  随机种子:  {cfg["seed"]}'))
    lines.append('╠' + '═' * W + '╣')

    # FPS 结果表格
    lines.append(section_line('FPS 测试结果'))
    lines.append(empty_line())

    fps_headers = ['测试指标', '平均 FPS', '标准差']
    fps_rows = [
        ['前处理+推理+后处理', f'{pipeline_stats["mean"]:.2f}', f'{pipeline_stats["std"]:.2f}'],
        ['仅模型推理',        f'{infer_stats["mean"]:.2f}',    f'{infer_stats["std"]:.2f}'],
    ]
    for tl in _make_table(fps_headers, fps_rows):
        lines.append(embed_line(tl))

    lines.append('╠' + '═' * W + '╣')

    # 延迟详情表格
    lines.append(section_line('延迟详情 (ms)'))
    lines.append(empty_line())

    lat_headers = ['测试项', '前处理', '推理', '后处理']
    lat_rows = [
        ['平均延迟',  f'{latency_stats["preprocess"]:.2f}',
                     f'{latency_stats["inference"]:.2f}',
                     f'{latency_stats["postprocess"]:.2f}'],
        ['总延迟',    f'{latency_stats["total"]:.2f}', '', ''],
    ]
    for tl in _make_table(lat_headers, lat_rows):
        lines.append(embed_line(tl))

    lines.append('╚' + '═' * W + '╝')

    # 使用 colorstr 输出
    from ultralytics.utils import colorstr
    for line in lines:
        print(colorstr('cyan', line))


# ─────────────────── 核心测量逻辑 ───────────────────

def load_model(model_type, model_path, device, half):
    """加载模型并移至设备"""
    if model_type.lower() == 'rtdetrmm':
        from ultralytics import RTDETRMM
        model = RTDETRMM(model_path)
    else:
        from ultralytics import YOLOMM
        model = YOLOMM(model_path)

    # 确保模型在正确设备上
    if hasattr(model, 'model') and model.model is not None:
        model.model.to(device)
        if half:
            model.model.half()
    return model


def measure_pipeline_fps(model, pairs, warmup, device, half):
    """测量全管线 FPS（前处理 + 推理 + 后处理）"""
    from ultralytics.utils import LOGGER

    # 预热
    LOGGER.info(f'全管线预热: {warmup} 轮 ...')
    warmup_pairs = pairs[:min(warmup, len(pairs))]
    for rgb_path, x_path in warmup_pairs:
        try:
            model.predict(rgb_source=rgb_path, x_source=x_path, verbose=False,
                          device=device, half=half, speed=True)
        except Exception:
            pass

    # 正式测量
    LOGGER.info(f'全管线测量: {len(pairs)} 张 ...')
    fps_list = []
    preprocess_ms_list = []
    inference_ms_list = []
    postprocess_ms_list = []

    for rgb_path, x_path in pairs:
        results = model.predict(rgb_source=rgb_path, x_source=x_path, verbose=False,
                                device=device, half=half, speed=True)
        if results and len(results) > 0:
            sp = results[0].speed
            if sp:
                total_ms = sp.get('preprocess', 0) + sp.get('inference', 0) + sp.get('postprocess', 0)
                if total_ms > 0:
                    fps_list.append(1000.0 / total_ms)
                    preprocess_ms_list.append(sp.get('preprocess', 0))
                    inference_ms_list.append(sp.get('inference', 0))
                    postprocess_ms_list.append(sp.get('postprocess', 0))

    return {
        'mean': float(np.mean(fps_list)) if fps_list else 0.0,
        'std': float(np.std(fps_list)) if fps_list else 0.0,
    }, {
        'preprocess': float(np.mean(preprocess_ms_list)) if preprocess_ms_list else 0.0,
        'inference': float(np.mean(inference_ms_list)) if inference_ms_list else 0.0,
        'postprocess': float(np.mean(postprocess_ms_list)) if postprocess_ms_list else 0.0,
        'total': float(np.mean([sum(x) for x in zip(preprocess_ms_list, inference_ms_list, postprocess_ms_list)])) if preprocess_ms_list else 0.0,
    }


def preprocess_pair(rgb_path, x_path, imgsz, device, half):
    """手动预处理一对 RGB+X 图像，返回 6ch tensor"""
    import cv2

    def _load_and_transform(path):
        """读取图像 → LetterBox → ToTensor → Normalize"""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # LetterBox: 等比缩放 + 填充到 imgsz
        h, w = img.shape[:2]
        scale = min(imgsz / h, imgsz / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充
        pad_h = imgsz - new_h
        pad_w = imgsz - new_w
        top = pad_h // 2
        left = pad_w // 2
        img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # HWC → CHW, 归一化
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img

    rgb = _load_and_transform(rgb_path)
    x = _load_and_transform(x_path)

    # 拼接: [RGB, RGB, RGB, X, X, X]（与训练一致）
    combined = np.concatenate([rgb, x], axis=0)  # (6, H, W)
    tensor = torch.from_numpy(combined).unsqueeze(0).to(device)  # (1, 6, H, W)
    if half:
        tensor = tensor.half()
    return tensor


def measure_infer_fps(model, pairs, warmup, device, half, imgsz):
    """测量纯推理 FPS（仅模型前向传播，不含前后处理）"""
    from ultralytics.utils import LOGGER

    inner_model = model.model
    inner_model.eval()

    # 预热
    LOGGER.info(f'纯推理预热: {warmup} 轮 ...')
    warmup_pairs = pairs[:min(warmup, len(pairs))]
    for rgb_path, x_path in warmup_pairs:
        try:
            tensor = preprocess_pair(rgb_path, x_path, imgsz, device, half)
            with torch.no_grad():
                _ = inner_model(tensor)
        except Exception:
            pass

    # 正式测量
    LOGGER.info(f'纯推理测量: {len(pairs)} 张 ...')
    fps_list = []

    for rgb_path, x_path in pairs:
        tensor = preprocess_pair(rgb_path, x_path, imgsz, device, half)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = inner_model(tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        if elapsed_ms > 0:
            fps_list.append(1000.0 / elapsed_ms)

    return {
        'mean': float(np.mean(fps_list)) if fps_list else 0.0,
        'std': float(np.std(fps_list)) if fps_list else 0.0,
    }


def get_device_info(device):
    """获取设备信息"""
    if device == 'cpu' or not torch.cuda.is_available():
        return 'CPU', '-'
    idx = int(device) if device.isdigit() else 0
    name = torch.cuda.get_device_name(idx)
    mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    return f'{name}', f'{mem:.1f} GB'


# ─────────────────── 入口 ───────────────────

def main():
    global MODEL_PATH, DATA_YAML, MODEL_TYPE, N_SAMPLES, SPLIT, DEVICE, WARMUP, HALF, SEED, IMGSZ

    # 可选 CLI 参数覆盖
    parser = argparse.ArgumentParser(description='YOLOMM / RTDETRMM FPS 性能测试')
    parser.add_argument('--model', type=str, help='模型权重路径 (.pt)')
    parser.add_argument('--data', type=str, help='数据集 YAML 路径')
    parser.add_argument('--model-type', type=str, choices=['yolomm', 'rtdetrmm'], help='模型类型')
    parser.add_argument('--n', type=int, help='抽样数量')
    parser.add_argument('--split', type=str, help='数据集分割')
    parser.add_argument('--device', type=str, help='推理设备')
    parser.add_argument('--warmup', type=int, help='预热轮数')
    parser.add_argument('--half', action='store_true', help='使用 FP16')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--imgsz', type=int, help='推理图像尺寸')
    args = parser.parse_args()

    # CLI 参数覆盖脚本配置（仅覆盖显式传入的值）
    if args.model:      MODEL_PATH = args.model
    if args.data:       DATA_YAML = args.data
    if args.model_type: MODEL_TYPE = args.model_type
    if args.n:          N_SAMPLES = args.n
    if args.split:      SPLIT = args.split
    if args.device:     DEVICE = args.device
    if args.warmup:     WARMUP = args.warmup
    if args.half:       HALF = True
    if args.seed:       SEED = args.seed
    if args.imgsz:      IMGSZ = args.imgsz

    from ultralytics.utils import LOGGER

    # 校验
    if not Path(MODEL_PATH).exists():
        LOGGER.error(f'模型文件不存在: {MODEL_PATH}')
        LOGGER.info('请修改脚本顶部的 MODEL_PATH 配置')
        return
    if not Path(DATA_YAML).exists():
        LOGGER.error(f'数据集配置不存在: {DATA_YAML}')
        LOGGER.info('请修改脚本顶部的 DATA_YAML 配置')
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 设备信息
    device_name, device_mem = get_device_info(DEVICE)
    torch_device = torch.device(f'cuda:{DEVICE}' if DEVICE.isdigit() else DEVICE)
    dataset_name = Path(DATA_YAML).stem

    # 加载数据
    from ultralytics.tools import MultiModalSampler
    sampler = MultiModalSampler(DATA_YAML, split=SPLIT, seed=SEED)
    n_available = len(sampler)
    n_actual = min(N_SAMPLES, n_available)
    if n_actual < N_SAMPLES:
        LOGGER.warning(f'可用图像对({n_available})少于请求数量({N_SAMPLES})，使用全部 {n_actual} 对')
    pairs = sampler.sample(n=n_actual, replace=False)
    LOGGER.info(f'采样完成: {len(pairs)} 对图像')

    # 加载模型
    LOGGER.info(f'加载模型: {MODEL_TYPE.upper()} ({MODEL_PATH})')
    model = load_model(MODEL_TYPE, MODEL_PATH, torch_device, HALF)

    cfg = {
        'model_type': MODEL_TYPE,
        'model_path': MODEL_PATH,
        'dataset_name': dataset_name,
        'n_samples': n_actual,
        'imgsz': IMGSZ,
        'device_name': device_name,
        'device_mem': device_mem,
        'half': HALF,
        'seed': SEED,
    }

    # 测量全管线 FPS
    print()
    pipeline_stats, latency_stats = measure_pipeline_fps(
        model, pairs, WARMUP, DEVICE, HALF
    )

    # 测量纯推理 FPS
    infer_stats = measure_infer_fps(
        model, pairs, WARMUP, torch_device, HALF, IMGSZ
    )

    # 输出报告
    print()
    print_ascii_report(cfg, pipeline_stats, infer_stats, latency_stats)
    print()


if __name__ == '__main__':
    main()
