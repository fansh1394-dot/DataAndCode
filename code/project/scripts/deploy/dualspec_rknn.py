# -*- coding: utf-8 -*-
"""RK3588 端实时双光谱检测：采集 → 预处理 → 推理 → 后处理 → 显示。

设计约束
--------
1. **三线程流水线**：采集 / 推理 / 显示解耦。
   单线程在 RK3588 上会被 USB 采集阻塞（V4L2 read 会等下一帧），
   导致实际帧率只有推理帧率的一半。
2. **BUFFERSIZE=1**：实时监控必须丢弃积压帧，否则延迟会累积到数秒。
3. **letterbox 严格复现训练**：等比缩放 + 114 补边，不能直接 resize 拉伸。
4. **NMS-free**：YOLO26 的 one2one 分支已输出最终框，板上不做 NMS。
5. **双模态路由**：RGB 走 rgb 头、IR 走 ir 头，按帧级置信度切换
   （阈值来自 test 实测 tau_low=0.30 / tau_high=0.50）。

用法（板端）
------------
    # 双光谱（两个 USB 摄像头）
    python dualspec_rknn.py --rgb-dev 0 --ir-dev 2 \
        --rgb-model yolo26_rgb.rknn --ir-model yolo26_ir.rknn

    # 单模态（只有 RGB 摄像头）
    python dualspec_rknn.py --rgb-dev 0 --rgb-model yolo26_rgb.rknn

    # 无桌面（headless）推 RTSP
    python dualspec_rknn.py --rgb-dev 0 --ir-dev 2 \
        --rgb-model a.rknn --ir-model b.rknn --headless --rtsp rtsp://0.0.0.0:8554/ds

依赖：``rknnlite``（板端）、``opencv-python``、``numpy``。
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:  # 桌面调试用
    RKNNLite = None


PAD_VALUE = 114
IR_EVERAGE_THRESH = 0.30   # 帧级路由下限（RGB 置信度低于此值才考虑切 IR）
IR_SWITCH_THRESH = 0.50    # 帧级路由上限（IR 置信度高于此值才允许切）


# ======================================================================
# 数据结构
# ======================================================================
@dataclass
class Frame:
    """一帧待推理数据。idx 用于对齐两路流，避免奇偶帧错配。"""
    idx: int
    rgb: Optional[np.ndarray] = None
    ir: Optional[np.ndarray] = None
    t_cap: float = field(default_factory=time.time)


@dataclass
class Detection:
    box: Tuple[float, float, float, float]   # x1,y1,x2,y2（原图坐标）
    conf: float
    cls: int
    source: str                               # 'rgb' / 'ir'
    is_candidate: bool = False                # 置信度落在候选带


# ======================================================================
# 采集线程
# ======================================================================
class Capture(threading.Thread):
    """单路 USB 采集。MJPEG 优先（RK3588 软解 MJPEG 比 YUYV 快）。"""

    def __init__(self, dev: int, out_q: 'queue.Queue[Frame]', gray: bool,
                 size: Tuple[int, int] = (640, 512), fps: int = 30):
        super().__init__(daemon=True, name=f'cap{dev}')
        self.dev = dev
        self.q = out_q
        self.gray = gray
        self.size = size
        self.fps = fps
        self.running = True
        self.cap = None
        self.last_err = ''

    def _open(self):
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f'摄像头 {self.dev} 打开失败')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 实时监控关键：只保留最新一帧，防止延迟累积
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        return cap

    def run(self):
        try:
            cap = self._open()
        except RuntimeError as e:
            self.last_err = str(e)
            return
        idx = 0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.gray and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 队列满时丢最旧帧，保证实时性
            try:
                self.q.put_nowait(Frame(idx=idx, **({'rgb': frame} if not self.gray else {'ir': frame})))
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.q.put_nowait(Frame(idx=idx, **({'rgb': frame} if not self.gray else {'ir': frame})))
                except queue.Empty:
                    pass
            idx += 1
        cap.release()

    def stop(self):
        self.running = False


# ======================================================================
# 预处理
# ======================================================================
def letterbox(img: np.ndarray, imgsz: int = 640) -> Tuple[np.ndarray, float, int, int]:
    """等比缩放 + 居中补 114，返回 (画布, 缩放比, 左补宽, 上补高)。

    必须与训练时的 LetterBox 完全一致，否则框会系统性偏移。
    """
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas = np.full((imgsz, imgsz, 3), PAD_VALUE, np.uint8)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def preprocess(rgb: Optional[np.ndarray], ir: Optional[np.ndarray],
               imgsz: int = 640, dual: bool = True):
    """构建 RKNN 输入。

    dual=True : 6 通道（RGB 3ch + IR 3ch），IR 缺失时补零
    dual=False: 3 通道单模态
    返回 (NCHW float32 [0,1], (r, dw, dh))
    """
    src = rgb if rgb is not None else ir
    base, r, dw, dh = letterbox(src, imgsz)
    if not dual:
        x = base[:, :, ::-1].transpose(2, 0, 1)   # BGR→RGB, HWC→NCHW
    else:
        if ir is None:
            ir3 = np.zeros_like(base)
        else:
            ir3, _, _, _ = letterbox(ir, imgsz)
        mm = np.concatenate([base, ir3], axis=2)  # 训练顺序：RGB 在前，IR 在后
        x = mm.transpose(2, 0, 1)
    return np.ascontiguousarray(x[None], dtype=np.float32) / 255.0, (r, dw, dh)


# ======================================================================
# 推理
# ======================================================================
class RKNNModel:
    """单头 RKNN 模型封装。"""

    def __init__(self, path: str, imgsz: int = 640, nch: int = 6):
        if RKNNLite is None:
            raise RuntimeError('rknnlite 不可用：本脚本只能在 RK3588 板端运行')
        self.imgsz = imgsz
        self.nch = nch
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(path) != 0:
            raise RuntimeError(f'load_rknn 失败: {path}')
        # 三核并行（RK3588 有 3 个 NPU core）
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            raise RuntimeError('init_runtime 失败')

    def infer(self, x: np.ndarray) -> np.ndarray:
        out = self.rknn.inference(inputs=[x])
        return np.asarray(out[0])

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass


def decode(pred: np.ndarray, meta, orig_hw, conf_thres: float, cand_thres: float,
           source: str) -> List[Detection]:
    """YOLO26 NMS-free 输出解码 + 还原到原图坐标。

    期望 pred 形状 [N, 6] = (x1, y1, x2, y2, conf, cls)，坐标在 letterbox 画布上。
    若实际导出的是 [6, N]，此处会转置。
    """
    r, dw, dh = meta
    p = pred.reshape(-1, pred.shape[-1]) if pred.ndim > 1 else pred[None]
    if p.shape[1] != 6 and p.shape[0] == 6:
        p = p.T
    if p.shape[1] < 6:
        # 兼容 [N, 4+nc] 的 raw 输出：取最大类别分作为 conf
        boxes, scores = p[:, :4], p[:, 4:]
        conf = scores.max(axis=1)
        cls = scores.argmax(axis=1)
    else:
        boxes, conf, cls = p[:, :4], p[:, 4], p[:, 5].astype(int)

    keep = conf >= cand_thres
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
    if len(boxes) == 0:
        return []

    # letterbox → 原图
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    H, W = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

    return [Detection(tuple(map(float, b)), float(c), int(k), source,
                      is_candidate=(c < conf_thres))
            for b, c, k in zip(boxes, conf, cls)]


# ======================================================================
# 帧级路由（决策级晚融合，test 实测 +0.00635 mAP）
# ======================================================================
def route_frame(det_rgb: List[Detection], det_ir: List[Detection]) -> Tuple[str, List[Detection]]:
    """按帧切换模态。

    规则（eval_late_fusion.py 已验证）：
      RGB 最高置信度 < IR_EVERAGE_THRESH 且 IR 最高置信度 > IR_SWITCH_THRESH
      → 用 IR 结果；否则用 RGB。
    """
    c_rgb = max((d.conf for d in det_rgb), default=0.0)
    c_ir = max((d.conf for d in det_ir), default=0.0)
    if c_rgb < IR_EVERAGE_THRESH and c_ir > IR_SWITCH_THRESH:
        return 'ir', det_ir
    return 'rgb', det_rgb


# ======================================================================
# 绘制
# ======================================================================
def draw(frame: np.ndarray, dets: List[Detection], src: str, fps: float) -> np.ndarray:
    """绿色实线=确认，黄色虚线=候选。"""
    vis = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.box]
        color = (0, 215, 255) if d.is_candidate else (0, 255, 0)
        if d.is_candidate:
            # 虚线：每 8px 画一段
            for seg in (('h', y1), ('h', y2)):
                for x in range(x1, x2, 16):
                    cv2.line(vis, (x, seg[1]), (min(x + 8, x2), seg[1]), color, 2)
            for seg in (('v', x1), ('v', x2)):
                for y in range(y1, y2, 16):
                    cv2.line(vis, (seg[1], y), (seg[1], min(y + 8, y2)), color, 2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tag = f'{d.cls} {d.conf:.2f}' + (' 候选' if d.is_candidate else '')
        cv2.putText(vis, tag, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    info = f'{src.upper()}  {fps:5.1f} FPS  det={len(dets)}'
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ======================================================================
# 主循环
# ======================================================================
def run(args):
    q_rgb: 'queue.Queue[Frame]' = queue.Queue(maxsize=2)
    q_ir: 'queue.Queue[Frame]' = queue.Queue(maxsize=2)

    caps = []
    cap_rgb = Capture(args.rgb_dev, q_rgb, gray=False,
                      size=(args.width, args.height), fps=args.fps)
    cap_rgb.start()
    caps.append(cap_rgb)
    if args.ir_dev is not None:
        cap_ir = Capture(args.ir_dev, q_ir, gray=True,
                         size=(args.width, args.height), fps=args.fps)
        cap_ir.start()
        caps.append(cap_ir)

    time.sleep(0.8)  # 等摄像头稳定
    for c in caps:
        if c.last_err:
            print(f'[错误] {c.last_err}')
            return 2

    dual = args.ir_model is not None
    m_rgb = RKNNModel(args.rgb_model, args.imgsz, nch=6 if dual else 3)
    m_ir = RKNNModel(args.ir_model, args.imgsz, nch=6) if dual else None

    writer = None
    if args.headless and args.rtsp:
        writer = cv2.VideoWriter(
            args.rtsp, cv2.CAP_GSTREAMER, 0, args.fps, (args.width, args.height))

    t_last, fps = time.time(), 0.0
    frame_i = 0
    try:
        while True:
            try:
                f_rgb = q_rgb.get(timeout=2.0)
            except queue.Empty:
                print('[警告] 2 秒未取到 RGB 帧，退出')
                break
            f_ir = None
            if dual:
                try:
                    f_ir = q_ir.get(timeout=0.5)
                except queue.Empty:
                    pass

            # ---- 推理 ----
            det_rgb, det_ir = [], []
            if f_rgb.rgb is not None and m_rgb is not None:
                x, meta = preprocess(f_rgb.rgb, f_ir.ir if f_ir else None,
                                     args.imgsz, dual=dual and not args.single_head)
                out = m_rgb.infer(x)
                det_rgb = decode(out, meta, f_rgb.rgb.shape[:2],
                                 args.conf, args.cand, 'rgb')
            if dual and m_ir is not None and f_ir is not None and f_ir.ir is not None:
                x_ir, meta_ir = preprocess(None, f_ir.ir, args.imgsz, dual=False)
                out_ir = m_ir.infer(x_ir)
                det_ir = decode(out_ir, meta_ir, f_ir.ir.shape[:2],
                                args.conf, args.cand, 'ir')

            src, dets = route_frame(det_rgb, det_ir) if dual else ('rgb', det_rgb)

            # ---- FPS（滑窗）----
            now = time.time()
            dt = now - t_last
            t_last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            # ---- 显示 ----
            base_img = f_rgb.rgb if f_rgb.rgb is not None else f_ir.ir
            vis = draw(base_img, dets, src, fps)
            frame_i += 1
            if writer is not None:
                writer.write(vis)
            else:
                cv2.imshow('dual-spec detect', vis)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    finally:
        for c in caps:
            c.stop()
        m_rgb.release()
        if m_ir is not None:
            m_ir.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f'[结束] 共处理 {frame_i} 帧')
    return 0


def build_argparser():
    ap = argparse.ArgumentParser(description='RK3588 实时双光谱检测')
    ap.add_argument('--rgb-dev', type=int, default=0, help='RGB 摄像头设备号')
    ap.add_argument('--ir-dev', type=int, default=None, help='IR 摄像头设备号（不给则单模态）')
    ap.add_argument('--rgb-model', required=True, help='RGB/fuse 头 .rknn')
    ap.add_argument('--ir-model', default=None, help='IR 头 .rknn')
    ap.add_argument('--single-head', action='store_true',
                    help='RGB 模型是 3 通道单模态（默认按 dual 传 6 通道）')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--conf', type=float, default=0.5, help='确认阈值（绿框）')
    ap.add_argument('--cand', type=float, default=0.25, help='候选阈值（黄虚线）')
    ap.add_argument('--headless', action='store_true', help='无桌面，走 RTSP 推流')
    ap.add_argument('--rtsp', default=None, help='RTSP 输出地址（GStreamer）')
    return ap


if __name__ == '__main__':
    import sys
    sys.exit(run(build_argparser().parse_args()))
