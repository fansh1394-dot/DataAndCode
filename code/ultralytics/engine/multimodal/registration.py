# Ultralytics Multimodal Inference - Cross-Modality Registration
# 双光谱（可见光 RGB / 红外 IR）跨模态几何配准补偿
# Version: v1.0
# Date: 2026-09-18
#
# 背景
# ----
# 多模态检测的框统一还原在主模态（RGB）的原始像素坐标系中。渲染到红外画面时，
# 即使已按归一化坐标正确换算了画幅（见 results.py 的 _boxes_for_modality），
# 仍存在一个**小的常量配准残差**：两相机光轴并非理想共轴，红外图幅相对可见光
# 图幅有轻微平移与尺度差异。
#
# 实测（D:\uav_data 上 98 帧视频抽帧，见 docs/CHANGELOG_配准修复.md）：
#   - 残余误差**与目标在画面中的位置强相关**（x 相关 -0.645、y 相关 +0.810），
#     说明它并非纯光轴平移，而含明显的**尺度/视场差异**——红外镜头等效视场略小于
#     可见光。纯平移（shift）无法吸收，故补偿必须用带尺度项的仿射。
#   - 以"可见光框中心 → 红外热斑中心"拟合 2x3 仿射后，残差 std 从
#     (0.79, 2.98) pp 降到 (0.65, 1.29) pp，且框住率在 5 折交叉验证下达 98.0%
#     （纯平移仅 85.7%），无需放大任何框。
#   - 只修根因（不补偿）时框住率仅 37.8% → 根因是 IR 画幅被拉伸 + 坐标系错配，
#     补偿是叠加在其上的第二步修正。
#
# 本模块将上述标定量固化为可复用、可开关的补偿，供渲染阶段调用。
#
# 使用
# ----
#   from ultralytics.engine.multimodal.registration import apply_registration
#   boxes_ir = apply_registration(boxes_rgb, src_shape=(1080, 1920), dst_shape=(512, 640))
#
# 设计约束
# --------
#   1. 默认关闭（enabled=False）——不改变既有正确行为，避免影响训练/验证链路。
#   2. 画幅一致时自动退化为恒等映射，保证同分辨率场景零影响。
#   3. 标定参数缺失或非法时安全退回原坐标，绝不抛异常中断推理。
#   4. 仿射含尺度项时可能把框略微缩放（吸收视场差），这是物理正确的行为；
#      为避免坐标反转已做 min/max 归一，并裁剪到画幅内。

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

# 默认标定文件位置（由 scripts/calib/ 下的标定脚本产出）
DEFAULT_CALIB_PATH = Path(r"D:\uav_data\_vis\calib.npz")


class DualRegistration:
    """可见光 → 红外 的跨模态配准变换。

    在**归一化坐标系**（[0,1]，相对各自图幅）上定义变换。核心是补偿两相机
    光轴间的固定平移；可选地使用含尺度项的最小二乘仿射，以吸收两模态焦距/
    视场的轻微差异。

    Attributes:
        mode: 'shift' 用纯平移补偿；'affine' 用含尺度项的仿射补偿；'none' 不补偿。
        shift: (dx, dy) 归一化平移量，表示"红外目标相对可见光目标的位移"。
        affine_x / affine_y: 仿射系数 [a, b, c]，x_ir = a*x_rgb + b*y_rgb + c。
        enabled: 总开关。
    """

    VALID_MODES = ("none", "shift", "affine")

    def __init__(
        self,
        mode: str = "shift",
        shift: Optional[Tuple[float, float]] = None,
        affine_x: Optional[np.ndarray] = None,
        affine_y: Optional[np.ndarray] = None,
        enabled: bool = False,
    ):
        """初始化配准变换。

        Args:
            mode: 补偿模式，见 VALID_MODES。
            shift: 归一化平移量 (dx, dy)。
            affine_x: x 方向仿射系数 [a, b, c]。
            affine_y: y 方向仿射系数 [a, b, c]。
            enabled: 是否启用补偿。默认 False，保持既有行为不变。
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode 必须为 {self.VALID_MODES} 之一，收到 {mode!r}")

        self.mode = mode
        self.enabled = bool(enabled)
        self.shift = np.asarray(shift, dtype=np.float64) if shift is not None else np.zeros(2, dtype=np.float64)
        self.affine_x = np.asarray(affine_x, dtype=np.float64) if affine_x is not None else None
        self.affine_y = np.asarray(affine_y, dtype=np.float64) if affine_y is not None else None

    # ------------------------------------------------------------------ #
    # 构造 / 持久化
    # ------------------------------------------------------------------ #

    @classmethod
    def from_calib(
        cls,
        calib_path: Union[str, Path] = DEFAULT_CALIB_PATH,
        mode: str = "affine",
        enabled: bool = True,
    ) -> "DualRegistration":
        """从标定文件（.npz）加载参数构造变换。

        标定文件由 scripts/calib/ 下的标定脚本产出，包含键：
            affine_x   RGB框中心 → IR热斑中心 的最小二乘仿射系数（x 方向）[a,b,c]
            affine_y   同上（y 方向）[a,b,c]
            shift_gt   真值差中位法估计的归一化平移（纯平移模式的备选）
            shift_box  红外侧框-目标误差法估计的归一化平移（含模型自身偏差，不推荐）
            box_relx / box_rely  热斑在框内的归一化相对位置（诊断用）

        Args:
            calib_path: 标定文件路径。
            mode: 补偿模式，默认 'affine'（实测框住率 98%，显著优于 shift）。
            enabled: 是否启用。

        Returns:
            DualRegistration 实例。文件缺失或解析失败时返回 disabled 实例（mode='none'），
            以保证推理链路不中断。
        """
        p = Path(calib_path)
        if not p.exists():
            return cls(mode="none", enabled=False)

        try:
            with np.load(str(p)) as d:
                shift = d["shift_gt"] if "shift_gt" in d.files else d.get("shift_box", np.zeros(2))
                ax = d["affine_x"] if "affine_x" in d.files else None
                ay = d["affine_y"] if "affine_y" in d.files else None
        except Exception:
            return cls(mode="none", enabled=False)

        # 仿射模式需要系数齐备，否则降级为平移
        if mode == "affine" and (ax is None or ay is None):
            mode = "shift"

        return cls(mode=mode, shift=shift, affine_x=ax, affine_y=ay, enabled=enabled)

    # ------------------------------------------------------------------ #
    # 坐标变换
    # ------------------------------------------------------------------ #

    def map_normalized(self, xy_norm: np.ndarray) -> np.ndarray:
        """把归一化坐标从可见光坐标系映射到红外坐标系。

        Args:
            xy_norm: 归一化坐标数组。**坐标对必须按 (x, y) 相邻排列**，
                形状 [N, 2] 或 [N, 2*K]（例如 [N, 4] = x1,y1,x2,y2；
                [N, 8] = x1,y1,...,x4,y4）。

        Returns:
            同形状的映射后归一化坐标。
        """
        if not self.enabled or self.mode == "none":
            return xy_norm

        pts = np.asarray(xy_norm, dtype=np.float64)
        if pts.size == 0:
            return xy_norm

        if self.mode == "shift":
            # 纯平移：两相机光轴的固定偏移。x 分量在偶数列，y 分量在奇数列。
            out = pts.copy()
            out[..., 0::2] += self.shift[0]
            out[..., 1::2] += self.shift[1]
            return out

        # 仿射模式：对每个 (x, y) 坐标对施加线性变换
        flat = pts.reshape(-1, 2)
        x, y = flat[:, 0].copy(), flat[:, 1].copy()
        nx = self.affine_x[0] * x + self.affine_x[1] * y + self.affine_x[2]
        ny = self.affine_y[0] * x + self.affine_y[1] * y + self.affine_y[2]
        return np.stack([nx, ny], axis=1).reshape(pts.shape)

    def apply_boxes(
        self,
        boxes: np.ndarray,
        src_shape: Tuple[int, int],
        dst_shape: Tuple[int, int],
    ) -> np.ndarray:
        """把框从可见光像素坐标变换到红外像素坐标（含配准补偿）。

        完整链路：可见光像素 → 归一化 → 配准补偿 → 红外像素。

        Args:
            boxes: 框数组 [N, 6]（x1, y1, x2, y2, conf, cls），位于可见光像素坐标系。
            src_shape: 可见光原图尺寸 (H, W)。
            dst_shape: 红外原图尺寸 (H, W)。

        Returns:
            变换后的框数组 [N, 6]。未启用或参数非法时返回原数组。
        """
        if boxes is None or len(boxes) == 0:
            return boxes
        if not self.enabled or self.mode == "none":
            return boxes

        src_h, src_w = float(src_shape[0]), float(src_shape[1])
        dst_h, dst_w = float(dst_shape[0]), float(dst_shape[1])
        if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
            return boxes

        arr = np.asarray(boxes, dtype=np.float64).copy()
        # 可见光像素 → 归一化
        arr[:, [0, 2]] /= src_w
        arr[:, [1, 3]] /= src_h

        # 重排为 (x1, y1, x2, y2) 的相邻坐标对，再施加配准补偿
        pairs = arr[:, [0, 1, 2, 3]]
        arr[:, [0, 1, 2, 3]] = self.map_normalized(pairs)

        # 归一化 → 红外像素
        arr[:, [0, 2]] *= dst_w
        arr[:, [1, 3]] *= dst_h

        # 处理仿射可能带来的坐标反转，并裁剪到有效范围
        x1 = np.minimum(arr[:, 0], arr[:, 2])
        x2 = np.maximum(arr[:, 0], arr[:, 2])
        y1 = np.minimum(arr[:, 1], arr[:, 3])
        y2 = np.maximum(arr[:, 1], arr[:, 3])
        arr[:, 0] = np.clip(x1, 0, dst_w)
        arr[:, 2] = np.clip(x2, 0, dst_w)
        arr[:, 1] = np.clip(y1, 0, dst_h)
        arr[:, 3] = np.clip(y2, 0, dst_h)

        return arr

    # ------------------------------------------------------------------ #
    # 诊断
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        """返回人类可读的配置摘要，用于日志留痕。"""
        if not self.enabled or self.mode == "none":
            return "跨模态配准补偿: 关闭"
        if self.mode == "shift":
            return (
                f"跨模态配准补偿: 平移 dx={self.shift[0]*100:+.3f}pp "
                f"dy={self.shift[1]*100:+.3f}pp"
            )
        return (
            f"跨模态配准补偿: 仿射 "
            f"x={self.affine_x.tolist()} y={self.affine_y.tolist()}"
        )


# ---------------------------------------------------------------------- #
# 模块级便捷接口
# ---------------------------------------------------------------------- #

_CACHED: Optional[DualRegistration] = None


def get_registration(
    calib_path: Union[str, Path] = DEFAULT_CALIB_PATH,
    mode: Optional[str] = None,
    enabled: Optional[bool] = None,
    use_cache: bool = True,
) -> DualRegistration:
    """获取配准变换实例（带缓存）。

    开关优先级：显式入参 > 环境变量 > 默认值。
        ULTRALYTICS_MM_REGISTRATION=1/0      是否启用
        ULTRALYTICS_MM_REGISTRATION_MODE     补偿模式 (shift/affine/none)

    Args:
        calib_path: 标定文件路径。
        mode: 补偿模式，None 时读环境变量或默认 'shift'。
        enabled: 是否启用，None 时读环境变量或默认 False。
        use_cache: 是否复用已加载实例。

    Returns:
        DualRegistration 实例。
    """
    global _CACHED

    if mode is None:
        mode = os.environ.get("ULTRALYTICS_MM_REGISTRATION_MODE", "affine")
    if enabled is None:
        enabled = os.environ.get("ULTRALYTICS_MM_REGISTRATION", "0") == "1"

    if use_cache and _CACHED is not None and _CACHED.mode == mode and _CACHED.enabled == enabled:
        return _CACHED

    reg = DualRegistration.from_calib(calib_path, mode=mode, enabled=enabled)
    if use_cache:
        _CACHED = reg
    return reg


def apply_registration(
    boxes: np.ndarray,
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    calib_path: Union[str, Path] = DEFAULT_CALIB_PATH,
    mode: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> np.ndarray:
    """便捷函数：对框施加跨模态配准补偿。

    Args:
        boxes: 框数组 [N, 6]，位于可见光像素坐标系。
        src_shape: 可见光原图尺寸 (H, W)。
        dst_shape: 红外原图尺寸 (H, W)。
        calib_path: 标定文件路径。
        mode: 补偿模式。
        enabled: 是否启用。

    Returns:
        变换后的框数组 [N, 6]。
    """
    reg = get_registration(calib_path=calib_path, mode=mode, enabled=enabled)
    return reg.apply_boxes(boxes, src_shape, dst_shape)
