# -*- coding: utf-8 -*-
"""双光配准：人工标注点对工具（纯几何标定第一步）。

用途
----
在一对 RGB / IR 图上，人工点选**同一物理点**在两幅图中的位置，
把归一化坐标写入 JSON。攒够 30~50 对后，交给 fit_calib_from_points.py
拟合仿射，产出与 registration.py 兼容的 calib.npz。

为什么标"点"而不是"框"
----------------------
红外热斑边界模糊、框选主观性强；同一物理点在两图上的中心位置则可精确复现。
标定点对被广泛用于多相机标定（对应点法），是几何标定的标准做法。

操作方式（OpenCV 窗口）
----------------------
    ---- RGB 窗口 ----
      鼠标左键 : 在 RGB 图上点选一个目标点
      退格键   : 撤销上一次点选
      n        : 下一对图像（已点满一对才允许）
      s        : 跳过当前对
      q / ESC  : 保存并退出

    ---- 控制台 ----
      程序会提示当前状态（已点 RGB 几个点 / 已点 IR 几个点）

标注约定（很重要）
------------------
1. 一次只标**一个**目标，点它的**几何中心**（如无人机机体中心、地面标记物中心）。
2. 先在 RGB 图上点，再到 IR 图上点**同一个物理点**。
3. 两图上的点会被按"第 k 个点"配对。若某图点错，用退格撤销重来。
4. 目标要选**两模态都清晰可见**的：红外侧优先选热特征明显的（车辆、人体、发热设备）。

用法
----
    # 1) 准备图像对：从数据集里挑若干同名 RGB/IR 对
    python scripts/calib/annotate_points.py --prepare --n 40

    # 2) 开始标注（逐对弹窗）
    python scripts/calib/annotate_points.py --annotate

    # 3) 查看标注进度
    python scripts/calib/annotate_points.py --status

输出
----
    D:\\uav_data\\_calib\\pairs.json    标注结果（增量保存，随时可续标）
    D:\\uav_data\\_calib\\images\\       --prepare 挑出的图像对副本

后续
----
    python scripts/calib/fit_calib_from_points.py     # 拟合仿射 → calib.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

WORK = Path(r"D:\uav_data\_calib")
IMGS = WORK / "images"
PAIRS_JSON = WORK / "pairs.json"

# 数据集位置（用于 --prepare 挑选图像对）
DATA_ROOT = Path(r"D:\uav_data")
RGB_SUBDIR = "images/val"
IR_SUBDIR = "images_ir/val"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def imread_u(path: Path):
    """读取图像，兼容中文/非 ASCII 路径。"""
    import cv2
    import numpy as np
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_pairs() -> dict:
    """读取已保存的标注结果；不存在则返回空结构。"""
    if PAIRS_JSON.exists():
        try:
            with open(PAIRS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[警告] 读取 {PAIRS_JSON} 失败: {exc}，将新建")
    return {"version": 1, "note": "双光配准人工标注点对（归一化坐标）", "pairs": []}


def save_pairs(data: dict) -> None:
    """保存标注结果（UTF-8，缩进便于人工检查）。"""
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = PAIRS_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(PAIRS_JSON)


def prepare_images(n: int, seed: int = 0, spread: bool = True) -> int:
    """从数据集挑选 n 对 RGB/IR 图像，复制到工作目录。

    Args:
        n: 需要多少对。
        seed: 随机种子，保证可复现。
        spread: True 时在全集里均匀散布取样（覆盖更多场景），
                False 时取前 n 个（同一段视频的连续帧）。

    Returns:
        实际复制的对数。
    """
    IMGS.mkdir(parents=True, exist_ok=True)
    rgb_dir = DATA_ROOT / RGB_SUBDIR
    ir_dir = DATA_ROOT / IR_SUBDIR
    if not rgb_dir.exists() or not ir_dir.exists():
        print(f"[错误] 找不到数据集目录:\n  {rgb_dir}\n  {ir_dir}")
        return 0

    stems = sorted(p.stem for p in rgb_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    # 只保留 IR 也存在的
    stems = [s for s in stems if any((ir_dir / (s + e)).exists() for e in IMAGE_EXTS)]
    if not stems:
        print("[错误] 没有找到同名配对的 RGB/IR 图像")
        return 0

    if spread and len(stems) > n:
        rng = random.Random(seed)
        # 均匀散布：把全集等分 n 段，每段随机取一个
        idx = []
        step = len(stems) / n
        for i in range(n):
            lo, hi = int(i * step), max(int((i + 1) * step), int(i * step) + 1)
            idx.append(rng.randrange(lo, min(hi, len(stems))))
        picks = [stems[i] for i in sorted(set(idx))]
    else:
        picks = stems[:n]

    copied = 0
    for st in picks:
        fr = next((rgb_dir / (st + e) for e in IMAGE_EXTS if (rgb_dir / (st + e)).exists()), None)
        fi = next((ir_dir / (st + e) for e in IMAGE_EXTS if (ir_dir / (st + e)).exists()), None)
        if fr is None or fi is None:
            continue
        shutil.copy2(fr, IMGS / f"{st}__rgb{fr.suffix}")
        shutil.copy2(fi, IMGS / f"{st}__ir{fi.suffix}")
        copied += 1

    print(f"[完成] 已复制 {copied} 对图像到 {IMGS}")
    print(f"       命名规则: <stem>__rgb.jpg 与 <stem>__ir.jpg")
    return copied


# --------------------------------------------------------------------------- #
# 交互标注
# --------------------------------------------------------------------------- #

class PointClicker:
    """一个窗口内的点选状态。"""

    def __init__(self, window: str, img, title: str):
        self.window = window
        self.img = img
        self.base = img.copy()
        self.title = title
        self.points: list[tuple[float, float]] = []   # 归一化坐标
        self._redraw()

    def _redraw(self) -> None:
        """重绘底图 + 已点标记。"""
        import cv2
        self.img = self.base.copy()
        h, w = self.img.shape[:2]
        for i, (nx, ny) in enumerate(self.points):
            px, py = int(nx * w), int(ny * h)
            cv2.drawMarker(self.img, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(self.img, (px, py), 9, (0, 255, 0), 2)
            cv2.putText(self.img, str(i + 1), (px + 12, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # 顶部信息条
        bar = self.img.copy()
        cv2.rectangle(bar, (0, 0), (w, 30), (0, 0, 0), -1)
        self.img = cv2.addWeighted(bar, 0.55, self.img, 0.45, 0)
        cv2.putText(self.img, f"{self.title}  points={len(self.points)}  "
                             f"[click=add  BS=undo  n=next  s=skip  q=quit]",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(self.window, self.img)

    def on_mouse(self, event, x, y, flags, param) -> None:
        """鼠标回调：左键添加点。"""
        import cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.base.shape[:2]
            self.points.append((x / w, y / h))
            self._redraw()

    def undo(self) -> None:
        """撤销最后一个点。"""
        if self.points:
            self.points.pop()
            self._redraw()


def annotate_one(rgb_path: Path, ir_path: Path, stem: str) -> dict | None:
    """对一对图像做交互标注。

    Args:
        rgb_path: RGB 图路径。
        ir_path: IR 图路径。
        stem: 图像对标识。

    Returns:
        标注结果 dict；用户跳过时返回 None。
    """
    import cv2

    rgb = imread_u(rgb_path)
    ir = imread_u(ir_path)
    if rgb is None or ir is None:
        print(f"[跳过] 图像读取失败: {stem}")
        return None

    # 两窗口并排：统一缩放到同高，便于对照观察
    TARGET_H = 560
    def fit(im):
        h, w = im.shape[:2]
        s = TARGET_H / h
        return cv2.resize(im, (int(w * s), TARGET_H))

    win_rgb, win_ir = "RGB  (visible)", "IR  (infrared)"
    cv2.namedWindow(win_rgb, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(win_ir, cv2.WINDOW_AUTOSIZE)
    cr = PointClicker(win_rgb, fit(rgb), f"{stem} RGB")
    ci = PointClicker(win_ir, fit(ir), f"{stem} IR")
    cv2.setMouseCallback(win_rgb, cr.on_mouse)
    cv2.setMouseCallback(win_ir, ci.on_mouse)

    while True:
        key = cv2.waitKey(30) & 0xFF
        # 退格
        if key in (8, 127):
            cr.undo()
            ci.undo()
            continue
        if key == ord("n"):
            if len(cr.points) == len(ci.points) and len(cr.points) > 0:
                break
            print(f"  [提示] 两图点数必须相同且不为 0（当前 RGB={len(cr.points)} "
                  f"IR={len(ci.points)}）")
            continue
        if key == ord("s"):
            cv2.destroyWindow(win_rgb)
            cv2.destroyWindow(win_ir)
            return None
        if key in (ord("q"), 27):
            cv2.destroyWindow(win_rgb)
            cv2.destroyWindow(win_ir)
            return {"__quit__": True}

    rects = {"stem": stem,
             "rgb_file": rgb_path.name, "ir_file": ir_path.name,
             "rgb_shape": [rgb.shape[1], rgb.shape[0]],
             "ir_shape": [ir.shape[1], ir.shape[0]],
             "points_rgb": [list(p) for p in cr.points],
             "points_ir": [list(p) for p in ci.points]}
    cv2.destroyWindow(win_rgb)
    cv2.destroyWindow(win_ir)
    return rects


def do_annotate(overwrite: bool = False) -> None:
    """遍历工作目录里的图像对，逐个弹窗标注。"""
    import cv2
    if not IMGS.exists():
        print(f"[错误] 工作目录不存在: {IMGS}\n       请先运行 --prepare")
        return

    rgbs = sorted(p for p in IMGS.iterdir() if "__rgb" in p.stem)
    if not rgbs:
        print(f"[错误] {IMGS} 下没有 __rgb 命名的图像\n       请先运行 --prepare")
        return

    data = load_pairs()
    done = {p["stem"] for p in data["pairs"]}
    if overwrite:
        done = set()

    todo = []
    for rp in rgbs:
        stem = rp.stem.replace("__rgb", "")
        ip = IMGS / f"{stem}__ir{rp.suffix}"
        if not ip.exists():
            print(f"[跳过] 缺 IR 配对: {stem}")
            continue
        if stem in done:
            continue
        todo.append((rp, ip, stem))

    if not todo:
        print("[完成] 所有图像对都已标注。")
        print(f"       共 {len(data['pairs'])} 对。运行 --status 查看统计。")
        return

    print("=" * 66)
    print("人工标注点对")
    print("=" * 66)
    print(f"待标注 {len(todo)} 对（已完成 {len(done)} 对）")
    print("\n操作说明：")
    print("  左键   在图上点选目标中心")
    print("  退格   撤销上一个点（两图同步撤销）")
    print("  n      完成当前对，进入下一对")
    print("  s      跳过当前对")
    print("  q/ESC  保存并退出")
    print("\n标注要点：先点 RGB，再点 IR 上**同一个物理点**；一次只标一个目标。")
    print("建议每张图标 2~4 个点（分布在不同区域），一对图内点数两边一致即可。\n")
    input("按 Enter 开始...")

    for k, (rp, ip, stem) in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {stem}")
        res = annotate_one(rp, ip, stem)
        if res is None:
            print("  已跳过")
            continue
        if res.get("__quit__"):
            print("  已退出（进度已保存）")
            break
        data["pairs"].append(res)
        save_pairs(data)
        print(f"  已保存：RGB {len(res['points_rgb'])} 点 / IR {len(res['points_ir'])} 点")

    cv2.destroyAllWindows()
    print("\n" + "=" * 66)
    print(f"当前累计 {len(data['pairs'])} 对，保存于 {PAIRS_JSON}")
    print("下一步: python scripts/calib/fit_calib_from_points.py")


def do_status() -> None:
    """打印标注进度统计。"""
    data = load_pairs()
    pairs = data.get("pairs", [])
    print(f"标注文件: {PAIRS_JSON}")
    print(f"已标注对数: {len(pairs)}")
    if not pairs:
        print("（还没有标注）")
        return
    total = sum(len(p["points_rgb"]) for p in pairs)
    shapes = {tuple(p["rgb_shape"]) for p in pairs}
    print(f"总点数: {total}")
    print(f"图像尺寸集合: {sorted(shapes)}")
    print("\n各对点数：")
    for p in pairs:
        print(f"  {p['stem']:42} RGB={len(p['points_rgb'])} IR={len(p['points_ir'])}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    """CLI 入口。"""
    ap = argparse.ArgumentParser(
        description="双光配准人工标注点对工具",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true",
                    help="从数据集挑选图像对到工作目录")
    ap.add_argument("--annotate", action="store_true", help="开始交互标注")
    ap.add_argument("--status", action="store_true", help="查看标注进度")
    ap.add_argument("--n", type=int, default=40, help="--prepare 挑多少对（默认 40）")
    ap.add_argument("--seed", type=int, default=0, help="--prepare 随机种子")
    ap.add_argument("--sequential", action="store_true",
                    help="--prepare 时取连续帧而非均匀散布")
    ap.add_argument("--overwrite", action="store_true",
                    help="--annotate 时忽略已有标注，重新标注所有对")
    args = ap.parse_args()

    if args.prepare:
        prepare_images(args.n, seed=args.seed, spread=not args.sequential)
    if args.annotate:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("[错误] 需要 opencv-python 才能标注")
            return 1
        do_annotate(overwrite=args.overwrite)
    if args.status:
        do_status()
    if not (args.prepare or args.annotate or args.status):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
