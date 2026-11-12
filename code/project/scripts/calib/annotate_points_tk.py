# -*- coding: utf-8 -*-
"""双光配准：人工标注点对工具（tkinter 版，不依赖 OpenCV GUI）。

为什么有这个版本
----------------
原版 annotate_points.py 用 cv2.imshow 弹窗，但当环境里装的是
opencv-python-headless 时（GUI: NONE），cv2.namedWindow 会直接报错：
    "The function is not implemented. Rebuild the library with ... support."

本脚本改用 **tkinter**（Python 自带，无需额外安装，也不受 opencv 版本影响）
来做标注界面，功能与原版一致，并且额外支持：
    - 图像**缩放 / 平移**（红外目标小的时候能放大点准）
    - 滚轮缩放、中键拖拽平移
    - 两图**同步缩放**（便于对照同一区域）

产出格式与原版**完全一致**，都写到 D:\\uav_data\\_calib\\pairs.json，
可直接交给 fit_calib_from_points.py 拟合。

操作方式
--------
    ---- 鼠标 ----
      左键单击     : 在当前图添加一个点（按当前序号配对）
      右键单击     : 删除该图最后一个点
      滚轮         : 以光标为中心缩放
      中键拖拽     : 平移视图
      （在任一图上缩放/平移，另一张图同步，便于对照）

    ---- 按钮 / 快捷键 ----
      Add Pair      : 完成当前对，进入下一对（两图点数需相同且 >0）
      Undo          : 撤销当前图最后一个点
      Reset View    : 恢复缩放与平移
      Skip (s)      : 跳过当前对
      Quit (q)      : 保存并退出

标注约定
--------
1. 一次只标**一个**目标，点它的**几何中心**。
2. 先在 RGB 图上点，再在 IR 图上点**同一个物理点**。
   两图上的点按"第 k 个"自动配对。
3. 目标要两模态都清晰可见；红外侧优先选热特征强的。
4. 一对图内建议点 2~4 个，尽量分布在画面**不同区域**（有利于估计尺度项）。

用法
----
    python scripts/calib/annotate_points_tk.py --prepare --n 40
    python scripts/calib/annotate_points_tk.py --annotate
    python scripts/calib/annotate_points_tk.py --status
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

WORK = Path(r"D:\uav_data\_calib")
IMGS = WORK / "images"
PAIRS_JSON = WORK / "pairs.json"

DATA_ROOT = Path(r"D:\uav_data")
RGB_SUBDIR = "images/val"
IR_SUBDIR = "images_ir/val"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 显示区最大尺寸（像素）
VIEW_W, VIEW_H = 720, 576


# --------------------------------------------------------------------------- #
# 通用
# --------------------------------------------------------------------------- #

def load_pairs() -> dict:
    """读取已保存的标注；不存在则返回空结构。"""
    if PAIRS_JSON.exists():
        try:
            with open(PAIRS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[警告] 读取 {PAIRS_JSON} 失败: {exc}，将新建")
    return {"version": 1, "note": "双光配准人工标注点对（归一化坐标）", "pairs": []}


def save_pairs(data: dict) -> None:
    """保存标注结果。"""
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = PAIRS_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(PAIRS_JSON)


def measure_brightness(path: Path) -> float:
    """测量一张图的平均亮度（0-255）。失败返回 -1。

    用 cv2 读取；不可用时退回 PIL，保证在 headless 环境也能算。
    """
    try:
        import cv2
        import numpy as np
        buf = np.fromfile(str(path), dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            return float(im.mean())
    except Exception:
        pass
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as im:
            return float(np.asarray(im.convert("L")).mean())
    except Exception:
        return -1.0


def scan_brightness(min_bright: float = 80.0, limit: int = 0,
                    verbose: bool = True) -> list:
    """扫描数据集，按 RGB 亮度筛出"看得清、可标注"的图像对。

    夜间 RGB 几乎全黑（实测有一批亮度仅约 4），这类图人工无法可靠定位目标，
    强行标注只会引入几十像素的随机误差、污染标定。故按亮度阈值预筛。

    Args:
        min_bright: RGB 平均亮度阈值，低于此值的对子被排除。
        limit: 最多返回多少对（0 表示不限）。
        verbose: 是否打印进度与统计。

    Returns:
        [(stem, brightness), ...]，按亮度从高到低排序。
    """
    import numpy as np

    rgb_dir = DATA_ROOT / RGB_SUBDIR
    ir_dir = DATA_ROOT / IR_SUBDIR
    if not rgb_dir.exists():
        print(f"[错误] 找不到 RGB 目录: {rgb_dir}")
        return []

    stems = sorted(p.stem for p in rgb_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    stems = [s for s in stems if any((ir_dir / (s + e)).exists() for e in IMAGE_EXTS)]
    if not stems:
        print("[错误] 没有找到同名配对的 RGB/IR 图像")
        return []

    rows = []
    for i, st in enumerate(stems):
        fr = next((rgb_dir / (st + e) for e in IMAGE_EXTS if (rgb_dir / (st + e)).exists()), None)
        if fr is None:
            continue
        b = measure_brightness(fr)
        if b >= 0:
            rows.append((st, b))
        if verbose and (i + 1) % 5000 == 0:
            print(f"    已扫描 {i + 1}/{len(stems)} ...", end="\r")
    if verbose:
        print(" " * 50, end="\r")

    if not rows:
        print("[错误] 未能读取任何图像亮度")
        return []

    arr = np.array([b for _, b in rows])
    keep = [(s, b) for s, b in rows if b >= min_bright]
    keep.sort(key=lambda t: -t[1])

    print("=" * 66)
    print(f"亮度扫描结果（共 {len(rows)} 对，阈值 {min_bright}）")
    print("=" * 66)
    print(f"  < 20  (几乎纯黑，无法标注) : {(arr < 20).sum():5d} 对")
    print(f"  < 40  (很暗)               : {(arr < 40).sum():5d} 对")
    print(f"  40-80 (偏暗)               : {((arr >= 40) & (arr < 80)).sum():5d} 对")
    print(f"  >= {min_bright:<3.0f}(清晰可用)           : {(arr >= min_bright).sum():5d} 对")
    print(f"  亮度中位 {np.median(arr):.1f}   最大 {arr.max():.1f}")
    print(f"\n  -> 筛出可用 {len(keep)} 对（亮度 >= {min_bright}）")

    if limit and len(keep) > limit:
        # 在可用集合里均匀取样，覆盖不同场景
        rng = random.Random(0)
        step = len(keep) / limit
        idx = []
        for i in range(limit):
            lo = int(i * step)
            hi = max(int((i + 1) * step), lo + 1)
            idx.append(rng.randrange(lo, min(hi, len(keep))))
        keep = [keep[i] for i in sorted(set(idx))]
        print(f"  -> 均匀取样后取 {len(keep)} 对")
    return keep


def prepare_images(n: int, seed: int = 0, spread: bool = True,
                   min_bright: float = 0.0) -> int:
    """从数据集挑选 n 对 RGB/IR 图像到工作目录。

    Args:
        n: 需要多少对。
        seed: 随机种子。
        spread: True 在全集均匀散布取样；False 取前 n 个。
        min_bright: 亮度阈值，>0 时只挑亮于此值的图（排除夜间黑图）。

    Returns:
        实际复制的对数。
    """
    IMGS.mkdir(parents=True, exist_ok=True)
    rgb_dir = DATA_ROOT / RGB_SUBDIR
    ir_dir = DATA_ROOT / IR_SUBDIR
    if not rgb_dir.exists() or not ir_dir.exists():
        print(f"[错误] 找不到数据集目录:\n  {rgb_dir}\n  {ir_dir}")
        return 0

    if min_bright > 0:
        # 先按亮度预筛，再在可用集合里取样
        keep = scan_brightness(min_bright=min_bright, limit=0, verbose=True)
        if not keep:
            print("[错误] 亮度筛选后没有可用图像对")
            return 0
        stems = [s for s, _ in keep]
    else:
        stems = sorted(p.stem for p in rgb_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        stems = [s for s in stems if any((ir_dir / (s + e)).exists() for e in IMAGE_EXTS)]
        if not stems:
            print("[错误] 没有找到同名配对的 RGB/IR 图像")
            return 0

    if spread and len(stems) > n:
        rng = random.Random(seed)
        idx, step = [], len(stems) / n
        for i in range(n):
            lo = int(i * step)
            hi = max(int((i + 1) * step), lo + 1)
            idx.append(rng.randrange(lo, min(hi, len(stems))))
        picks = [stems[i] for i in sorted(set(idx))]
    else:
        picks = stems[:n]

    # 清掉旧的图，避免与上一批混淆
    removed = 0
    for old in IMGS.iterdir():
        if old.is_file():
            try:
                old.unlink()
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"  [清理] 移除旧图 {removed} 个")

    copied = 0
    for st in picks:
        fr = next((rgb_dir / (st + e) for e in IMAGE_EXTS if (rgb_dir / (st + e)).exists()), None)
        fi = next((ir_dir / (st + e) for e in IMAGE_EXTS if (ir_dir / (st + e)).exists()), None)
        if fr is None or fi is None:
            continue
        shutil.copy2(fr, IMGS / f"{st}__rgb{fr.suffix}")
        shutil.copy2(fi, IMGS / f"{st}__ir{fi.suffix}")
        copied += 1

    print(f"\n[完成] 已复制 {copied} 对图像到 {IMGS}")
    return copied


# --------------------------------------------------------------------------- #
# tkinter 标注界面
# --------------------------------------------------------------------------- #

def run_gui(todo, data, single: bool = False) -> None:
    """启动 tkinter 标注界面。

    Args:
        todo: [(rgb_path, ir_path, stem), ...] 待标注列表。
        data: 已载入的标注字典（增量保存用）。
        single: 单点模式，每对图只标 1 个点。
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    class Panel:
        """单个图像面板：负责显示、缩放、平移、点选。"""

        def __init__(self, parent, title, img_path):
            self.frame = tk.Frame(parent, bd=1, relief="solid")
            self.title = title
            self.image = Image.open(img_path)
            self.w, self.h = self.image.size

            self.scale = min(VIEW_W / self.w, VIEW_H / self.h)
            self.offset = [0.0, 0.0]        # 平移（显示坐标）
            self.points = []                # 归一化坐标列表

            self.canvas = tk.Canvas(self.frame, width=VIEW_W, height=VIEW_H,
                                    bg="#222", highlightthickness=0)
            self.canvas.pack()
            self.photo = None
            self._render()

        # ---------- 坐标换算 ----------
        def to_disp(self, nx, ny):
            """归一化 -> 显示坐标。"""
            x = nx * self.w * self.scale + self.offset[0]
            y = ny * self.h * self.scale + self.offset[1]
            return x, y

        def to_norm(self, dx, dy):
            """显示坐标 -> 归一化。"""
            nx = (dx - self.offset[0]) / (self.w * self.scale)
            ny = (dy - self.offset[1]) / (self.h * self.scale)
            return nx, ny

        # ---------- 渲染 ----------
        def _render(self):
            """重绘图像与标记点。"""
            w = max(1, int(self.w * self.scale))
            h = max(1, int(self.h * self.scale))
            resized = self.image.resize((w, h), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(resized)
            self.canvas.delete("img")
            self.canvas.create_image(self.offset[0], self.offset[1],
                                     anchor="nw", image=self.photo, tags="img")
            self.canvas.tag_lower("img")
            # 画点
            self.canvas.delete("pt")
            for i, (nx, ny) in enumerate(self.points):
                x, y = self.to_disp(nx, ny)
                r = 7
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        outline="#00ff00", width=2, tags="pt")
                self.canvas.create_line(x - 12, y, x + 12, y,
                                        fill="#ffff00", width=2, tags="pt")
                self.canvas.create_line(x, y - 12, x, y + 12,
                                        fill="#ffff00", width=2, tags="pt")
                self.canvas.create_text(x + 14, y - 14, text=str(i + 1),
                                        fill="#ffff00", anchor="w",
                                        font=("Arial", 11, "bold"), tags="pt")
            self.canvas.delete("hud")
            self.canvas.create_text(
                6, 6, anchor="nw", fill="#00ff00", tags="hud",
                font=("Consolas", 10),
                text=f"{self.title}  {self.w}x{self.h}  zoom={self.scale:.2f}x  "
                     f"points={len(self.points)}")

        def zoom(self, factor, cx, cy):
            """以 (cx,cy) 为锚点缩放。"""
            old = self.scale
            new = max(0.2, min(12.0, old * factor))
            if new == old:
                return
            # 保持锚点在图像上的归一化位置不变
            nx = (cx - self.offset[0]) / (self.w * old)
            ny = (cy - self.offset[1]) / (self.h * old)
            self.scale = new
            self.offset[0] = cx - nx * self.w * new
            self.offset[1] = cy - ny * self.h * new

        def pan(self, dx, dy):
            """平移视图。"""
            self.offset[0] += dx
            self.offset[1] += dy

        def reset(self):
            """恢复默认缩放与居中。"""
            self.scale = min(VIEW_W / self.w, VIEW_H / self.h)
            self.offset = [(VIEW_W - self.w * self.scale) / 2,
                           (VIEW_H - self.h * self.scale) / 2]

    class App:
        """标注主界面。"""

        def __init__(self, root, todo, data, single=False):
            self.root = root
            self.todo = todo
            self.data = data
            self.idx = 0
            self._pan = None
            self.single = single            # 单点模式：每对图仅 1 个点
            self.pending = []               # 本次会话已确认的点对（用于实时质检）

            root.title("双光配准 · 人工标注点对")
            root.configure(bg="#111")

            self.hdr = tk.Label(root, text="", fg="#0ff", bg="#111",
                                font=("Microsoft YaHei", 11, "bold"))
            self.hdr.pack(pady=4)

            body = tk.Frame(root, bg="#111")
            body.pack()
            self.left = Panel(body, "RGB (visible)", todo[0][0])
            self.right = Panel(body, "IR (infrared)", todo[0][1])
            self.left.frame.grid(row=0, column=0, padx=6)
            self.right.frame.grid(row=0, column=1, padx=6)

            # 事件绑定
            for p in (self.left, self.right):
                c = p.canvas
                c.bind("<Button-1>", lambda e, pp=p: self.on_left(pp, e))
                c.bind("<Button-3>", lambda e, pp=p: self.on_right(pp, e))
                c.bind("<MouseWheel>", lambda e, pp=p: self.on_wheel(pp, e))
                c.bind("<Button-2>", lambda e, pp=p: self.on_pan_start(pp, e))
                c.bind("<B2-Motion>", lambda e, pp=p: self.on_pan_move(pp, e))
                c.bind("<ButtonRelease-2>", lambda e, pp=p: self.on_pan_end(pp, e))

            btns = tk.Frame(root, bg="#111")
            btns.pack(pady=6)
            for text, cmd in (("Add Pair  (n)", self.next_pair),
                              ("Undo", self.undo),
                              ("Reset View", self.reset_view),
                              ("Skip  (s)", self.skip),
                              ("Quit  (q)", self.quit)):
                tk.Button(btns, text=text, command=cmd, width=13,
                          font=("Microsoft YaHei", 9)).pack(side="left", padx=3)

            self.hint = tk.Label(
                root, bg="#111", fg="#aaa", justify="left",
                font=("Microsoft YaHei", 9),
                text="左键=加点  右键=删最后一点  滚轮=缩放  中键拖拽=平移（两图同步）｜"
                     "先在 RGB 点目标中心，再到 IR 点同一物理点；两图点数须相同")
            self.hint.pack(pady=(0, 6))

            root.bind("<Key-n>", lambda e: self.next_pair())
            root.bind("<Key-s>", lambda e: self.skip())
            root.bind("<Key-q>", lambda e: self.quit())
            root.bind("<Escape>", lambda e: self.quit())
            root.bind("<BackSpace>", lambda e: self.undo())

            self.reset_view()
            self.load_current()

        # ---------- 事件 ----------
        def on_left(self, panel, event):
            """左键：添加点。单点模式下每图只允许一个点，再点则替换。"""
            nx, ny = panel.to_norm(event.x, event.y)
            if not (0 <= nx <= 1 and 0 <= ny <= 1):
                return
            if self.single and len(panel.points) >= 1:
                # 单点模式：覆盖而非追加，避免多点错序
                panel.points = [(nx, ny)]
            else:
                panel.points.append((nx, ny))
            panel._render()
            self.update_hdr()

        def on_right(self, panel, event):
            """右键：删除该图最后一个点。"""
            if panel.points:
                panel.points.pop()
                panel._render()
                self.update_hdr()

        def on_wheel(self, panel, event):
            """滚轮：缩放（两图同步）。"""
            factor = 1.1 if event.delta > 0 else 1 / 1.1
            for p in (self.left, self.right):
                p.zoom(factor, event.x, event.y)
                p._render()

        def on_pan_start(self, panel, event):
            """中键按下：开始拖拽。"""
            self._pan = (event.x, event.y)

        def on_pan_move(self, panel, event):
            """中键拖拽：平移（两图同步）。"""
            if self._pan is None:
                return
            dx = event.x - self._pan[0]
            dy = event.y - self._pan[1]
            self._pan = (event.x, event.y)
            for p in (self.left, self.right):
                p.pan(dx, dy)
                p._render()

        def on_pan_end(self, panel, event):
            """中键释放。"""
            self._pan = None

        # ---------- 流程 ----------
        def load_current(self):
            """载入当前图像对。"""
            rgb_path, ir_path, stem = self.todo[self.idx]
            self.left = self._swap(self.left, "RGB (visible)", rgb_path, 0)
            self.right = self._swap(self.right, "IR (infrared)", ir_path, 1)
            self.hdr.config(text=f"[{self.idx + 1}/{len(self.todo)}]  {stem}")
            self.update_hdr()

        def _swap(self, old, title, img_path, col):
            """替换某个面板（换图）。"""
            old.frame.destroy()
            p = Panel(self.left.frame.master, title, img_path)
            p.frame.grid(row=0, column=col, padx=6)
            c = p.canvas
            c.bind("<Button-1>", lambda e, pp=p: self.on_left(pp, e))
            c.bind("<Button-3>", lambda e, pp=p: self.on_right(pp, e))
            c.bind("<MouseWheel>", lambda e, pp=p: self.on_wheel(pp, e))
            c.bind("<Button-2>", lambda e, pp=p: self.on_pan_start(pp, e))
            c.bind("<B2-Motion>", lambda e, pp=p: self.on_pan_move(pp, e))
            c.bind("<ButtonRelease-2>", lambda e, pp=p: self.on_pan_end(pp, e))
            p.reset()
            p._render()
            return p

        def update_hdr(self):
            """更新标题栏提示（含实时质检信息）。"""
            rgb_path, ir_path, stem = self.todo[self.idx]
            n_r, n_i = len(self.left.points), len(self.right.points)
            flag = "  ✓ 可进入下一对" if (n_r == n_i and n_r > 0) else "  ⚠ 两图点数须相同且>0"
            extra = ""
            # 实时质检：用已确认的点拟合，显示当前点的预测偏差
            if not self.single and len(self.pending) >= 8 and n_r == n_i and n_r > 0:
                import numpy as _np
                P = _np.array([[a, b] for a, b, _ in self.pending])
                Q = _np.array([[c, d] for _, c, d in self.pending])
                A = _np.hstack([P, _np.ones((len(P), 1))])
                try:
                    cx, *_ = _np.linalg.lstsq(A, Q[:, 0], rcond=None)
                    cy, *_ = _np.linalg.lstsq(A, Q[:, 1], rcond=None)
                    for k in range(n_r):
                        px, py = self.left.points[k]
                        gx, gy = self.right.points[k]
                        ex = (cx[0] * px + cx[1] * py + cx[2] - gx) * 640
                        ey = (cy[0] * px + cy[1] * py + cy[2] - gy) * 512
                        d = (ex ** 2 + ey ** 2) ** 0.5
                        mark = "OK" if d < 8 else ("偏大" if d < 20 else "★疑似错配")
                        extra += f"   点{k + 1}: {d:5.1f}px {mark}"
                except Exception:
                    pass
            self.hdr.config(text=f"[{self.idx + 1}/{len(self.todo)}]  {stem}   "
                                 f"RGB={n_r}  IR={n_i}{flag}{extra}")

        def reset_view(self):
            """恢复两图默认视图。"""
            for p in (self.left, self.right):
                p.reset()
                p._render()

        def undo(self):
            """撤销当前图最后一个点（优先作用在后来操作的那张）。"""
            for p in (self.right, self.left):
                if p.points:
                    p.points.pop()
                    p._render()
                    self.update_hdr()
                    return

        def save_current(self) -> bool:
            """保存当前对；点数不一致时返回 False。"""
            n_r, n_i = len(self.left.points), len(self.right.points)
            if n_r != n_i or n_r == 0:
                return False
            rgb_path, ir_path, stem = self.todo[self.idx]
            rec = {
                "stem": stem,
                "rgb_file": rgb_path.name,
                "ir_file": ir_path.name,
                "rgb_shape": [self.left.w, self.left.h],
                "ir_shape": [self.right.w, self.right.h],
                "points_rgb": [[round(x, 6), round(y, 6)] for x, y in self.left.points],
                "points_ir": [[round(x, 6), round(y, 6)] for x, y in self.right.points],
            }
            self.data["pairs"].append(rec)
            # 记录到本次会话的待质检列表
            for k in range(n_r):
                a, b = self.left.points[k]
                c, d = self.right.points[k]
                self.pending.append((a, b, c, d))
            save_pairs(self.data)
            print(f"  已保存 {stem}：RGB {n_r} 点 / IR {n_i} 点")
            return True

        def next_pair(self):
            """完成当前对并前进。"""
            if not self.save_current():
                n_r, n_i = len(self.left.points), len(self.right.points)
                print(f"  [提示] 两图点数须相同且 >0（当前 RGB={n_r} IR={n_i}）")
                self.update_hdr()
                return
            self.idx += 1
            if self.idx >= len(self.todo):
                print("\n[完成] 所有待标注图像对已处理。")
                self.root.quit()
                return
            self.load_current()

        def skip(self):
            """跳过当前对。"""
            rgb_path, ir_path, stem = self.todo[self.idx]
            print(f"  已跳过 {stem}")
            self.idx += 1
            if self.idx >= len(self.todo):
                self.root.quit()
                return
            self.load_current()

        def quit(self):
            """保存并退出。"""
            print("\n[退出] 进度已保存。")
            self.root.quit()

    root = tk.Tk()
    App(root, todo, data, single=single)
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass


def do_annotate(overwrite: bool = False, single: bool = False) -> None:
    """准备待标注列表并启动界面。

    Args:
        overwrite: 忽略已有标注，重新标注所有对。
        single: 单点模式，每对图只标 1 个点（避免多点错序）。
    """
    if not IMGS.exists():
        print(f"[错误] 工作目录不存在: {IMGS}\n       请先运行 --prepare")
        return

    rgbs = sorted(p for p in IMGS.iterdir() if "__rgb" in p.stem)
    if not rgbs:
        print(f"[错误] {IMGS} 下没有 __rgb 命名的图像\n       请先运行 --prepare")
        return

    data = load_pairs()
    done = set() if overwrite else {p["stem"] for p in data["pairs"]}

    todo = []
    for rp in rgbs:
        stem = rp.stem.replace("__rgb", "")
        ip = IMGS / f"{stem}__ir{rp.suffix}"
        if not ip.exists():
            continue
        if stem in done:
            continue
        todo.append((rp, ip, stem))

    if not todo:
        print("[完成] 所有图像对都已标注。")
        print(f"       共 {len(data['pairs'])} 对。运行 --status 查看统计。")
        return

    print("=" * 66)
    print("人工标注点对（tkinter 版）")
    print("=" * 66)
    print(f"待标注 {len(todo)} 对（已完成 {len(done)} 对）")
    if single:
        print("模式：**单点模式**（每对图只标 1 个点，彻底避免多点错序）")
    print("\n界面操作：")
    print("  左键        在图上点选目标中心")
    print("  右键        删除该图最后一个点")
    print("  滚轮        以光标为中心缩放（两图同步）")
    print("  中键拖拽    平移视图（两图同步）")
    print("  Add Pair/n  完成当前对，进入下一对")
    print("  Undo/退格   撤销一个点")
    print("  Skip/s      跳过当前对")
    print("  Quit/q      保存并退出")
    if single:
        print("标注要点：先点 RGB，再点 IR 上**同一个物理点**；每对图只标 1 个点。\n")
    else:
        print("标注要点：先点 RGB，再点 IR 上**同一个物理点**；一对图 2~4 个点。\n")

    run_gui(todo, data, single=single)

    print("\n" + "=" * 66)
    print(f"当前累计 {len(data['pairs'])} 对，保存于 {PAIRS_JSON}")
    print("下一步: python scripts/calib/fit_calib_from_points.py")


def do_check() -> None:
    """质检：鲁棒拟合 + 离群点定位。

    用迭代最小二乘剔除离群点，报告内点残差与可疑点清单，
    帮助在正式拟合前发现"标错物理点"或"点序错配"的样本。
    """
    import numpy as np

    data = load_pairs()
    pairs = data.get("pairs", [])
    if not pairs:
        print("（还没有标注，无法质检）")
        return

    P, Q, meta = [], [], []
    for p in pairs:
        pr, pi = p.get("points_rgb", []), p.get("points_ir", [])
        for k in range(min(len(pr), len(pi))):
            P.append(pr[k])
            Q.append(pi[k])
            meta.append((p["stem"], k + 1))
    P, Q = np.array(P, float), np.array(Q, float)
    n = len(P)

    print("=" * 66)
    print(f"标注质检   共 {n} 个点，来自 {len(pairs)} 对图")
    print("=" * 66)
    if n < 6:
        print("点太少（<6），无法拟合仿射。请继续标注。")
        return

    def fit(PP, QQ):
        A = np.hstack([PP, np.ones((len(PP), 1))])
        cx, *_ = np.linalg.lstsq(A, QQ[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(A, QQ[:, 1], rcond=None)
        return cx, cy

    def apply(PP, cx, cy):
        A = np.hstack([PP, np.ones((len(PP), 1))])
        return np.stack([A @ cx, A @ cy], axis=1)

    def dist(pr, qq):
        d = pr - qq
        return np.hypot(d[:, 0] * 640, d[:, 1] * 512)

    # 全量
    cx, cy = fit(P, Q)
    d_all = dist(apply(P, cx, cy), Q)
    print(f"\n全量最小二乘: 中位 {np.median(d_all):.2f}px  "
          f"p90 {np.percentile(d_all, 90):.2f}px  max {d_all.max():.2f}px")

    # 鲁棒迭代
    mask = np.ones(n, bool)
    for _ in range(8):
        cxr, cyr = fit(P[mask], Q[mask])
        dd = dist(apply(P, cxr, cyr), Q)
        thr = max(3.0, np.median(dd[mask]) * 3)
        new = dd < thr
        if new.sum() == mask.sum():
            break
        mask = new

    d_in = dd[mask]
    print(f"\n鲁棒拟合（剔除离群后）: 内点 {mask.sum()}/{n}")
    print(f"  内点残差: 中位 {np.median(d_in):.2f}px  "
          f"p90 {np.percentile(d_in, 90):.2f}px  max {d_in.max():.2f}px")

    # 健康度判据
    med = np.median(d_in)
    if med < 3:
        verdict = "优秀 —— 可以用于拟合标定"
    elif med < 6:
        verdict = "良好 —— 可以用于拟合标定"
    elif med < 10:
        verdict = "一般 —— 建议复查离群点后重标"
    else:
        verdict = "较差 —— 标注存在系统性错配，建议重标"
    print(f"\n  健康度: {verdict}")

    # 离群点清单
    bad = np.where(~mask)[0]
    if len(bad):
        print(f"\n离群点（建议复查/重标）共 {len(bad)} 个:")
        for i in bad[np.argsort(-d_all[bad])][:15]:
            st, k = meta[i]
            print(f"  {st:44} 第{k}点  {d_all[i]:6.2f}px")

    # 按图对聚合
    by = {}
    for i, (st, _) in enumerate(meta):
        by.setdefault(st, []).append(d_all[i])
    worst = sorted(((st, np.median(v)) for st, v in by.items()),
                   key=lambda t: -t[1])[:12]
    print(f"\n偏差最大的图像对:")
    for st, m in worst:
        print(f"  {st:44} 中位 {m:6.2f}px")
    print("\n建议: 对上面列出的对，用 --overwrite 重标（或删掉该对后重标）")


def do_status() -> None:
    """打印标注进度。"""
    data = load_pairs()
    pairs = data.get("pairs", [])
    print(f"标注文件: {PAIRS_JSON}")
    print(f"已标注对数: {len(pairs)}")
    if not pairs:
        print("（还没有标注）")
        return
    print(f"总点数: {sum(len(p['points_rgb']) for p in pairs)}")
    print(f"图像尺寸集合: {sorted({tuple(p['rgb_shape']) for p in pairs})}")
    print("\n各对点数：")
    for p in pairs:
        print(f"  {p['stem']:44} RGB={len(p['points_rgb'])} IR={len(p['points_ir'])}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    """CLI 入口。"""
    ap = argparse.ArgumentParser(
        description="双光配准人工标注点对工具（tkinter 版，不依赖 OpenCV GUI）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="从数据集挑选图像对")
    ap.add_argument("--scan", action="store_true",
                    help="扫描数据集亮度分布（不复制图片，仅统计）")
    ap.add_argument("--min-bright", type=float, default=80.0,
                    help="亮度阈值，--prepare 时只挑亮于此值的图（默认 80，排除夜间黑图）")
    ap.add_argument("--annotate", action="store_true", help="开始交互标注")
    ap.add_argument("--status", action="store_true", help="查看标注进度")
    ap.add_argument("--n", type=int, default=40, help="--prepare 挑多少对")
    ap.add_argument("--seed", type=int, default=0, help="--prepare 随机种子")
    ap.add_argument("--sequential", action="store_true", help="取连续帧而非均匀散布")
    ap.add_argument("--overwrite", action="store_true", help="重新标注所有对")
    ap.add_argument("--single", action="store_true",
                    help="单点模式：每对图只标 1 个点（推荐，避免多点错序）")
    ap.add_argument("--check", action="store_true",
                    help="质检：对已标注点做鲁棒拟合，列出偏差大的点")
    args = ap.parse_args()

    if args.prepare:
        prepare_images(args.n, seed=args.seed, spread=not args.sequential,
                       min_bright=args.min_bright)
    if args.scan:
        scan_brightness(min_bright=args.min_bright, limit=args.n, verbose=True)
    if args.annotate:
        try:
            import tkinter  # noqa: F401
            import PIL.Image  # noqa: F401
        except ImportError as exc:
            print(f"[错误] 缺少依赖: {exc}")
            print("       tkinter 为 Python 自带；PIL 需 pip install pillow")
            return 1
        do_annotate(overwrite=args.overwrite, single=args.single)
    if args.check:
        do_check()
    if args.status:
        do_status()
    if not (args.prepare or args.annotate or args.status or args.check or args.scan):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
