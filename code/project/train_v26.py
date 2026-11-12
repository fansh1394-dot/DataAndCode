#!/usr/bin/env python3
"""YOLO26 专用训练入口（与 YOLO11 彻底隔离）。

为什么需要这个文件
------------------
`train_full.py` 同时承载 YOLO11 与 YOLO26 两代配置，`CONFIG_ALIASES` 里
13 条 YOLO11 别名 + 2 条 v26 别名混在一张表里。一旦有人漏写 `--config`
或误用了 yolo11 别名，实验会静默跑成另一代。本入口用**白名单 + 强制断言**
把这条路径堵死：

===========================  ==========================================
约束                          行为
===========================  ==========================================
`--config`                    只能是 `V26_CONFIGS` 里的别名，或路径中
                              含 `/yolo26/` 的 YAML；否则直接退出 2
`--project`                   强制为 `YOLO26/ResTest`，不接受覆盖
`--single-modal` / 双模态      必须与配置自洽（单头 yaml ↔ 单模态）
`--label-from`                单模态 RGB 默认 `vis`，IR 默认 `ir`，
                              双头默认 `dual`
续训 `--resume`               只认 `YOLO26/ResTest/<name>/weights/last.pt`
===========================  ==========================================

用法::

    python train_v26.py --config v26-single-baseline --single-modal rgb \
        --label-from vis --name v26_E1_rgbonly_s0 --seed 0 --amp false

其余参数与 `train_full.py` 完全一致（本文件只是透传，不重复实现训练逻辑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent

#: YOLO26 允许的配置别名（白名单；不在表内一律拒绝）
V26_CONFIGS = {
    "v26-single-baseline",
    "v26-dual-fusion-deform",
}

#: 单头（单模态）配置：必须配 --single-modal
V26_SINGLE_MODAL_CONFIGS = {"v26-single-baseline"}

#: 四头/双头配置：必须不配 --single-modal，且强制 --label-from dual
V26_DUAL_CONFIGS = {"v26-dual-fusion-deform"}

#: YOLO26 结果根目录（相对项目根），不接受覆盖
V26_PROJECT = "YOLO26/ResTest"


def _fail(msg: str, hint: str = "") -> int:
    print(f"\n[YOLO26 入口拒绝] {msg}")
    if hint:
        print(f"                   {hint}")
    return 2


def main() -> int:
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print("提示：其余参数与 train_full.py 相同，本入口只负责代际校验与强制隔离。")
        return 0

    # ---- 1. 解析本入口关心的几个参数（不引入 argparse，避免与 train_full 冲突）----
    def _get(flag: str) -> str | None:
        for i, a in enumerate(argv):
            if a == flag:
                return argv[i + 1] if i + 1 < len(argv) else ""
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
        return None

    config = _get("--config")
    if not config:
        return _fail(
            "缺少 --config。",
            f"可用：{' | '.join(sorted(V26_CONFIGS))}",
        )

    key = config.strip()
    is_alias = key in V26_CONFIGS
    is_yolo26_path = "/yolo26/" in key.replace("\\", "/")
    if not (is_alias or is_yolo26_path):
        return _fail(
            f"--config {key} 不是 YOLO26 配置。",
            f"本入口只接受 {' | '.join(sorted(V26_CONFIGS))} 或含 /yolo26/ 的 YAML 路径。",
        )

    # ---- 2. --project 强制为 YOLO26/ResTest ----
    given_project = _get("--project")
    if given_project is not None and given_project.strip().strip("/") != V26_PROJECT:
        return _fail(
            f"--project {given_project} 与 YOLO26 隔离约定冲突。",
            f"YOLO26 结果必须落在 {V26_PROJECT}（由本入口强制，无需传 --project）。",
        )
    # 统一把 --project 注入为固定值（覆盖掉用户传的或缺失的情况）
    argv = [a for i, a in enumerate(argv)
            if a != "--project" and not a.startswith("--project=")
            and not (i > 0 and argv[i - 1] == "--project")]
    argv += ["--project", V26_PROJECT]

    # ---- 3. 单模态 / 双头与配置自洽 ----
    single_modal = _get("--single-modal")
    if key in V26_SINGLE_MODAL_CONFIGS and not single_modal:
        return _fail(
            f"{key} 是单头（单模态）配置，必须显式指定 --single-modal rgb|x。",
            "否则会构建出带多模态路由的双输入结构，与 YAML 自相矛盾。",
        )
    if key in V26_DUAL_CONFIGS and single_modal:
        return _fail(
            f"{key} 是双头融合配置，不能配 --single-modal。",
            "双头需要 RGB/IR 两条通路同时存在。",
        )

    # ---- 4. 标签口径校验（防跨代串用）----
    label_from = _get("--label-from")
    if key in V26_DUAL_CONFIGS and label_from not in (None, "dual"):
        return _fail(
            f"双头配置的 --label-from 必须是 dual（当前 {label_from}）。",
            "否则 IR 头与 RGB 头同源监督，融合分支失去意义。",
        )
    if key in V26_SINGLE_MODAL_CONFIGS and single_modal == "rgb" and label_from not in (None, "vis"):
        return _fail(
            f"RGB 单模态基线应用 --label-from vis（当前 {label_from}）。",
            "dual 会把 data.yaml 写成双标签映射，与历史 E1 口径不一致。",
        )
    if key in V26_SINGLE_MODAL_CONFIGS and single_modal == "x" and label_from not in (None, "ir"):
        return _fail(
            f"IR 单模态基线应用 --label-from ir（当前 {label_from}）。",
            "用 labels/ 会让 IR 头继承 RGB-IR 的系统性偏移。",
        )

    # ---- 5. 续训只认 YOLO26 目录下的 ckpt ----
    if "--resume" in argv:
        name = _get("--name")
        if not name:
            return _fail("--resume 必须同时给 --name。")
        ckpt = PROJ / V26_PROJECT / name / "weights" / "last.pt"
        if not ckpt.is_file():
            return _fail(
                f"续训 ckpt 不存在：{ckpt}",
                "YOLO26 续训只读取 YOLO26/ResTest 下的权重，不会跨代去 ResTest/ 找。",
            )

    print(f"[YOLO26 入口] config={key}  project={V26_PROJECT}  "
          f"single_modal={single_modal or '-'}  label_from={label_from or 'auto'}")

    # ---- 6. 透传给 train_full.py（同进程调用，保证参数解析只有一套）----
    sys.path.insert(0, str(PROJ))
    import train_full

    sys.argv = [str(PROJ / "train_full.py")] + argv
    return train_full.main()


if __name__ == "__main__":
    raise SystemExit(main())
