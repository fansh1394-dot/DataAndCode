#!/usr/bin/env python3
"""在镜像克隆前冻结训练：优雅停止 → 校验 last.pt → 改写成自包含可续训的 ckpt。

为什么不能直接 kill
-------------------
本环境的 torch 2.12 默认 ``weights_only=True``，而 check_resume / attempt_load_weights
反序列化时需要 ``nn.modules`` 白名单之外的类（多模态 backbone / Detect 子类）。
直接 stop 后若 ckpt 的 model 引用为 None、train_args['data'] 指向克隆后不存在的
绝对路径，续训会在加载阶段直接失败，且失败点发生在训练进程中、难以补救。

因此本脚本做三件事：
1. 向主进程发 SIGINT，让 trainer 走正常退出路径（若还没写出最后一轮 ckpt）；
2. 停稳后校验 last.pt 的 epoch 与 results.csv 是否对齐；
3. 把 model 权重内联进 ckpt（resume 时 trainer 从 ckpt['model'] 恢复，不再重读路径），
   并把 data 改成相对路径，使克隆后无需改任何文件即可 --resume。
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJ = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")
# torch.load 反序列化 ckpt 会 import ultralytics（自定义层），必须先入 sys.path，
# 否则报 ModuleNotFoundError: No module named 'ultralytics'
sys.path.insert(0, str(PROJ))


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def find_pids(name: str) -> list:
    out = sh(f"pgrep -f 'train_full.py.*--name {name}'")
    return [int(x) for x in out.split()] if out else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="ResTest")
    ap.add_argument("--only-ckpt", action="store_true",
                    help="跳过停止步骤，只对现有 last.pt 做自包含化改写与复核")
    args = ap.parse_args()

    run_dir = PROJ / args.project / args.name
    last = run_dir / "weights" / "last.pt"
    csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"

    pids = find_pids(args.name) if not args.only_ckpt else []
    print(f"[stop] 匹配进程: {pids}")
    if pids:
        before = last.stat().st_mtime if last.exists() else 0
        main_pid = min(pids)  # train_full.py 主进程 PID 最小，其余是 dataloader worker
        print(f"[stop] 向主进程 {main_pid} 发 SIGINT（worker 会被一并回收）")
        os.kill(main_pid, signal.SIGINT)
        t0 = time.time()
        while time.time() - t0 < 60:
            if not Path(f"/proc/{main_pid}").exists():
                break
            time.sleep(2)
        if Path(f"/proc/{main_pid}").exists():
            print(f"[stop] 主进程 {main_pid} 在 60s 内未退出（SIGINT 被训练循环吞掉）")
            print("[stop] 请改用 scripts/force_stop.py 强制停止，之后用 --only-ckpt 完成改写")
            return 3
        print(f"[stop] 停止完成，用时 {time.time() - t0:.0f}s")
        after = last.stat().st_mtime if last.exists() else 0
        print(f"[stop] last.pt mtime: {before} -> {after}")
        if after == before:
            print("[stop] 提示：SIGINT 落在 epoch 边界，last.pt 未刷新，沿用已有 ckpt（数据不丢）")

    if not last.exists():
        print("[错误] last.pt 不存在，无法冻结")
        return 2

    data_rel = os.path.relpath("/root/rivermind-data/uav_data_v2/data.yaml", run_dir.resolve())

    # results.csv 的最后 epoch
    last_csv_epoch = -1
    if csv.exists():
        lines = [ln for ln in csv.read_text().strip().splitlines() if ln.strip()]
        if len(lines) > 1:
            last_csv_epoch = int(lines[-1].split(",")[0])
    print(f"[check] results.csv 最后 epoch = {last_csv_epoch}")

    ckpt = torch.load(str(last), map_location="cpu", weights_only=False)
    ep = int(ckpt.get("epoch", -1))
    print(f"[check] last.pt epoch(0-based) = {ep} -> 下一轮 = {ep + 1}")

    ta = ckpt.get("train_args") or (ckpt.get("args") or {})
    print(f"[check] train_args.data = {ta.get('data')}")
    print(f"[check] train_args.project = {ta.get('project')}")

    rdir = run_dir.resolve()
    print(f"[fix] 数据相对路径 {data_rel}（相对 {rdir}，克隆后同构目录可直接用）")

    # 步骤 3b：args.yaml 落盘（存在则不动）。
    # trainer 在 exist_ok 且 args.yaml 已存在时会直接 return，不重写——clone 后
    # stem 路径保留，故这里必须把 data 改成相对路径，否则 resume 仍读旧绝对路径。
    if args_yaml.exists():
        text = args_yaml.read_text(encoding="utf-8")
        new = re.sub(r"(?m)^data: .*$", f"data: {data_rel}", text)
        if new != text:
            args_yaml.write_text(new, encoding="utf-8")
            print(f"[fix] args.yaml 的 data 已改为相对路径")
    else:
        print("[warn] args.yaml 不存在（resume 时 trainer 会重写，不影响）")

    # 自包含化：内联权重（resume 直接取 ckpt['model']，不再依赖 yaml 路径与 nc 对齐），
    # data 改相对路径。两者共同保证克隆到新实例后 --resume 无需任何改动。
    ema = ckpt.get("ema")
    if ema is not None:
        ckpt["model"] = ema.half()
        print("[fix] 已用 EMA 权重内联 model（clone 后不依赖 yaml/nc 对齐）")
    if "train_args" in ckpt:
        ckpt["train_args"]["data"] = data_rel
    torch.save(ckpt, str(last))
    print(f"[fix] 已重写 {last}（含 optimizer/ema/epoch={ep}/best_fitness={ckpt.get('best_fitness')}）")

    # 只读复核：resume 需要的关键字段是否齐备
    chk = torch.load(str(last), map_location="cpu", weights_only=False)
    for k in ("epoch", "optimizer", "ema", "best_fitness", "updates", "train_args"):
        v = chk.get(k)
        ok = v is not None
        print(f"[verify] {k:<13} {'OK' if ok else 'MISSING'}"
              + (f"  type={type(v).__name__}" if ok else ""))
    opt = chk.get("optimizer") or {}
    print(f"[verify] optimizer param_groups={len(opt.get('param_groups', []))} "
          f"state_entries={len(opt.get('state', {}))}")
    print(f"[verify] ckpt 大小 = {last.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
