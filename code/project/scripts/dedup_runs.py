#!/usr/bin/env python3
"""列出/清理重复的 train_full 主进程，只保留指定 PID。

背景：chain 脚本、手动重启、DDP 重生都可能留下多个同名的根进程，
同时写同一个 ResTest/<name> 会导致 results.csv 交错、ckpt 互相覆盖。
本脚本按「根进程（父进程不是另一个 train_full）」识别，保留 --keep 指定的 PID。
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


def roots(match: str) -> list:
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            if "train_full.py" not in cmd or match not in cmd:
                continue
            ppid = int([l for l in open(f"/proc/{p}/status") if l.startswith("PPid")][0].split()[1])
            out.append({"pid": int(p), "ppid": ppid, "start": os.stat(f"/proc/{p}").st_mtime, "cmd": cmd})
        except (FileNotFoundError, ProcessLookupError):
            pass
    pids = {r["pid"] for r in out}
    return [r for r in out if r["ppid"] not in pids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True, help="命令行里需包含的子串，如 v2c_E1_rgbonly_s2")
    ap.add_argument("--keep", type=int, default=None, help="保留的根进程 PID")
    args = ap.parse_args()

    rs = roots(args.match)
    for r in sorted(rs, key=lambda x: x["start"]):
        print(f"root pid={r['pid']} ppid={r['ppid']} start={time.strftime('%H:%M:%S', time.localtime(r['start']))}")

    if args.keep is None:
        return 0

    targets = [r["pid"] for r in rs if r["pid"] != args.keep]
    if not targets:
        print("[dedup] 只有一个根进程，无需处理")
        return 0
    print(f"[dedup] 保留 {args.keep}，终止 {targets}")
    for pid in targets:
        try:
            os.kill(pid, signal.SIGKILL)  # 多余进程不需要优雅退出，直接杀以免争抢写盘
            print(f"[dedup] SIGKILL {pid}")
        except ProcessLookupError:
            print(f"[dedup] {pid} 已不存在")

    time.sleep(5)
    left = roots(args.match)
    print(f"[dedup] 处理后根进程: {[r['pid'] for r in left]}")
    print("[dedup] 残留 worker:", subprocess.run(
        f"pgrep -fc 'train_full.py.*{args.match}'", shell=True, capture_output=True, text=True).stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
