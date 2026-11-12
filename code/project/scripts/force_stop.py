#!/usr/bin/env python3
"""强制停止训练进程（SIGINT 已被训练循环吞掉时的兜底）。

只杀主进程 PID（train_full.py 主进程），SIGTERM 不行再用 SIGKILL，
然后在停稳后由 freeze_run.py --only-ckpt 完成 ckpt 自包含化改写。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# 训练进程停止后，ultralytics 会 strip_optimizer 把 last.pt 里的 optimizer 清空，
# 那样就无法续训了。停下后立即把 last.pt 备份成 resume.pt 永久保留。
PROJ = Path("/3250410011/root/rivermind-data/MutilModel_3485988898")


def main() -> int:
    name = sys.argv[1]
    out = subprocess.run(
        f"pgrep -f 'train_full.py.*--name {name}'", shell=True, capture_output=True, text=True
    ).stdout.split()
    pids = [int(x) for x in out] if out else []
    if not pids:
        print("[kill] 无匹配进程")
        return 0
    main_pid = min(pids)
    print(f"[kill] 匹配 {len(pids)} 个进程，主进程 {main_pid}")

    for sig in (signal.SIGTERM,):
        os.kill(main_pid, sig)
        print(f"[kill] 已发 {sig.name}，等待退出")
        for _ in range(40):
            if not os.path.exists(f"/proc/{main_pid}"):
                print("[kill] 主进程已退出")
                break
            time.sleep(3)
        else:
            state = open(f"/proc/{main_pid}/stat").read().split()[2]
            print(f"[kill] 仍存活（state={state}），升级 SIGKILL")
            os.kill(main_pid, signal.SIGKILL)
            for _ in range(20):
                if not os.path.exists(f"/proc/{main_pid}"):
                    print("[kill] SIGKILL 后退出")
                    break
                time.sleep(1)

    left = subprocess.run(
        f"pgrep -f 'train_full.py.*--name {name}'", shell=True, capture_output=True, text=True
    ).stdout.split()
    print(f"[kill] 残留进程: {left if left else '无'}")

    # 双保险：若最后一次保存带 optimizer，先另存一份，避免被 final_eval 的
    # strip_optimizer 清空（清空后无法续训，未知性会逼我们重训，代价极高）。
    import torch

    last = PROJ / "ResTest" / name / "weights" / "last.pt"
    resume = PROJ / "ResTest" / name / "weights" / "resume.pt"
    if last.exists():
        sys.path.insert(0, str(PROJ))
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
        if ck.get("optimizer") is not None:
            torch.save(ck, str(resume))
            print(f"[kill] 已备份可续训 ckpt: {resume} (epoch={ck.get('epoch')})")
        else:
            print("[kill] last.pt 无 optimizer（已被 strip），若存在 resume.pt 则以其为准")
    return 0


if __name__ == "__main__":
    sys.exit(main())
