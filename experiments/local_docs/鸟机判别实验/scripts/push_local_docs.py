# -*- coding: utf-8 -*-
"""打包本地文档并通过 SFTP 上传到服务器，解压到 upload_repo/experiments/local_docs。"""
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(r"c:\Users\不如喝茶去\Desktop\双光谱毕设\scripts")))
from _ssh_new import connect  # noqa: E402

ROOT = Path(r"c:\Users\不如喝茶去\Desktop\双光谱毕设")
TAR = Path(r"c:\Users\不如喝茶去\AppData\Local\Temp\local_docs.tar.gz")
REMOTE = "/root/rivermind-fs/birddrone/local_docs.tar.gz"

DIRS = [
    "鸟机判别实验", "项目交接文档", "方案设计Word文档", "Markdown文档",
    "配准工作归档", "对齐性核查", "困难子集评测脚本", "基础知识", "子方向选题",
]
FILES = [
    "待做实验清单.xlsx", "待做实验清单_备份_20261110.xlsx", "实验总览看板.html",
    "2026-10-31_项目交接文档.md",
    "2026-11-01_M17交接文档_真实视频流与双光谱对齐.md",
    "2026-11-01_RK3588部署验证与M17状态报告.md",
    "2026-11-04_RK3588跟踪告警专项测试与根因报告.md",
    "2026-11-04_相机标定与真实视频流接入技术指导手册.docx",
    "双光谱反无人机小论文选题可行性报告_20261111.md",
    "双光谱反无人机小论文选题可行性报告_20261111.docx",
    "双光谱反无人机检测_小论文选题与实施规划.pptx",
    "低小慢无人机双光谱空域视觉监视_开题报告.pptx",
    "低小慢无人机双光谱空域视觉监视_开题论证.pptx",
    "开题论证_图文充实版.pptx",
    "开题论证_重构版.pptx",
    "面向边缘部署的双光谱低慢小无人机检测方法研究与系统实现.pptx",
]


def main():
    items = []
    for d in DIRS:
        p = ROOT / d
        if p.exists():
            items.append(p)
        else:
            print("[skip] 目录不存在:", d)
    for f in FILES:
        p = ROOT / f
        if p.exists():
            items.append(p)
        else:
            print("[skip] 文件不存在:", f)

    with tarfile.open(TAR, "w:gz") as tf:
        for p in items:
            tf.add(str(p), arcname=p.name)
    print(f"[tar] {TAR} {TAR.stat().st_size/1048576:.1f} MB, {len(items)} 项")

    c = connect()
    sf = c.open_sftp()
    sf.put(str(TAR), REMOTE)
    sf.close()
    cmd = ("mkdir -p /root/rivermind-fs/birddrone/upload_repo/experiments/local_docs && "
           f"tar -xzf {REMOTE} -C /root/rivermind-fs/birddrone/upload_repo/experiments/local_docs && "
           "du -sh /root/rivermind-fs/birddrone/upload_repo/experiments/local_docs && ls "
           "/root/rivermind-fs/birddrone/upload_repo/experiments/local_docs")
    _, o, e = c.exec_command(cmd, timeout=600)
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("--- STDERR ---\n", err[-1000:])
    c.close()
    TAR.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
