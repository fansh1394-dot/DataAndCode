import hashlib
import json
import random
from pathlib import Path

AVID = Path("/root/rivermind-fs/AVID")
MEN = Path("/3250410011/root/rivermind-data/mendeley_split")
OUT = Path("/root/rivermind-fs/birddrone")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261110

# ---------- AVID video stats ----------
def video_stats(split):
    res = {}
    base = AVID / "labels" / split
    for vdir in sorted(p for p in base.iterdir() if p.is_dir()):
        bird = drone = frames = 0
        for lp in vdir.glob("*.txt"):
            frames += 1
            for ln in lp.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                c = int(ln.split()[0])
                if c == 0:
                    bird += 1
                elif c == 1:
                    drone += 1
        cls = "bird" if bird and not drone else "drone" if drone and not bird else "mixed"
        res[vdir.name] = {"video": vdir.name, "class": cls, "frames": frames, "bird": bird, "drone": drone}
    return res

train_stats = video_stats("train")
val_stats = video_stats("val")

# 官方 train 内按视频 80/20 分层
rng = random.Random(SEED)
internal_train, internal_val = [], []
for cls in ("bird", "drone", "mixed"):
    group = [v for v in train_stats.values() if v["class"] == cls]
    rng.shuffle(group)
    n_val = max(1, round(len(group) * 0.2)) if group else 0
    internal_val += group[:n_val]
    internal_train += group[n_val:]

def sort_key(v):
    return v["video"]

internal_train.sort(key=sort_key)
internal_val.sort(key=sort_key)
official_val = [val_stats[k] for k in sorted(val_stats)]

def agg(vids):
    return {
        "videos": len(vids),
        "frames": sum(v["frames"] for v in vids),
        "bird": sum(v["bird"] for v in vids),
        "drone": sum(v["drone"] for v in vids),
        "classes": {c: sum(1 for v in vids if v["class"] == c) for c in ("bird", "drone", "mixed")},
    }

split = {
    "version": 1,
    "created": "2026-11-10",
    "seed": SEED,
    "unit": "video",
    "policy": {
        "avid": "官方 Train 内按视频 80/20 分层生成 train/val；官方 Val 只作外部 test，不参与训练与调参",
        "forbid": "禁止按帧随机划分；同一视频、同一拍摄日不得跨集",
        "mendeley": "沿用项目既有 train/val/test；test 只作 UAV 护栏",
    },
    "sources": {
        "avid": {
            "root": str(AVID),
            "modality": ["rgb", "ir"],
            "alignment": "weak",
            "label_space": {"0": "bird", "1": "drone"},
            "label_files": "每帧一套 YOLO 标签，RGB 坐标；无独立 IR 框",
            "splits": {
                "train": internal_train,
                "val": internal_val,
                "test": official_val,
            },
            "counts": {
                "train": agg(internal_train),
                "val": agg(internal_val),
                "test": agg(official_val),
                "total": agg(internal_train + internal_val + official_val),
            },
        },
        "mendeley": {
            "root": str(MEN),
            "modality": ["rgb", "ir"],
            "alignment": "aligned",
            "label_space": {"0": "UAV -> remap to 1 (drone)"},
            "counts": json.loads((MEN / "split_stats.json").read_text(encoding="utf-8")),
            "note": "labels_ir 与 labels 逐框相同；与 AVID 的弱对齐性质不同，混训前 AVID IR 必须离线配准",
        },
    },
    "training_plan": {
        "nc": 2,
        "names": {"0": "bird", "1": "drone"},
        "train": "avid.train + mendeley.train",
        "val": "avid.val + mendeley.val",
        "test_guardrail": "mendeley.test",
        "test_external": "avid.test (official val)",
    },
}

(OUT / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
(AVID / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- checksums ----------
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest_sha = sha256(Path("/root/rivermind-fs/AVID_MANIFEST.tsv"))
men_yaml_sha = sha256(MEN / "data.yaml")
men_stats_sha = sha256(MEN / "split_stats.json")

md = f"""# 鸟机判别数据源登记（SOURCES）

更新时间：2026-11-10

原则：每个来源登记名称、URL、许可、引用、下载日期、校验和与**对齐性质**；许可不明的一律不进训练，只进内部压力测试并在论文标注。

## 1. AVID（主训练源）

| 项 | 内容 |
|---|---|
| 角色 | bird + drone 主训练源；小目标/弱对齐压力测试 |
| 本地路径 | `{AVID}` |
| 规模 | 32 040 对帧、115 视频、35 952 框（bird 15 471 / drone 20 481） |
| 原生分辨率 | 320×256 |
| 划分 | 官方 Train 102 视频 / Val 13 视频；本项目在官方 Train 内按视频 80/20 分层生成 train/val，官方 Val **只作外部 test** |
| 标签 | YOLO `class cx cy w h`，`0=bird`、`1=drone`；每帧一套 RGB 坐标标签，**无独立 IR 框** |
| 对齐 | **弱对齐**：论文 32.3% 的配对全局位移 >5 px，16.3% 超过一个中位目标边长；对应帧不等于像素配准 |
| 本项目处理 | E28 前对 IR 做离线相位相关配准，逐帧记录 shift；未配准版本只用于对齐消融与压力测试 |
| 许可 | CC BY-NC 4.0（源影像 CC0）；**仅非商业学术使用** |
| 引用 | Raghavan, Rao, Selvi. *AVID: A Weakly Aligned RGB-Infrared Dataset and Perturbation Benchmark for Tiny Aerial Targets.* arXiv:2610.40949；Svanström, Alonso-Fernandez, Englund. *Drone Detection Dataset* |
| 校验和 | `AVID_MANIFEST.tsv` SHA-256 `{manifest_sha}` |
| 已知限制 | 0 鸟机同框、0 纯背景帧；39 段纯鸟 + 76 段纯机；68.1% 目标边长 8–16 px |

## 2. Mendeley EO/IR（UAV 护栏与背景）

| 项 | 内容 |
|---|---|
| 角色 | drone 正样本、背景帧、UAV 精度护栏、≥16 px 验收 |
| 本地路径 | `{MEN}` |
| 规模 | train 4 604（背景 629）/ val 616（背景 88）/ test 614（背景 66） |
| 对齐 | **像素对齐**；`labels_ir` 与 `labels` 逐框相同，两模态共用一套标签 |
| 许可 | 项目已确认可用于本研究；具体条款以 Mendeley Data 页面为准，论文提交前复核 |
| 校验和 | `data.yaml` SHA-256 `{men_yaml_sha}`；`split_stats.json` SHA-256 `{men_stats_sha}` |
| 已知限制 | 783 帧背景；99.3% 单目标帧；与 AVID 的对齐性质不同 |

## 3. 待接入来源

| 来源 | 角色 | 状态 | 许可 |
|---|---|---|---|
| Drone-vs-Bird Challenge（DVB 2019/2021） | RGB-only 补充、跨域 | 未下载 | 研究用途，逐条核 |
| Anti-UAV 系列 | 多目标/复杂场景补充 | 未下载 | 未核 |
| 自采（校园/公园） | 同框、背景、真实复杂场景、域内测试 | 待采集 | 自建，需拍摄记录与伦理说明 |

## 4. 混训与对齐规则（硬约束）

1. **只允许已对齐数据进入 6 通道检测训练**。AVID 的 IR 必须先离线配准到 RGB，才能与 Mendeley 混训；否则标签只对 RGB 成立。
2. 配准只作用于训练数据，**不改变部署链路**；部署端按相机自身标定处理。
3. 每个样本保留 `source` 标记；论文指标分源报告，禁止只报混合总数。
4. AVID 官方 Val 只能跑一次最终测试，**不得用于调参/选模**。
5. AVID 为 CC BY-NC 4.0，不得用于商业用途；论文中须同时引用 AVID 与原始 Drone Detection Dataset。
6. 任何新来源接入前，先补全本文件的许可、引用、校验和与对齐性质，否则不得进入训练。
"""
(OUT / "SOURCES.md").write_text(md, encoding="utf-8")

print("SPLIT", json.dumps({
    "train": split["sources"]["avid"]["counts"]["train"],
    "val": split["sources"]["avid"]["counts"]["val"],
    "test": split["sources"]["avid"]["counts"]["test"],
}, ensure_ascii=False))
print("OUT", OUT / "split.json", OUT / "SOURCES.md")
