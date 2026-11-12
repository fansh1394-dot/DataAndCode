# -*- coding: utf-8 -*-
"""双光谱综合基准数据集建库脚本（v2：使用解压后的真实路径）。

相对 v1 的变更：
  1. UAV-CB 改用解压后的真实文件路径（不再 zip::internal）
  2. Mendeley 尺寸改为实际读取（不再硬编码 586x503）
  3. Dataset.zip 一并解压，负样本池用真实路径
  4. 每个数据集输出 YOLO 格式的 labels / labels_ir 软链接视图（便于对接现有训练框架）

用法：
  python build_dualspec_bench.py --stage all --clean
  python build_dualspec_bench.py --stage index
  python build_dualspec_bench.py --stage link
"""
import argparse
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path('/3250410011/root/rivermind-data/dualspec_bench')
FS = Path('/3250410011/root/rivermind-fs')

SRC_A = Path('/root/rivermind-data/uav_data_v2')
SRC_B = FS / 'UAV-CB'
SRC_C = FS / 'mendeley_eo_ir/Dataset_Mendeley_data/Labelled_UAV_Images/Labelled EO_IR Dataset'
SRC_D = FS / 'Dataset'
SRC_D_ZIP = FS / 'Dataset.zip'

SCENES_B = ['building', 'cloud', 'ground', 'powerline', 'vegetation']


def img_size(p):
    """读取图像尺寸，失败返回 None。"""
    try:
        import cv2
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        return [int(im.shape[1]), int(im.shape[0])]
    except Exception:
        return None


def read_yolo(p):
    if not p.exists():
        return None
    out = []
    for line in p.read_text(errors='ignore').splitlines():
        s = line.split()
        if len(s) >= 5:
            try:
                out.append([int(float(s[0])), float(s[1]), float(s[2]),
                            float(s[3]), float(s[4])])
            except ValueError:
                continue
    return out


def labelme_boxes(js):
    w0, h0 = js.get('imageWidth', 0), js.get('imageHeight', 0)
    if not w0 or not h0:
        return []
    out = []
    for s in js.get('shapes', []):
        pts = s.get('points', [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        out.append([0, (x1 + x2) / 2 / w0, (y1 + y2) / 2 / h0,
                    (x2 - x1) / w0, (y2 - y1) / h0])
    return out


def ensure_unzipped(zp, dest):
    """若无解压目录则解压。"""
    if dest.exists():
        return True
    if not zp.exists():
        return False
    print(f'  解压 {zp.name} -> {dest} ...')
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(dest.parent)
    return True


# ---------------- A ----------------
def index_anti_uav300(recs):
    n = 0
    for split in ('train', 'val', 'test'):
        meta_p = SRC_A / 'metadata' / f'{split}.json'
        meta = json.loads(meta_p.read_text())['frames'] if meta_p.exists() else {}
        img_dir = SRC_A / 'images' / split
        img_ir_dir = SRC_A / 'images_ir' / split
        lab_dir = SRC_A / 'labels' / split
        lab_ir_dir = SRC_A / 'labels_ir' / split
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob('*.jpg')):
            stem = p.stem
            pi = img_ir_dir / f'{stem}.jpg'
            m = meta.get(stem, {})
            rb = read_yolo(lab_dir / f'{stem}.txt')
            ib = read_yolo(lab_ir_dir / f'{stem}.txt')
            tail = stem.rsplit('_', 1)[-1]
            recs.append({
                'id': f'anti_uav300/{split}/{stem}',
                'source': 'anti_uav300',
                'split': split,
                'scene': m.get('sequence', stem.rsplit('_', 1)[0]),
                'frame_index': m.get('frame_index', int(tail) if tail.isdigit() else -1),
                'kind': m.get('kind', 'unknown'),
                'rgb_path': str(p),
                'ir_path': str(pi) if pi.exists() else None,
                'rgb_size': [640, 512],
                'ir_size': [640, 512],
                'rgb_boxes': rb or [],
                'ir_boxes': ib or [],
                'rgb_label_missing': rb is None,
                'ir_label_missing': ib is None,
                'aligned': False,
                'has_ir': pi.exists(),
            })
            n += 1
    print(f'[A] anti_uav300: {n} 帧')


# ---------------- B ----------------
def index_uav_cb(recs):
    if not SRC_B.exists():
        print('[B] 解压目录不存在，跳过')
        return
    n = 0
    for sc in SCENES_B:
        vdir = SRC_B / sc / 'visible'
        tdir = SRC_B / sc / 'thermal'
        if not vdir.exists():
            continue
        for jp in sorted(vdir.glob('*.json')):
            stem_v = jp.stem
            tp = tdir / f'{stem_v.replace("_visible_", "_thermal_")}.json'
            vimg = jp.with_suffix('.jpg')
            timg = tdir / f'{stem_v.replace("_visible_", "_thermal_")}.jpg'
            if not (tp.exists() and vimg.exists() and timg.exists()):
                continue
            try:
                dv = json.loads(jp.read_text(errors='ignore'))
                dt = json.loads(tp.read_text(errors='ignore'))
            except Exception:
                continue
            idx = stem_v.rsplit('_', 1)[-1]
            seq = '_'.join(stem_v.split('_')[:-2]) if stem_v.count('_') >= 2 else sc
            recs.append({
                'id': f'uav_cb/{sc}/{stem_v}',
                'source': 'uav_cb',
                'split': 'unsplit',
                'scene': f'{sc}/{seq}',
                'category': sc,
                'frame_index': int(idx) if idx.isdigit() else -1,
                'kind': 'both',
                'rgb_path': str(vimg),
                'ir_path': str(timg),
                'rgb_size': [dv.get('imageWidth'), dv.get('imageHeight')],
                'ir_size': [dt.get('imageWidth'), dt.get('imageHeight')],
                'rgb_boxes': labelme_boxes(dv),
                'ir_boxes': labelme_boxes(dt),
                'rgb_label_missing': False,
                'ir_label_missing': False,
                'aligned': False,
                'has_ir': True,
            })
            n += 1
    print(f'[B] uav_cb: {n} 对')


# ---------------- C ----------------
def index_mendeley(recs):
    n = 0
    size_cache = {}
    for split in ('train', 'val', 'test'):
        eo_img = SRC_C / 'UAV_eo_Dataset' / 'images' / split
        ir_img = SRC_C / 'UAV_ir_Dataset' / 'images' / split
        eo_lab = SRC_C / 'UAV_eo_Dataset' / 'labels' / split
        ir_lab = SRC_C / 'UAV_ir_Dataset' / 'labels' / split
        if not eo_img.exists():
            continue
        eo = {p.stem: p for p in eo_img.iterdir()
              if p.suffix.lower() in ('.png', '.jpg', '.jpeg')}
        ir = {p.stem: p for p in ir_img.iterdir()
              if p.suffix.lower() in ('.png', '.jpg', '.jpeg')} if ir_img.exists() else {}
        for stem in sorted(set(eo) & set(ir)):
            rb = read_yolo(eo_lab / f'{stem}.txt')
            ib = read_yolo(ir_lab / f'{stem}.txt')
            # 实际读取尺寸（每个 split 只读首张，避免过慢）
            if split not in size_cache:
                size_cache[split] = img_size(eo[stem]) or [586, 503]
            sz = size_cache[split]
            recs.append({
                'id': f'mendeley/{split}/{stem}',
                'source': 'mendeley',
                'split': split,
                'scene': split,
                'frame_index': int(stem) if stem.isdigit() else -1,
                'kind': 'both',
                'rgb_path': str(eo[stem]),
                'ir_path': str(ir[stem]),
                'rgb_size': sz,
                'ir_size': sz,
                'rgb_boxes': rb or [],
                'ir_boxes': ib or [],
                'rgb_label_missing': rb is None,
                'ir_label_missing': ib is None,
                'aligned': True,
                'has_ir': True,
            })
            n += 1
    print(f'[C] mendeley: {n} 对  尺寸={size_cache}')


# ---------------- D ----------------
def index_drone_bird(recs):
    if not SRC_D.exists():
        if not ensure_unzipped(SRC_D_ZIP, SRC_D):
            print('[D] Dataset.zip 不存在，跳过')
            return
    n = 0
    for sp_map, sp in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
        d = SRC_D / sp_map / 'images'
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            fn = p.name
            pref = 'BT' if fn.startswith('BT') else ('DT' if fn.startswith('DT') else '??')
            lab = SRC_D / sp_map / 'labels' / f'{p.stem}.txt'
            rb = read_yolo(lab)
            recs.append({
                'id': f'drone_bird/{sp}/{p.stem}',
                'source': 'drone_bird',
                'split': sp,
                'scene': sp,
                'frame_index': -1,
                'kind': 'single',
                'rgb_path': str(p),
                'ir_path': None,
                'rgb_size': [640, 640],
                'ir_size': None,
                'rgb_boxes': rb or [],
                'ir_boxes': [],
                'rgb_label_missing': rb is None,
                'ir_label_missing': True,
                'aligned': None,
                'role': 'negative_pool',
                'prefix': pref,
                'has_ir': False,
            })
            n += 1
    print(f'[D] drone_bird: {n} 张（BT={sum(1 for r in recs if r["source"]=="drone_bird" and r["prefix"]=="BT")}）')


def write_manifest(recs):
    OUT.mkdir(parents=True, exist_ok=True)
    mp = OUT / 'manifest.jsonl'
    with mp.open('w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n[MANIFEST] {mp}  {len(recs)} 条  {mp.stat().st_size/1e6:.1f} MB')


def write_stats(recs):
    by_src = Counter(r['source'] for r in recs)
    tgt = defaultdict(int)
    kinds = Counter()
    for r in recs:
        tgt[(r['source'], 'rgb')] += len(r['rgb_boxes'])
        tgt[(r['source'], 'ir')] += len(r['ir_boxes'])
        if r['source'] != 'drone_bird':
            kinds[(r['source'], r['kind'])] += 1

    # 配准状态：每源抽样算 RGB/IR 框 IoU（用归一化坐标）
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2
        bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0.0

    iou_by_src = defaultdict(list)
    for r in recs:
        if r['source'] == 'drone_bird':
            continue
        if len(r['rgb_boxes']) == 1 and len(r['ir_boxes']) == 1:
            iou_by_src[r['source']].append(_iou(r['rgb_boxes'][0], r['ir_boxes'][0]))

    import statistics
    stats = {
        'total_records': len(recs),
        'by_source': dict(by_src),
        'n_boxes': {f'{k[0]}_{k[1]}': v for k, v in tgt.items()},
        'kind_dist': {f'{k[0]}/{k[1]}': v for k, v in kinds.items()},
        'single_target_iou': {
            s: {'n': len(v), 'median': round(statistics.median(v), 4),
                'mean': round(statistics.mean(v), 4)} for s, v in iou_by_src.items()
        },
    }
    meta = OUT / 'meta'
    meta.mkdir(exist_ok=True)
    (meta / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[STATS] {meta / "stats.json"}')

    print('\n=== 统计 ===')
    for s, c in sorted(by_src.items()):
        print(f'  {s:16s} {c:7d} 条')
    print('\n  目标框数：')
    for (s, m), v in sorted(tgt.items()):
        print(f'    {s:16s} {m:3s} {v:7d}')
    print('\n  单目标帧的跨模态 IoU：')
    for s, d in stats['single_target_iou'].items():
        print(f'    {s:16s} n={d["n"]:6d} 中位={d["median"]:.4f} 均值={d["mean"]:.4f}')


def write_splits(recs):
    seq_by_src = defaultdict(lambda: defaultdict(set))
    for r in recs:
        if r['source'] in ('anti_uav300', 'mendeley'):
            seq_by_src[r['source']][r['split']].add(r['scene'])
        elif r['source'] == 'uav_cb':
            seq_by_src['uav_cb']['unsplit'].add(r['category'])
    splits = {s: {k: sorted(v) for k, v in d.items()} for s, d in seq_by_src.items()}
    meta = OUT / 'meta'
    meta.mkdir(exist_ok=True)
    (meta / 'splits.json').write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding='utf-8')
    # 泄漏检查
    allseq = {}
    for s, d in seq_by_src.items():
        seen = {}
        for sp, seqs in d.items():
            for q in seqs:
                if q in seen and seen[q] != sp:
                    print(f'  [警告] {s} 序列 {q} 同时出现在 {seen[q]} 与 {sp}')
                seen[q] = sp
        allseq[s] = len(seen)
    print(f'[SPLITS] 序列数: {allseq}（无警告即无跨 split 泄漏）')


def make_links(recs):
    """按 YOLO 目录约定建立软链接视图。"""
    linked = 0
    layouts = {
        'anti_uav300': OUT / 'misaligned' / 'anti_uav300',
        'uav_cb': OUT / 'misaligned' / 'uav_cb',
        'mendeley': OUT / 'aligned' / 'mendeley',
        'drone_bird': OUT / 'negatives' / 'drone_bird',
    }
    for r in recs:
        base = layouts.get(r['source'])
        if base is None:
            continue
        sp = r['split'] if r['source'] != 'uav_cb' else (r.get('category') or 'all')
        # RGB
        for key, sub in (('rgb_path', 'images'), ('ir_path', 'images_ir')):
            p = r.get(key)
            if not p:
                continue
            d = base / sp / sub
            d.mkdir(parents=True, exist_ok=True)
            dst = d / (Path(r['id']).name + Path(p).suffix)
            if not dst.exists():
                try:
                    os.symlink(p, dst)
                    linked += 1
                except OSError:
                    pass
        # 标签：源已在 YOLO 格式时直接链原 txt；LabelMe 源需写转换后的 txt
        if r['source'] in ('anti_uav300', 'mendeley'):
            lab_src = Path(r['rgb_path']).parent.parent / 'labels' / (Path(r['rgb_path']).stem + '.txt')
            lab_ir_src = Path(r['ir_path']).parent.parent / 'labels' / (Path(r['ir_path']).stem + '.txt') if r.get('ir_path') else None
            for src_lab, sub in ((lab_src, 'labels'), (lab_ir_src, 'labels_ir')):
                if src_lab is None or not src_lab.exists():
                    continue
                d = base / sp / sub
                d.mkdir(parents=True, exist_ok=True)
                dst = d / (Path(r['id']).name + '.txt')
                if not dst.exists():
                    try:
                        os.symlink(src_lab.resolve(), dst)
                        linked += 1
                    except OSError:
                        pass
        else:
            # uav_cb / drone_bird：写转换后的 txt
            for boxes, sub in ((r['rgb_boxes'], 'labels'), (r['ir_boxes'], 'labels_ir')):
                if not boxes:
                    continue
                d = base / sp / sub
                d.mkdir(parents=True, exist_ok=True)
                dst = d / (Path(r['id']).name + '.txt')
                if not dst.exists():
                    dst.write_text('\n'.join(
                        f'{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}' for b in boxes), encoding='utf-8')
                    linked += 1
    print(f'[LINK] 建立 {linked} 项')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['all', 'index', 'link', 'stats'])
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
        print(f'[CLEAN] 已删除 {OUT}')

    if args.stage in ('all', 'index'):
        recs = []
        index_anti_uav300(recs)
        index_uav_cb(recs)
        index_mendeley(recs)
        index_drone_bird(recs)
        write_manifest(recs)
        write_splits(recs)
        write_stats(recs)
        if args.stage == 'all':
            make_links(recs)
    elif args.stage == 'link':
        recs = [json.loads(l) for l in (OUT / 'manifest.jsonl').open(encoding='utf-8')]
        make_links(recs)
    elif args.stage == 'stats':
        recs = [json.loads(l) for l in (OUT / 'manifest.jsonl').open(encoding='utf-8')]
        write_stats(recs)

    print('\n完成。输出目录:', OUT)


if __name__ == '__main__':
    main()
