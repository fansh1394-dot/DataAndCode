# -*- coding: utf-8 -*-
"""统计无 GT 帧上的误检：FP/frame、FPR@0.25。"""
import json
import os
import sys

PROJ = "/3250410011/root/rivermind-data/MutilModel_3485988898"
GT_DIR = "/3250410011/root/rivermind-data/mendeley_split/labels/test"


def has_gt(fname):
    p = os.path.join(GT_DIR, os.path.splitext(fname)[0] + ".txt")
    if not os.path.isfile(p):
        return None  # 无标签文件
    with open(p, encoding="utf-8") as f:
        for line in f:
            if len(line.split()) >= 5:
                return True
    return False


def stat(tag, thr=0.25):
    path = os.path.join(PROJ, "perframe_boxes2", "boxes_%s.json" % tag)
    data = json.load(open(path, encoding="utf-8"))
    n_all = len(data)
    no_gt = []
    missing = 0
    for fname in data:
        g = has_gt(fname)
        if g is None:
            missing += 1
        elif not g:
            no_gt.append(fname)
    fp_dets = 0
    fp_frames = 0
    for fname in no_gt:
        ent = data[fname].get("fuse") or {}
        c = ent.get("conf", [])
        k = sum(1 for v in c if v >= thr)
        fp_dets += k
        if k > 0:
            fp_frames += 1
    print("[%s] frames=%d missing_label=%d no_gt=%d" % (tag, n_all, missing, len(no_gt)))
    if no_gt:
        print("    FP/frame@%.2f = %.4f  FPR@%.2f = %.4f (%d/%d)" % (
            thr, fp_dets / len(no_gt), thr, fp_frames / len(no_gt), fp_frames, len(no_gt)))
    else:
        print("    无无 GT 帧")


for t in sys.argv[1:]:
    stat(t)
