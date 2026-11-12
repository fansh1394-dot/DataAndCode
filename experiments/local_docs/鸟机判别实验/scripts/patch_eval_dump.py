# -*- coding: utf-8 -*-
"""给 scripts/eval_split.py 增加 --dump-preds（只增量、可回滚）。

背景：D0/E27 需要「鸟目标被任何预测框命中」的提议召回，而验证器的
predictions.json 只在 is_coco/is_lvis 时落盘；这里直接把主分支
（fuse，RGB 坐标系）的原生尺度预测写成 jsonl，供离线匹配。
"""
from pathlib import Path

p = Path("/3250410011/root/rivermind-data/MutilModel_3485988898/scripts/eval_split.py")
s = p.read_text(encoding="utf-8")

if "--dump-preds" in s:
    print("already patched")
    raise SystemExit(0)

anchor_arg = "    args = ap.parse_args()"
arg_block = '''    ap.add_argument("--dump-preds", default=None,
                    help="把主分支（fuse）原生尺度预测写成 jsonl（im_file/bboxes/conf/cls），"
                         "用于提议召回等离线匹配。默认关闭。")
'''
assert s.count(anchor_arg) == 1
s = s.replace(anchor_arg, arg_block + anchor_arg)

anchor_call = "    ret = validator(model=net)"
patch = '''    if args.dump_preds:
        import types

        dump_path = Path(args.dump_preds)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("", encoding="utf-8")

        def _dump_pred_to_json(self, predn, filename):
            rec = {
                "im_file": str(filename),
                "bboxes": predn["bboxes"].detach().cpu().tolist(),
                "conf": predn["conf"].detach().cpu().tolist(),
                "cls": predn["cls"].detach().cpu().tolist(),
            }
            with dump_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\\n")

        validator.pred_to_json = types.MethodType(_dump_pred_to_json, validator)
        print(f"[eval_split] 预测转储 -> {dump_path}")

'''
assert s.count(anchor_call) == 1
s = s.replace(anchor_call, patch + anchor_call)

p.with_suffix(".py.bak_dump").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
p.write_text(s, encoding="utf-8")
print("patched", p)
