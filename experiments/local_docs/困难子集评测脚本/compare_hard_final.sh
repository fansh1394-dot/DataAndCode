#!/bin/bash
# 同口径对比 A / B 两个 TAG 在困难子集上的结果（默认 N0 / S3）。
#
# 为什么要单独跑：eval_hard_subset.py 的 JSON 只落盘了全集的 precision/recall，
# 分组(only_ir/only_vis)的 precision/recall 只在日志表格里，必须回读日志才能补齐。
#
# 用法: compare_hard_final.sh [TAG_A] [TAG_B]
set -u
ROOT=/root/rivermind-data
L="${ROOT}/logs"
RES="${ROOT}/MutilModel_3485988898/ResTest"

A="${1:-N0}"
B="${2:-S3}"

# 尽量给出 "TAG(epN)" 这种能自解释的标签：轮数取 results.csv 的行数。
epochs_of() {
    tag="$1"
    csv="${RES}/v2_${tag}_s0/results.csv"
    if [ -f "$csv" ]; then
        echo $(( $(wc -l < "$csv") - 1 ))
    else
        echo "?"
    fi
}
EA=$(epochs_of "$A")
EB=$(epochs_of "$B")
LA="${A}(ep${EA})"
LB="${B}(ep${EB})"

newest() {
    ls -t "${L}"/eval_hard_"$1"_*.log 2>/dev/null | head -1
}

echo "########## 对比对象 ##########"
echo "  A = ${LA}   日志=$(newest "$A")"
echo "  B = ${LB}   日志=$(newest "$B")"
echo "  A json = ${L}/hard_eval_${A}.json"
echo "  B json = ${L}/hard_eval_${B}.json"

echo
echo "########## A) ${A} 日志里的三张表（原样） ##########"
ALOG=$(newest "$A")
echo "ALOG=${ALOG}"
tr '\r' '\n' < "$ALOG" 2>/dev/null | sed -n '/困难子集 · 全集/,/^报数口径提醒/p'

echo
echo "########## B) ${B} 日志里的三张表（原样） ##########"
BLOG=$(newest "$B")
echo "BLOG=${BLOG}"
tr '\r' '\n' < "$BLOG" 2>/dev/null | sed -n '/困难子集 · 全集/,/^报数口径提醒/p'

echo
echo "########## C) JSON mAP50-95 分支对比 ##########"
VARA="$A" VARB="$B" LABA="$LA" LABB="$LB" /opt/conda/bin/python - <<'PY'
import json, os

L = "/root/rivermind-data/logs"
A = os.environ["VARA"]; B = os.environ["VARB"]
LA = os.environ["LABA"]; LB = os.environ["LABB"]

def load(tag):
    p = f"{L}/hard_eval_{tag}.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

da, db = load(A), load(B)
print(f"{A} json:", "OK" if da else "MISSING", f" | {B} json:", "OK" if db else "MISSING")
if da is None and db is None:
    print("两边 JSON 都缺失，本节省略")
    raise SystemExit(0)

def row(d, scope, br):
    if d is None:
        return float("nan")
    if scope == "ALL":
        return d.get("branch_map50_95_ALL", {}).get(br, float("nan"))
    return d.get("branch_map50_95_by_group", {}).get(scope, {}).get(br, float("nan"))

SC = (("ALL", "全集(1036 帧)"),
      ("only_ir", "分组A 仅IR可见(967 帧)"),
      ("only_vis", "分组B 仅RGB可见(69 帧)"))

for scope, title in SC:
    print()
    print(f"=== {title} mAP50-95 ===")
    print(f"{'分支':<10}{LA:>14}{LB:>14}{'B-A':>12}")
    for br in ("rgb", "ir", "fuse", "fuse_ir"):
        a, b = row(da, scope, br), row(db, scope, br)
        fa = "  —" if a != a else f"{a:.5f}"
        fb = "  —" if b != b else f"{b:.5f}"
        dd = "  —" if (a != a or b != b) else f"{b-a:+.5f}"
        print(f"{br:<10}{fa:>14}{fb:>14}{dd:>12}")

print()
print("=== B 内部：fuse_ir 相对 ir 的增减（核心主张口径，仅 B 有 fuse_ir） ===")
if db:
    for scope, label in SC:
        fi, ir = row(db, scope, "fuse_ir"), row(db, scope, "ir")
        if fi != fi or ir != ir:
            print(f"  {label:<18} 缺 fuse_ir/ir")
            continue
        print(f"  {label:<18} fuse_ir={fi:.5f}  ir={ir:.5f}  Δ={fi-ir:+.5f}"
              f"  {'✔ 提升' if fi > ir else '✘ 未提升'}")
else:
    print(f"  {B} JSON 缺失，无法核对")
PY

echo
echo "########## D) 日志解析：分组 precision/recall 与核心主张 ##########"
VARA="$A" VARB="$B" LABA="$LA" LABB="$LB" /opt/conda/bin/python - <<'PY'
import re, os, glob

L = "/root/rivermind-data/logs"
A = os.environ["VARA"]; B = os.environ["VARB"]
LA = os.environ["LABA"]; LB = os.environ["LABB"]

BRANCHES = ("rgb", "ir", "fuse", "fuse_ir")

def parse(fn):
    """从日志里抽 {scope: {branch: {map,map50,prec,rec}}}。

    坑（第一版就是踩在这上面，整表全空）：日志实际排版是
        ======================（分隔线）
        困难子集 · 全集   （帧数 1036）      <- 标题独占一个 block
        ======================（分隔线）
        分支   mAP50-95  mAP50  precision  recall   <- 表头+数据在下一个 block
        ------
        rgb ...
    用 re.split(r"={20,}\\n") 切完之后，标题和数据行分属相邻两个 block。
    因此不能"找到标题就只解析那一个 block"，必须用 cur 记住当前 scope，
    并且对每个 block 都尝试解析数据行（表头 block 也要过一遍）。
    """
    if not fn or not os.path.exists(fn):
        return {}
    txt = open(fn, encoding="utf-8", errors="ignore").read().replace("\r", "\n")
    out, cur = {}, None
    for b in re.split(r"={20,}\n", txt):
        head = b.strip().split("\n")[0] if b.strip() else ""

        if head.startswith("困难子集 · 全集"):
            cur = "ALL"
        elif head.startswith("分组 A"):
            cur = "only_ir"
        elif head.startswith("分组 B"):
            cur = "only_vis"
        elif head.startswith(("分组方向性校验", "报数口径提醒")):
            cur = None          # 表格区结束，后面是校验/说明文字
            continue
        # 表头 block（"分支 ..." / "----"）不重置 cur：数据行就在它下面。

        if cur is None:
            continue

        for line in b.split("\n"):
            line = line.strip()
            for br in BRANCHES:
                # "fuse " 不能误吞 "fuse_ir "；^ 锚定避免匹配到行中间的 ir
                if line.startswith(br + " ") or re.match(rf"^{br}\s+\d", line):
                    p = line.split()
                    if len(p) >= 5 and p[0] == br:
                        try:
                            vals = [float(x) for x in p[1:5]]
                        except ValueError:
                            continue
                        out.setdefault(cur, {})[br] = {
                            "map": vals[0], "map50": vals[1],
                            "prec": vals[2], "rec": vals[3]}
    return out

def newest(tag):
    fs = sorted(glob.glob(f"{L}/eval_hard_{tag}_*.log"), key=os.path.getmtime)
    return fs[-1] if fs else None

da, db = parse(newest(A)), parse(newest(B))
print(f"{A} 日志:", "OK" if da else "MISSING", f" | {B} 日志:", "OK" if db else "MISSING")
if da:
    print(f"  {A} 解析到: " + ", ".join(f"{s}={sorted(v)}" for s, v in da.items()))
if db:
    print(f"  {B} 解析到: " + ", ".join(f"{s}={sorted(v)}" for s, v in db.items()))

print()
print(f"{'范围':<10}{'分支':<9}{'指标':<10}{LA:>14}{LB:>14}{'差值':>12}")
print("-" * 72)
SC = (("ALL", "全集"), ("only_ir", "A-仅IR"), ("only_vis", "B-仅RGB"))
for scope, label in SC:
    for br in BRANCHES:
        for mk, mn in (("map", "mAP50-95"), ("rec", "recall"), ("prec", "precision")):
            a = da.get(scope, {}).get(br, {}).get(mk)
            b = db.get(scope, {}).get(br, {}).get(mk)
            fa = "—" if a is None else f"{a:.5f}"
            fb = "—" if b is None else f"{b:.5f}"
            dd = "—" if (a is None or b is None) else f"{b-a:+.5f}"
            print(f"{label:<10}{br:<9}{mn:<10}{fa:>14}{fb:>14}{dd:>12}")
    print("-" * 72)

print()
print("=== 核心主张（困难场景召回率）逐帧核对 ===")
print(f"  注：fuse_ir 仅 {B} 存在；{A} 只有 rgb/ir/fuse。")
hits = []
for scope, label in SC:
    ri = db.get(scope, {}).get("ir", {}).get("rec")
    rf = db.get(scope, {}).get("fuse_ir", {}).get("rec")
    rfuse = db.get(scope, {}).get("fuse", {}).get("rec")
    n0_ir = da.get(scope, {}).get("ir", {}).get("rec")
    n0_fuse = da.get(scope, {}).get("fuse", {}).get("rec")
    if ri is None and n0_ir is None:
        print(f"  {label:<14} 无可用 recall 数据")
        continue
    bits = []
    if n0_ir is not None:
        bits.append(f"{LA} ir={n0_ir:.5f}")
    if ri is not None:
        bits.append(f"{LB} ir={ri:.5f}")
        bits.append(f"{LB} fuse={rfuse if rfuse is None else round(rfuse, 5)}")
    verdict = ""
    if ri is not None and rf is not None:
        verdict = f"  fuse_ir-ir={rf-ri:+.5f}  {'✔ 提升' if rf > ri else '✘ 未提升'}"
        hits.append((scope, label, rf - ri, n0_fuse, rfuse))
    print(f"  {label:<14} " + "  ".join(bits) + verdict)

print()
print("=== 核心判定汇总（供守护脚本抓取） ===")
if not hits:
    print("  核心判定: 数据不足，无法判定（检查 B 侧日志是否含 fuse_ir 行）")
for scope, label, d, n0f, bf in hits:
    dm = (bf - n0f) if (n0f is not None and bf is not None) else None
    dm_s = "—" if dm is None else f"{dm:+.5f}"
    ok = "✔" if d > 0 else "✘"
    print(f"核心判定[{scope:<8}] {ok} Δrecall(fuse_ir-ir)={d:+.5f}  ΔmAP(fuse, {B}-{A})={dm_s}"
          f"  融合是否可用={'是' if d > 0 else '否（需按预案调整）'}")
PY
