#!/bin/bash
# 自测：watch 脚本里新增的 json_ok 校验逻辑（抽取同一段 Python）能否正确拒绝坏 JSON。
# 只操作 /tmp，不动真实产物。
set -u
T=/tmp/json_ok_test
rm -rf "$T"; mkdir -p "$T"

# 校验器：与 watch_s3_final_eval.sh 内嵌的判据保持一致
cat > "$T/check.py" <<'PY'
import json, os, sys
p = os.environ["PJ"]
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception:
    sys.exit(1)
if not isinstance(d, dict):
    sys.exit(1)
if "branch_map50_95_ALL" not in d or "fuse" not in d.get("branch_map50_95_ALL", {}):
    sys.exit(1)
if d.get("n_samples", 0) < 1000:
    sys.exit(1)
sys.exit(0)
PY

run_case() {
    PJ="$1" /opt/conda/bin/python "$T/check.py" >/dev/null 2>&1
    rc=$?
    echo "$2  ->  rc=$rc  $([ $rc -eq 0 ] && echo '接受' || echo '拒绝')"
}

# 1) 正常文件：应接受
cat > "$T/good.json" <<'J'
{"n_samples": 1036,
 "branch_map50_95_ALL": {"rgb": 0.0284, "ir": 0.3514, "fuse": 0.0287, "fuse_ir": 0.3507},
 "branch_map50_95_by_group": {"only_ir": {}, "only_vis": {}}}
J
run_case "$T/good.json" "正常(n=1036,有fuse)        "

# 2) 半截文件：写一半被切断，应拒绝
head -c 60 "$T/good.json" > "$T/trunc.json"
run_case "$T/trunc.json" "半截(截断的 JSON)          "

# 3) 缺 fuse 分支：应拒绝
cat > "$T/nofuse.json" <<'J'
{"n_samples": 1036, "branch_map50_95_ALL": {"rgb": 0.02, "ir": 0.35}}
J
run_case "$T/nofuse.json" "缺 fuse 分支               "

# 4) 样本数不足：应拒绝
cat > "$T/few.json" <<'J'
{"n_samples": 12, "branch_map50_95_ALL": {"fuse": 0.02}}
J
run_case "$T/few.json" "n_samples 仅 12            "

# 5) 空文件：应拒绝
: > "$T/empty.json"
run_case "$T/empty.json" "空文件                     "

# 6) JSON 合法但顶层是数组：应拒绝
echo '[1,2,3]' > "$T/arr.json"
run_case "$T/arr.json" "顶层是数组                 "

rm -rf "$T"
echo "SMOKE_DONE"
