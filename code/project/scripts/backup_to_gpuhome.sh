#!/bin/bash
# =============================================================================
# 源服务器 → 智川云 全量备份（独立目录，不与目标端已有数据混淆）
#
# 布局（目标端）：
#   /root/rivermind-data/rivermind-data_20261022/   <- 本次快照根
#       rivermind-data/   (源 /3250410011/root/rivermind-data 的内容)
#       _manifest/        校验清单与传输日志
#   /root/rivermind-data             <- 目标端原有旧版数据，完全不动
#
# 用法：bash scripts/backup_to_gpuhome.sh [--dry-run]
# 执行位置：源服务器项目根（脚本会被上传后远端执行）
# =============================================================================
set -u
SRC_ROOT=/3250410011/root/rivermind-data
DST_HOST=sc01-ssh.gpuhome.cc
DST_PORT=31017
DST_USER=root
KEY=/root/.ssh/id_ed25519_rivermind_backup
SNAP=rivermind-data_20261022
DST_BASE=/root/rivermind-data/${SNAP}
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

SSH="ssh -i $KEY -p $DST_PORT -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
LOG=/tmp/backup_to_gpuhome_$(date +%H%M%S).log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "==== 0. 前置检查 ===="
TOTAL=$(du -sb "$SRC_ROOT" 2>/dev/null | cut -f1)
log "源数据大小: $(du -sh "$SRC_ROOT" | cut -f1) ($TOTAL 字节)"
$SSH "$DST_USER@$DST_HOST" "echo 密钥登录 OK; df -h /root/rivermind-data | tail -1" || { log "SSH 失败，中止"; exit 2; }

log "==== 1. 目标端建目录 ===="
$SSH "$DST_USER@$DST_HOST" "mkdir -p ${DST_BASE}/rivermind-data ${DST_BASE}/_manifest && ls -ld ${DST_BASE}" || exit 3

log "==== 2. rsync 同步（断点续传/校验跳过已存在）===="
# 保留 -a 全属性；不做 --delete（避免源端临时缺文件时误删备份）
# 排除项：__pycache__/.cache 等可再生内容、以及 .ipynb_checkpoints
rsync -a --info=progress2 $DRY \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='lost+found/' \
  -e "$SSH" \
  "$SRC_ROOT"/ "$DST_USER@$DST_HOST:${DST_BASE}/rivermind-data/" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
log "rsync 退出码: $RC"
[ "$RC" -ne 0 ] && { log "rsync 失败，保留日志 $LOG"; exit 4; }
[ -n "$DRY" ] && { log "dry-run 结束"; exit 0; }

log "==== 3. 生成校验清单（源端）===="
MAN=$SRC_ROOT/_manifest_src_$(date +%Y%m%d).txt
( cd "$SRC_ROOT" && find . -type f -size +1M -print0 | xargs -0 md5sum ) > "$MAN" 2>/dev/null
log "源端 >1M 文件数: $(wc -l < "$MAN")  清单: $MAN"

log "==== 4. 目标端复核（大小 + 清单）===="
rsync -a -e "$SSH" "$MAN" "$DST_USER@$DST_HOST:${DST_BASE}/_manifest/" 2>&1 | tail -2
DUP=$(du -sb "$DST_BASE/rivermind-data" | cut -f1)
log "目标端备份大小: $(numfmt --to=iec $DUP) ($DUP 字节) / 源端 $TOTAL 字节"
$SSH "$DST_USER@$DST_HOST" "cd ${DST_BASE}/_manifest && md5sum -c \$(basename $MAN) --quiet 2>&1 | head -20; echo 'check-done'"

log "==== 5. 关键产物抽查 ===="
$SSH "$DST_USER@$DST_HOST" "ls -l ${DST_BASE}/rivermind-data/MutilModel_3485988898/ResTest/v2c_E1_rgbonly_s1/weights/last.pt ${DST_BASE}/rivermind-data/MutilModel_3485988898/ResTest/_test_eval/ 2>&1 | head -6; ls ${DST_BASE}/rivermind-data/uav_data_v2/data.yaml"

log "==== 完成 ===="
log "目标路径: $DST_BASE/rivermind-data"
log "本地日志: $LOG"
