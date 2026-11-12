#!/bin/bash
# 下载 Mendeley sn9vy5c8sm（UAV_dataset_EO_IR.zip, 3.18GB）并校验 sha256
set -uo pipefail

DEST=/root/rivermind-data/ext_datasets/mendeley_eo_ir
URL='https://data.mendeley.com/public-files/datasets/sn9vy5c8sm/files/a5c25f55-282f-4a95-9d4b-5ae2afd52970/file_downloaded'
SHA=3530e24c3aff89dc20c0e7afad3b1672fce23e4428dd41a7cd4d27c583a32a2a
ZIP=$DEST/UAV_dataset_EO_IR.zip
LOG=$DEST/download.log

mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "[$(date '+%F %T')] 开始下载（断点续传）" | tee -a "$LOG"
curl -L --retry 5 --retry-delay 10 -C - --max-time 7200 \
     -o "$ZIP" "$URL" >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] curl rc=$rc  size=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" | tee -a "$LOG"

if [ "$rc" -ne 0 ]; then
  echo "下载未完成，可重跑本脚本续传" | tee -a "$LOG"
  exit 1
fi

echo "[$(date '+%F %T')] 校验 sha256..." | tee -a "$LOG"
GOT=$(sha256sum "$ZIP" | cut -d' ' -f1)
echo "  want=$SHA" | tee -a "$LOG"
echo "  got =$GOT" | tee -a "$LOG"
if [ "$GOT" != "$SHA" ]; then
  echo "❌ 校验失败" | tee -a "$LOG"
  exit 2
fi
echo "✅ 校验通过" | tee -a "$LOG"
ls -lh "$ZIP" | tee -a "$LOG"
