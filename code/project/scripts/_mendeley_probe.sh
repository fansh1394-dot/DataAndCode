#!/bin/bash
# 探测 Mendeley 数据集 sn9vy5c8sm 的真实文件下载地址（只读，不下载）
set -uo pipefail
DS=sn9vy5c8sm
BASE="https://data.mendeley.com"

echo "=== 1. 数据集元数据 API ==="
curl -sS --max-time 30 -H 'Accept: application/json' \
  "$BASE/api/datasets/$DS/versions/1" -o /tmp/md_meta.json -w 'http=%{http_code} size=%{size_download}\n'
head -c 1200 /tmp/md_meta.json; echo

echo
echo "=== 2. 文件列表 API ==="
curl -sS --max-time 30 -H 'Accept: application/json' \
  "$BASE/api/datasets/$DS/files?folder_id=root&version=1" -o /tmp/md_files.json -w 'http=%{http_code} size=%{size_download}\n'
head -c 1500 /tmp/md_files.json; echo

echo
echo "=== 3. 页面内嵌 JSON 里的 file id ==="
curl -sS --max-time 30 "$BASE/datasets/$DS/1" -o /tmp/md_page.html -w 'http=%{http_code} size=%{size_download}\n'
grep -oE '"(id|file_id|filename|size)":[^,}]{1,120}' /tmp/md_page.html | head -30
echo '--- 含 download 的链接 ---'
grep -oE 'https?://[^"'"'"' ]*download[^"'"'"' ]*' /tmp/md_page.html | sort -u | head -20
