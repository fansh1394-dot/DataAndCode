#!/bin/bash
# 测试 Mendeley 下载链路的多连接/直连接口速度（只测不占用原下载）
set -uo pipefail
URL='https://data.mendeley.com/public-files/datasets/sn9vy5c8sm/files/a5c25f55-282f-4a95-9d4b-5ae2afd52970/file_downloaded'

echo "=== 解析最终域名 ==="
curl -sSIL --max-time 20 "$URL" 2>&1 | grep -iE '^(HTTP|location|content-length|server)' | head -20

echo
echo "=== 单连接 8 秒测速 ==="
curl -sS --max-time 8 -o /dev/null -w 'speed=%{speed_download} B/s  size=%{size_download}\n' -r 0-104857600 "$URL" 2>&1 | tail -2

echo
echo "=== 多连接(python) 8 秒测速 ==="
/opt/conda/envs/rivermind/bin/python - "$URL" <<'PY'
import sys, time, threading, urllib.request
url = sys.argv[1]
res = []
def w(i):
    try:
        req = urllib.request.Request(url, headers={'Range': f'bytes={i*20_000_000}-{i*20_000_000+20_000_000}'})
        t0 = time.time(); n = 0
        with urllib.request.urlopen(req, timeout=10) as r:
            while time.time() - t0 < 8:
                b = r.read(65536)
                if not b: break
                n += len(b)
        res.append(n / (time.time() - t0))
    except Exception as e:
        res.append(0.0)
ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"4 连接合计 {sum(res)/1024:.0f} KB/s  单流 {[round(x/1024) for x in res]} KB/s")
PY
