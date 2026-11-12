#!/bin/bash
# 4 路并行流测总带宽：判断单流 300 kB/s 是「每流限速」还是「链路总带宽上限」
KEY=/root/.ssh/id_ed25519_rivermind_backup
SSH="ssh -i $KEY -p 31017 -o BatchMode=yes -o Compression=no root@sc01-ssh.gpuhome.cc"

echo "=== 4 路并行 10s ==="
for i in 1 2 3 4; do
  $SSH "cat > /tmp/bw$i.bin" < /dev/zero &
done
sleep 10
pkill -f "bw[1-4].bin" 2>/dev/null
sleep 1
$SSH 'for i in 1 2 3 4; do stat -c %s /tmp/bw$i.bin 2>/dev/null; done' | awk '{s+=$1} END {printf "总计 %d 字节 = %d kB/s (4 流合计)\n", s, s/10/1024}'
$SSH 'rm -f /tmp/bw?.bin'

echo "=== 当前 rsync ==="
tail -2 /tmp/backup_run.log
