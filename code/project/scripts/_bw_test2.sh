#!/bin/bash
# 直接测吞吐：在 15 秒窗口内统计目标端实际收到的字节数
KEY=/root/.ssh/id_ed25519_rivermind_backup
SSH="ssh -i $KEY -p 31017 -o BatchMode=yes -o Compression=no root@sc01-ssh.gpuhome.cc"

echo "=== A. 单流 ssh 吞吐（8s 窗口）==="
$SSH 'cat > /tmp/bwtest.bin' < /dev/zero &
SSHPID=$!
sleep 8
kill $SSHPID 2>/dev/null
SZ=$($SSH 'stat -c %s /tmp/bwtest.bin 2>/dev/null || echo 0')
$SSH 'rm -f /tmp/bwtest.bin'
echo "8s 收到 $SZ 字节 = $(( SZ / 8 / 1024 )) kB/s"

echo "=== B. 单流 + 弱加密 ==="
SSH2="ssh -i $KEY -p 31017 -o BatchMode=yes -c aes128-gcm@openssh.com root@sc01-ssh.gpuhome.cc"
$SSH2 'cat > /tmp/bwtest2.bin' < /dev/zero &
SSHPID2=$!
sleep 8
kill $SSHPID2 2>/dev/null
SZ2=$($SSH 'stat -c %s /tmp/bwtest2.bin 2>/dev/null || echo 0')
$SSH 'rm -f /tmp/bwtest2.bin'
echo "8s 收到 $SZ2 字节 = $(( SZ2 / 8 / 1024 )) kB/s"

echo "=== C. 当前 rsync 状态 ==="
tail -2 /tmp/backup_run.log
echo "=== D. 目标端已落盘 ==="
$SSH 'du -sh /root/rivermind-data/rivermind-data_20261022; find /root/rivermind-data/rivermind-data_20261022 -type f | wc -l'
