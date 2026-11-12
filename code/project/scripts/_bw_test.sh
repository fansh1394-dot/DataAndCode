#!/bin/bash
# 测源 → 智川云的裸带宽，判断 338 kB/s 是链路上限还是 ssh 加密/单流问题
KEY=/root/.ssh/id_ed25519_rivermind_backup
ssh_() { ssh -i "$KEY" -p 31017 -o BatchMode=yes root@sc01-ssh.gpuhome.cc "$@"; }

echo "=== 1. 单流 /dev/zero → 目标 /dev/null（10s）==="
timeout 10 bash -c 'dd if=/dev/zero bs=1M count=4000 2>/dev/null | ssh -i '"$KEY"' -p 31017 -o BatchMode=yes root@sc01-ssh.gpuhome.cc "cat > /dev/null"' || true
echo "（上面若无输出说明 dd 计时被 timeout 截断，见下方 iperf 式计数）"

echo "=== 2. 单流 10s 实际字节数 ==="
timeout 12 bash -c 'dd if=/dev/zero bs=1M 2>/dev/null | ssh -i '"$KEY"' -p 31017 -o BatchMode=yes root@sc01-ssh.gpuhome.cc "cat > /dev/null"' 2>/dev/null &
sleep 10
pkill -P $$ dd 2>/dev/null || true
wait 2>/dev/null || true
echo "=== 3. ssh 加密算法 ==="
ssh_ "echo target-ssh-ok"
ssh -vv -i "$KEY" -p 31017 -o BatchMode=yes root@sc01-ssh.gpuhome.cc true 2>&1 | grep -E "cipher:|MAC:" | head -4
echo "=== 4. 当前 rsync 进度 ==="
tail -3 /tmp/backup_run.log
