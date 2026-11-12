# 清理尝试备份到智川云的全部内容
# 1) 目标端删除本次快照目录、临时测试文件、本次添加的 SSH 公钥
# 2) 源端删除为备份生成的一次性密钥与测试脚本
set -u

echo "=== 1. 目标端（智川云）清理 ==="
ssh -i /root/.ssh/id_ed25519_rivermind_backup -p 31017 -o BatchMode=yes root@sc01-ssh.gpuhome.cc '
  echo "删除前占用:"; du -sh /root/rivermind-data/rivermind-data_20261022 2>/dev/null || echo "（不存在）"
  rm -rf /root/rivermind-data/rivermind-data_20261022
  rm -rf /tmp/bt_a /tmp/bt_b /tmp/bt_c /tmp/pt /tmp/big100.bin /tmp/big200.bin /tmp/big200b.bin /tmp/bw*.bin /tmp/bwtest*.bin
  echo "删除后:"; ls -d /root/rivermind-data/rivermind-data_20261022 2>/dev/null || echo "已删除"
  echo "--- 原路径旧数据是否完好（应仍在）---"; ls /root/rivermind-data | head -6
  echo "--- 移除本次新增的备份公钥 ---"
  if [ -f /root/.ssh/authorized_keys ]; then
    cp /root/.ssh/authorized_keys /root/.ssh/authorized_keys.bak_$(date +%s)
    grep -v "rivermind-backup-" /root/.ssh/authorized_keys > /root/.ssh/ak.tmp || true
    mv /root/.ssh/ak.tmp /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "剩余公钥行数: $(wc -l < /root/.ssh/authorized_keys)"
  fi
'
RC=$?
echo "目标端清理退出码: $RC"

echo "=== 2. 源端清理（一次性密钥 + 测试脚本）==="
rm -f /root/.ssh/id_ed25519_rivermind_backup /root/.ssh/id_ed25519_rivermind_backup.pub
rm -f /tmp/_bt.sh /tmp/_bt2.sh /tmp/_bt3.sh /tmp/_bt4.sh /tmp/_fb_*.txt
echo "源端密钥: $(ls /root/.ssh/id_ed25519_rivermind_backup 2>/dev/null || echo '已删除')"
