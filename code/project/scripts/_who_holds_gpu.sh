#!/bin/bash
echo "=== GPU 占用进程（宿主视角）==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  echo "--- pid=$p ---"
  if [ -d "/proc/$p" ]; then
    ps -p "$p" -o pid,ppid,user,etime,rss,cmd --no-headers 2>/dev/null | cut -c1-180
    echo "  cgroup: $(cat /proc/$p/cgroup 2>/dev/null | head -2 | tr '\n' ' ')"
  else
    echo "  /proc/$p 不可见（属于其它容器/命名空间）"
  fi
done
echo
echo "=== 本容器 GPU 显存合计 ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
