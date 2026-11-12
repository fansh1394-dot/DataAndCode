#!/bin/bash
# D3-P2 完整证据汇总：动力学、阶段统计、与对照组对比
R=/3250410011/root/rivermind-data/MutilModel_3485988898/ResTest
CSV=$R/v2_D3P2_forceir_headonly_s0/results.csv

echo "########## 1. 分段统计（坍缩期 / 地板期 / 回升期） ##########"
awk -F, 'NR==1{next} {e=$1; f=$32} 
e<=10 && e>=1 {c+=f; cn++}
e>=11 && e<=25 {m+=f; mn++}
e>=26 {r+=f; rn++; last=f}
END{
  printf "  ep1-10   (坍缩期)  均值 fuse95 = %.4f\n", c/cn;
  printf "  ep11-25  (地板期)  均值 fuse95 = %.4f\n", m/mn;
  printf "  ep26-40  (回升期)  均值 fuse95 = %.4f   末轮 = %.4f\n", r/rn, last;
}' "$CSV"

echo ""
echo "########## 2. 回升期单调性检验（ep26-40 是否持续上升） ##########"
awk -F, 'NR==1{next} e=$1; f=$32
e>=26 {printf "ep%-3d %.4f %s\n", e, f, (e>26 && f>prev ? "↑" : (e>26 ? "↓" : "")); prev=f}' "$CSV"

echo ""
echo "########## 3. 四条曲线全程对照（冻结验证 + 判据） ##########"
awk -F, 'NR==1{next} {printf "ep%-3s rgb=%-8.4f ir=%-8.4f fuse=%-8.4f fuse_ir=%-8.4f\n", $1,$22,$27,$32,$37}' "$CSV" | awk 'NR%5==1 || NR>=38'

echo ""
echo "########## 4. 与历史实验同口径对比（fuse 在无 RGB 条件下） ##########"
echo "  注：D3-P2 是「训练+验证全程强制纯 IR」，其余是「训练完再掩码」"
echo "  ---------------------------------------------------------------"
for f in \
  "_d2p1/probe_D0_scaled.json:D0 (掩码探针)" \
  "_d2p1/probe_D2P1b_scaled.json:D2-P1b (掩码探针)" \
  "_d2p1/probe_C4_recheck.json:C4 (掩码探针)" \
  "_d2p1/probe_D3_v2.json:D3-P1 (掩码探针)" ; do
  P=/3250410011/root/rivermind-data/MutilModel_3485988898/${f%%:*}
  L=${f##*:}
  if [ -f "$P" ]; then
    V=$(grep -A3 '"map50_95_overall_masked"' "$P" | grep '"fuse"' | head -1 | sed 's/.*: *//; s/,//')
    printf "  %-28s fuse=%.4f\n" "$L" "${V:-0}" 2>/dev/null || printf "  %-28s fuse=%s\n" "$L" "$V"
  else
    printf "  %-28s [文件缺失]\n" "$L"
  fi
done
echo "  ---------------------------------------------------------------"
awk -F, 'NR==1{next} {last=$32} END{printf "  %-28s fuse=%.4f  (训练+验证全程强制纯IR，40轮)\n", "D3-P2 (本次)", last}' "$CSV"

echo ""
echo "########## 5. 冻结有效性验证（rgb/ir/fuse_ir 应全程恒定） ##########"
awk -F, 'NR==1{next} {if(NR==2){r1=$22;i1=$27;fi1=$37} r=$22;i=$27;fi=$37}
END{printf "  rgb     ep1=%.4f -> ep40=%.4f  漂移=%.6f\n", r1,r,r-r1;
    printf "  ir      ep1=%.4f -> ep40=%.4f  漂移=%.6f\n", i1,i,i-i1;
    printf "  fuse_ir ep1=%.4f -> ep40=%.4f  漂移=%.6f\n", fi1,fi,fi-fi1}' "$CSV"

echo ""
echo "########## 6. 权重文件 ##########"
ls -la $R/v2_D3P2_forceir_headonly_s0/weights/ 2>/dev/null
