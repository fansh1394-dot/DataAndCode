#!/bin/bash
# 两个基线的最终结果汇总
R=/3250410011/root/rivermind-data/MutilModel_3485988898

for NAME in v2_E1_rgbonly_s0 v2_E2_ironly_s0; do
  CSV=$R/ResTest/$NAME/results.csv
  echo "########## $NAME ##########"
  if [ ! -f "$CSV" ]; then echo "  [无 results.csv]"; echo ""; continue; fi
  echo "  完成 $(($(wc -l < "$CSV") - 1)) 轮"
  echo "  --- 每 10 轮 + 末 3 轮 ---"
  awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; n=0; next}
  {n++; e=$1; m50=$(h["metrics/mAP50(B)"]); m95=$(h["metrics/mAP50-95(B)"]); b=$(h["train/box_loss"]); c=$(h["train/cls_loss"])}
  END{
    for(i=1;i<=n;i++){} }' "$CSV" >/dev/null
  awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
  {printf "%s %s %s %s %s\n", $1, $(h["metrics/mAP50(B)"]), $(h["metrics/mAP50-95(B)"]), $(h["train/box_loss"]), $(h["train/cls_loss"])}' "$CSV" \
  | awk '{n=NR} {a[n]=$0} END{
      for(i=1;i<=n;i++){
        if(i%10==1 || i>n-3){
          split(a[i],f," ");
          printf "    ep%-3s mAP50=%-7.4f mAP50-95=%-7.4f box=%-7.4f cls=%-7.4f\n", f[1],f[2],f[3],f[4],f[5];
        }
      }}'
  echo "  --- 最终值 ---"
  awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
  {m50=$(h["metrics/mAP50(B)"]); m95=$(h["metrics/mAP50-95(B)"]); e=$1}
  END{printf "    ep%s  mAP50=%.4f  mAP50-95=%.4f\n", e, m50, m95}' "$CSV"
  echo ""
done

echo "########## 关键对照 ##########"
echo "  双头模型（C4, 100ep）:"
echo "    rgb 头     = 0.7052"
echo "    ir  头     = 0.6417"
echo "    fuse 头    = 0.7126"
echo "    fuse_ir 头 = 0.6427"
echo ""
E1=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} {v=$(h["metrics/mAP50-95(B)"])} END{printf "%.4f", v}' $R/ResTest/v2_E1_rgbonly_s0/results.csv 2>/dev/null)
E2=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} {v=$(h["metrics/mAP50-95(B)"])} END{printf "%.4f", v}' $R/ResTest/v2_E2_ironly_s0/results.csv 2>/dev/null)
echo "  单模态基线:"
echo "    RGB-only   = ${E1:-未完成}"
echo "    IR-only    = ${E2:-未完成}"
echo ""
echo "########## 增益核算（bash awk 计算） ##########"
awk -v e1="${E1:-0}" -v e2="${E2:-0}" 'BEGIN{
  fuse=0.7126; fuse_ir=0.6427; rgbh=0.7052; irh=0.6417;
  if(e1>0){ printf "  fuse    (0.7126) - RGB-only (%.4f) = %+.4f\n", e1, fuse-e1 }
  else    { printf "  fuse    (0.7126) - RGB-only (未完成)\n" }
  if(e2>0){ printf "  fuse_ir (0.6427) - IR-only  (%.4f) = %+.4f\n", e2, fuse_ir-e2 }
  else    { printf "  fuse_ir (0.6427) - IR-only  (未完成)\n" }
  printf "\n  参考：头 vs 基线（同模态）\n"
  if(e1>0){ printf "    rgb 头 (0.7052) - RGB-only (%.4f) = %+.4f\n", e1, rgbh-e1 }
  if(e2>0){ printf "    ir  头 (0.6417) - IR-only  (%.4f) = %+.4f\n", e2, irh-e2 }
}'
echo ""
echo "########## GPU / 负载 ##########"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
uptime | sed 's/^ *//'
