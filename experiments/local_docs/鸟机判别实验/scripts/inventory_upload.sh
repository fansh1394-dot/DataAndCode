#!/bin/bash
D=/3250410011/root/rivermind-data/MutilModel_3485988898/YOLO26/ResTest
find "$D" -name 'best.pt' -printf '%s\n' | awk '{s+=$1;n++} END{printf "best.pt 数量=%d 合计=%.2f GB\n", n, s/1073741824}'
find "$D" -name 'last.pt' -printf '%s\n' | awk '{s+=$1;n++} END{printf "last.pt 数量=%d 合计=%.2f GB\n", n, s/1073741824}'
echo "results.csv 数量: $(find "$D" -name 'results.csv' | wc -l)"
echo "run 数量: $(find "$D" -maxdepth 1 -mindepth 1 -type d | wc -l)"
echo "bdc3 相关 run:"
find "$D" -maxdepth 1 -mindepth 1 -type d -name 'bdc3*' -printf '%f\n'
