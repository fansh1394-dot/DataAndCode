#!/bin/bash
cd /3250410011/root/rivermind-data/MutilModel_3485988898 || exit 1
for d in YOLO26/ResTest/*/; do
  n=$(basename "$d")
  csv="$d/results.csv"
  if [ -f "$csv" ]; then
    lines=$(wc -l < "$csv")
    best="no"; [ -f "$d/weights/best.pt" ] && best="yes"
    last="no"; [ -f "$d/weights/last.pt" ] && last="yes"
    echo "$n  epochs=$((lines-1))  best=$best  last=$last"
  else
    echo "$n  (no results.csv)"
  fi
done
echo '=== _test_eval ==='
ls YOLO26/ResTest/_test_eval/ 2>/dev/null || echo '(empty)'
