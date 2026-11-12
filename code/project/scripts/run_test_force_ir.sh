#!/bin/bash
cd /3250410011/root/rivermind-data/MutilModel_3485988898
timeout 180 /opt/conda/envs/rivermind/bin/python -u scripts/test_force_ir.py 2>&1 \
  | grep -vE "FutureWarning|warnings.warn|check_for_updates|timm.models|def forward|def backward|albumentations"
echo "=== rc=$? ==="
