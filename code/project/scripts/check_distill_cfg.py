# -*- coding: utf-8 -*-
"""快速校验蒸馏配置解析结果（不加载模型）。"""
from ultralytics.nn.mm.distill.schema import load_distill_config

cfg = load_distill_config("ultralytics/cfg/mm_distill_dual800.yaml")
print("teachers:", [(t.name, t.role, t.imgsz) for t in cfg.teachers])
print("feature mappings:", len(cfg.feature_mappings), "| output teachers:", len(cfg.output_teachers))
for m in cfg.feature_mappings:
    print("  feature:", m.teacher, m.teacher_layer, "->", m.student_layer, m.tap)
