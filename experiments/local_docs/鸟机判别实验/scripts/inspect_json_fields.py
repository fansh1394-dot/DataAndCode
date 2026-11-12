# -*- coding: utf-8 -*-
import json
from pathlib import Path

d = json.loads(Path("/root/rivermind-fs/birddrone/other_raw/balloon/custom_train.json").read_text())
print(d["images"][0])
print("keys", list(d["images"][0].keys()))
print("cats", d.get("categories"))
