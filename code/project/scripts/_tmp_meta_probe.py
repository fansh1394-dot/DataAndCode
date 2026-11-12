# -*- coding: utf-8 -*-
"""打印 metadata 结构样例。"""
import json

p = '/root/rivermind-data/uav_data_v2/metadata/test.json'
d = json.load(open(p))
print('top keys:', list(d.keys()))
fr = d['frames']
ks = list(fr)[:3]
print('n frames:', len(fr))
for k in ks:
    print(k, '->', json.dumps(fr[k], ensure_ascii=False))
