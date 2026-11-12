import json
from pathlib import Path

P = Path('/3250410011/root/rivermind-data/MutilModel_3485988898/ResTest/_test_eval/xdom_E1_to_mendeley_test.json')
d = json.load(open(P))
print('tag    :', d['tag'])
print('weights:', d['weights'])
print('domain :', d.get('domain'), 'split', d.get('split'))
print('frames :', d['n_frames'], ' gt', d['n_gt'], ' empty', d['n_empty_frames'])
for r in d['rows']:
    print('  conf>=%-6s TP=%-5d FP=%-5d FN=%-5d P=%.4f R=%.4f FP/空帧=%.2f IoU=%.3f'
          % (r['conf'], r['TP'], r['FP'], r['FN'], r['precision'], r['recall'],
             r['fp_per_empty_frame'], r['mean_iou_tp']))
