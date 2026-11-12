import sys
sys.path.insert(0, ".")
import torch
from ultralytics.utils.loss import BboxLoss

b = BboxLoss(16, iou_type="nwd")
a = torch.tensor([[10.0, 10.0, 30.0, 30.0], [100.0, 100.0, 120.0, 120.0]])
c = torch.tensor([[200.0, 200.0, 220.0, 220.0], [200.0, 200.0, 220.0, 220.0]])
print("sim(same box) =", b._compute_extended_iou(a, a).tolist(), " 期望 ~1")
print("sim(far  box) =", b._compute_extended_iou(a, c).tolist(), " 期望 ~0")
