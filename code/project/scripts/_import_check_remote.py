import sys
sys.path.insert(0, '/3250410011/root/rivermind-data/MutilModel_3485988898')
import ultralytics
print('ultralytics', ultralytics.__version__, ultralytics.__file__)
from ultralytics import YOLOMM
print('YOLOMM OK')
from ultralytics.nn.tasks import DualHeadFusionDetectV26
print('DualHeadFusionDetectV26 OK')
from ultralytics.nn.mm.dual_head import DualHeadFusionDetect
print('DualHeadFusionDetect(v11) OK')
from ultralytics.nn.mm.dual_head_v26 import DualHeadFusionDetectV26 as V26b
print('DualHeadFusionDetectV26 (mm) OK, same obj:', V26b is DualHeadFusionDetectV26)
import inspect
from ultralytics.models.yolo.multimodal import train as _t
src = inspect.getsource(_t)
print('plot_training_samples passes bboxes_ir:', 'bboxes_ir' in src)
from ultralytics.nn.mm import dual_val as _dv
print('dual_val late_route wired:', '_late_route_cfg' in inspect.getsource(_dv))
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.tasks as _tk
print('tasks has v26 detect map:', 'DualHeadFusionDetectV26' in inspect.getsource(_tk))
