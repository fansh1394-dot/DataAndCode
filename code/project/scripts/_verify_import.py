import sys

sys.path.insert(0, '/3250410011/root/rivermind-data/MutilModel_3485988898')
from ultralytics import YOLOMM  # noqa: F401
from ultralytics.nn.tasks import DualHeadFusionDetectV26, DualHeadFusionDetect  # noqa: F401
print('import OK')
print('DualHeadFusionDetectV26:', DualHeadFusionDetectV26.__name__)
