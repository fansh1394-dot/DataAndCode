import sys,os,json
PROJ="/3250410011/root/rivermind-data/MutilModel_3485988898"
sys.path.insert(0,PROJ); sys.path.insert(0,PROJ+"/scripts")
from ultralytics.nn.mm import dual_val as dv
REC={}
_orig=dv.DualHeadDetectionValidator._prepare_batch
def prep(self,si,batch):
    out=_orig(self,si,batch)
    f=batch.get("im_file"); f=str(f[si]) if f is not None else f"idx{si}"
    rp=out["ratio_pad"]
    REC[os.path.basename(f)]={"ori_shape":[int(x) for x in out["ori_shape"]],
                              "imgsz":[int(x) for x in out["imgsz"]],
                              "gain":[float(rp[0][0]),float(rp[0][1])],
                              "pad":[float(rp[1][0]),float(rp[1][1])],
                              "valid_rgb":bool(batch.get("valid_rgb")[si].item()) if batch.get("valid_rgb") is not None else True}
    return out
dv.DualHeadDetectionValidator._prepare_batch=prep
import eval_split as es
sys.argv=["eval_split.py","--weights","YOLO26/ResTest/mdp150_M0_concat_s0/weights/best.pt",
 "--data","/3250410011/root/rivermind-data/mendeley_split/data.yaml","--split","test",
 "--labels-mode","dual","--tag","pfrp","--batch","8","--workers","2","--device","cpu",
 "--out","/tmp/pfrp_eval.json"]
try: es.main()
except SystemExit: pass
finally:
    json.dump(REC,open(PROJ+"/perframe_boxes/ratio_pad.json","w"))
    print("ratio_pad frames:",len(REC),flush=True)
