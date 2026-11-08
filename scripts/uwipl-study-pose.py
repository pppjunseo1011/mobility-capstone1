#!/usr/bin/env python3
"""Run official HRNet-W48 on actual SCT boxes before STCRA, no GT."""
import collections,json,time
from pathlib import Path
import cv2,numpy as np,torch
from mmpose.apis import init_pose_model,inference_top_down_pose_model
from mmpose.datasets import DatasetInfo

torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
out=Path('/out');rows=json.loads((out/'pose_requests.json').read_text());start=time.monotonic()
cfg='/opt/uwipl/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
weight='/weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
model=init_pose_model(cfg,checkpoint=None,device='cpu')
state=torch.load(weight,map_location='cpu');state=state.get('state_dict',state)
result=model.load_state_dict(state,strict=True);model.cuda().eval()
groups=collections.defaultdict(list)
for r in rows:groups[r['input_file']].append(r)
kp=np.empty((len(rows),17,3),np.float32)
for i,(name,rr) in enumerate(groups.items()):
    img=cv2.imread('/inference/input/'+name);assert img is not None
    boxes=[{'bbox':np.array([*r['bbox_xywh'],r['score']],dtype=float)} for r in rr]
    with torch.no_grad():
        pred,_=inference_top_down_pose_model(model,img,boxes,bbox_thr=None,format='xywh',
            dataset=model.cfg.data.test.type,dataset_info=DatasetInfo(model.cfg.data.test.dataset_info),return_heatmap=False)
    assert len(pred)==len(rr)
    for r,p in zip(rr,pred):kp[r['row']]=p['keypoints']
    if i%100==0:print('POSE_SCT',i,len(groups),'people',len(rr),flush=True)
assert np.isfinite(kp).all();np.save(out/'sct_keypoints.npy',kp)
report={'status':'PASS','model':'HRNet-W48 top-down (official code; paper calls HigherHRNet)',
    'weight':Path(weight).name,'matched_keys':len(state),'missing_keys':list(result.missing_keys),
    'unexpected_keys':list(result.unexpected_keys),'shape':list(kp.shape),'gt_used':False,
    'scope':'Pose on SCT predicted boxes, not detector boxes reused as if tracking boxes',
    'elapsed_seconds':time.monotonic()-start,'torch':torch.__version__,'cuda':torch.version.cuda}
(out/'pose_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
