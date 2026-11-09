#!/usr/bin/env python3
"""V4 YOLOX single-variable detection variants; inference has no GT access."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torchvision.ops import nms
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import preproc
p=argparse.ArgumentParser();p.add_argument('--height',type=int,default=800);p.add_argument('--width',type=int,default=1440);p.add_argument('--nms',type=float,default=.7);p.add_argument('--tiles',action='store_true');a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);start=time.monotonic()
o=Path('/outputs');o.mkdir(parents=True,exist_ok=True)
exp=get_exp('/opt/uwipl/BoT-SORT/yolox/exps/example/mot/yolox_x_mix_det.py',None);net=exp.get_model().eval()
w=Path('/weights/bytetrack_x_mot17.pth.tar');state=torch.load(str(w),map_location='cpu');net.load_state_dict(state.get('model',state.get('state_dict',state)),strict=True);net.cuda()
frames=json.loads(Path('/sample/input/manifest.json').read_text())['frames'];results=[]
for ix,frame in enumerate(frames):
 im=cv2.imread('/sample/input/'+frame['file']);assert im is not None and im.shape==(1080,1920,3)
 # Four overlapping 60%-width/height tiles, 20% full-image overlap; no full-image pass.
 windows=[(0,0,1920,1080)] if not a.tiles else [(x,y,x+1152,y+648) for y in (0,432) for x in (0,768)]
 parts=[]
 for x,y,x2,y2 in windows:
  tile=im[y:y2,x:x2];data,ratio=preproc(tile,(a.height,a.width),(.485,.456,.406),(.229,.224,.225))
  with torch.no_grad():d=postprocess(net(torch.from_numpy(data[None]).cuda()),exp.num_classes,exp.test_conf,a.nms)[0]
  if d is None:continue
  b=d.cpu().numpy();b[:,:4]/=ratio;b=b[b[:,4]>.1];b[:,[0,2]]=b[:,[0,2]].clip(0,x2-x)+x;b[:,[1,3]]=b[:,[1,3]].clip(0,y2-y)+y;parts.append(b)
 boxes=np.concatenate(parts) if parts else np.empty((0,7),np.float32)
 if a.tiles and len(boxes):
  keep=nms(torch.from_numpy(boxes[:,:4]),torch.from_numpy(boxes[:,4]*boxes[:,5]),a.nms).numpy();boxes=boxes[keep]
 boxes[:,:4]=np.trunc(boxes[:,:4]);boxes=boxes[(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])];assert np.isfinite(boxes).all()
 selected=boxes[boxes[:,4]>=.3];selected=selected[np.argsort(-selected[:,4],kind='stable')]
 results.append(dict(frame,detections=boxes.tolist(),selected=selected.tolist()))
 if ix%100==0:print('DETECTION',ix,len(frames),'elapsed',round(time.monotonic()-start,1),flush=True)
(o/'detections.json').write_text(json.dumps(results))
torch.cuda.synchronize();report=dict(status='PASS',frames=len(frames),input_size=[a.height,a.width],nms=a.nms,tiles=a.tiles,tile_windows=windows,tiling='2x2, 20% full-image overlap, tile and global NMS, no full-image pass' if a.tiles else None,native_test_conf=exp.test_conf,objectness_filter=.1,downstream_objectness_filter=.3,max_people_per_frame=None,detections=sum(len(d['detections']) for d in results),selected=sum(len(d['selected']) for d in results),gt_used=False,elapsed_seconds=time.monotonic()-start,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch=torch.__version__,cuda=torch.version.cuda,checkpoint_sha256=hashlib.sha256(w.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(o/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
