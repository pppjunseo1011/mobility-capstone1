#!/usr/bin/env python3
"""V4 cached diagnostics and downstream trials; reused V3 metric kernels."""
import argparse,collections,copy,importlib.util,json,shutil,time
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
R=Path('/project');O=Path('/out');V3=R/'outputs/uwipl-v3-20261107'
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
V=mod(R/'scripts/uwipl-v3.py','v3');E=V.E;save=E.save;csvout=E.csvout;load=V.load
CFG=dict(area=1000,distance=80,score=.95,samples='11',vote=15,new=.7,high=.4)
def paths(scope):return R/'outputs/uwipl-study-20261101/inference30' if scope=='development' else V3/'holdout/inference'
def truth(scope):
 if scope=='development':return V.truth()
 V.O=V3;return V.holdout_truth()
def matches(boxes,gt):
 iou=E.iou(boxes,[g['bbox'] for g in gt]);ri,ci=linear_sum_assignment(-((iou>=.5)*1000+iou))
 return {int(j):int(i) for i,j in zip(ri,ci) if iou[i,j]>=.5}
def detection(scope,variant):
 root=paths(scope) if variant=='baseline' else O/'inference'/scope/variant
 ds=load(root/'uwipl/detection/detections.json');t=truth(scope);out=O/'diagnostics'/scope/variant;rows=[]
 for n in [50,300]:
  for stage in ['detections','selected']:
   stats=collections.defaultdict(collections.Counter);examples=[]
   for d in ds:
    cam=int(d['camera'][4:])+1;f=d['index']
    if f>=n:continue
    gt=t.get((cam,f),[]);bb=[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in d[stage]];ma=matches(bb,gt)
    for key in [(0,'all'),(cam,'all')]:stats[key].update(gt=len(gt),pred=len(bb),tp=len(ma))
    for j,g in enumerate(gt):
     h=g['bbox'][3];band='lt50' if h<50 else '50to100' if h<100 else 'ge100'
     for key in [(0,band),(cam,band)]:stats[key].update(gt=1,tp=int(j in ma))
    if n==300 and stage=='selected' and cam in [1,2,4,5,6] and variant=='baseline' and len(gt)>len(ma):examples.append((len(gt)-len(ma),cam,f,d,gt,ma))
   for (cam,band),s in sorted(stats.items()):rows.append(dict(scope=scope,variant=variant,seconds=n//10,stage=stage,camera=cam,scout_camera='all' if cam==0 else f'cam_{cam-1}',gt_height=band,gt_boxes=s['gt'],pred_boxes=s['pred'] if band=='all' else None,true_positives=s['tp'],recall=s['tp']/s['gt'] if s['gt'] else None,precision=s['tp']/s['pred'] if band=='all' and s['pred'] else None,iou=.5))
   if examples:
    chosen=[]
    for cam in [1,2,4,5,6]:
     picked=[]
     for e in sorted((x for x in examples if x[1]==cam),reverse=True,key=lambda x:x[0]):
      if all(abs(e[2]-q[2])>=10 for q in picked):picked.append(e)
      if len(picked)==3:break
     chosen+=picked
    for missed,cam,f,d,gt,ma in chosen:
     im=cv2.imread(str(paths(scope)/'input'/d['file']));assert im is not None
     for j,g in enumerate(gt):
      x,y,w,h=map(int,g['bbox']);color=(0,180,0) if j in ma else (0,0,255);cv2.rectangle(im,(x,y),(x+w,y+h),color,2);cv2.putText(im,str(g['id']),(x,max(15,y)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
     cv2.putText(im,f'{scope} c{cam:03d} / cam_{cam-1} frame={d["source_frame"]} missed={missed}; RED=miss GREEN=hit IoU>=0.5',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
     dest=out/'missed_frames';dest.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(dest/f'c{cam:03d}_f{f:03d}.jpg'),im)
    save(out/'missed_frames/selection.json',[dict(camera=c,scout_camera=f'cam_{c-1}',model_frame=f,source_frame=d['source_frame'],missed_gt=miss,gt_boxes=len(gt)) for miss,c,f,d,gt,ma in chosen])
 csvout(out/'detection_metrics.csv',rows);save(out/'protocol.json',dict(matching='maximum-cardinality IoU>=0.5 one-to-one, then maximum IoU',height_bins='GT box height: <50, [50,100), >=100 px',height_precision=None,height_precision_reason='False positives have no GT height; report precision only for overall/per-camera groups',selected='objectness >=0.3; upstream detections objectness >0.1 after NMS',gt_usage='evaluation only'))
 print('DETECTION_METRICS',scope,variant,flush=True)
def tracking(scope,variant,vote=15):
 start=time.monotonic();root=paths(scope) if variant=='baseline' else O/'inference'/scope/variant;ds,ft,keys=V.inputs(root)
 for c in range(1,9):
  for f in range(300):
   if (c,f) not in keys:keys[c,f]=(dict(selected=[],source_frame=None,file=None),np.empty((0,2560),np.float32),-1)
 cfg=dict(CFG,vote=vote);t=truth(scope)
 if variant=='baseline':
  p=V3/'sct'/scope/'new_0.7_high_0.4';tracks=load(p/'rows.json');sft=np.load(p/'features.npy')
 else:
  V.O=O;tracks,sft=V.sct(ds,keys,cfg,scope+'_'+variant)
 for n in [50,300]:
  dest=O/'experiments'/scope/(variant if vote==15 else f'vote_{vote}')/f's{n//10}'
  V.predict(cfg,n,keys,tracks,sft,dest);V.evaluate(dest,t,n)
  conf=load(dest/'summary.json');conf.update(scope=scope,variant=variant,stcra=False,tracking_elapsed_seconds=time.monotonic()-start);save(dest/'summary.json',conf)
  if variant=='baseline' and vote==15:
   old=V3/(f'experiments/E5/high_0.4_only/s{n//10}' if scope=='development' else f'holdout/anchor/s{n//10}')
   expected=load(old/'metrics.json')['summary'];actual=load(dest/'metrics.json')['summary']
   for k in ['HOTA_2D','IDF1_2D','IDF1_view_aware','IDSW_2D']:assert expected[k]==actual[k],(k,expected[k],actual[k])
   save(dest/'v3_equivalence.json',dict(status='PASS',source=str(old),metrics='exact equality'))
 print('TRACKING_DONE',scope,variant,vote,flush=True)
def postvote(scope):
 t=truth(scope)
 for n in [50,300]:
  src=V3/(f'experiments/E6/4_3_2/s{n//10}' if scope=='development' else f'holdout/final/s{n//10}')
  orig=load(src/'final_predictions.json');pred=copy.deepcopy(orig);groups=collections.defaultdict(list)
  for r in pred:groups[r['camera'],r['local_id']].append(r)
  changed=0
  for rs in groups.values():
   rs.sort(key=lambda r:r['frame']);last=None
   for k in range(0,len(rs),15):
    chunk=rs[k:k+15];choice=last if len(chunk)<15 and last is not None else collections.Counter(r['id'] for r in chunk).most_common(1)[0][0]
    for r in chunk:changed+=r['id']!=choice;r['id']=choice
    last=choice
  for label,rr in [('stcra_on',orig),('stcra_postvote15',pred)]:
   dest=O/'experiments'/scope/label/f's{n//10}'
   for name in ['summary.json','anchor_members.json','hungarian_unassigned.json']:save(dest/name,load(src/name))
   save(dest/'anchor_predictions.json',load(src/'anchor_predictions.json'));save(dest/'final_predictions.json',rr)
   conf=load(dest/'summary.json');conf.update(postvote=label=='stcra_postvote15',postvote_changed_rows=changed if label=='stcra_postvote15' else 0,stcra_thresholds=[4,3,2],scope=scope,variant=label,pose_reused=True);save(dest/'summary.json',conf);V.evaluate(dest,t,n,rr)
 print('POSTVOTE_DONE',scope,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['detection','tracking','postvote']);p.add_argument('--scope',choices=['development','holdout'],required=True);p.add_argument('--variant',default='baseline');p.add_argument('--vote',type=int,default=15);a=p.parse_args()
 if a.action=='detection':detection(a.scope,a.variant)
 elif a.action=='tracking':tracking(a.scope,a.variant,a.vote)
 else:postvote(a.scope)
