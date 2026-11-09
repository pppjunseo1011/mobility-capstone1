#!/usr/bin/env python3
"""Config-only experiments using unchanged V3/V4 functions and cached inference."""
import argparse,collections,copy,importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
R=Path('/project');O=Path('/out');V4=R/'outputs/uwipl-v4-20261108';V3=R/'outputs/uwipl-v3-20261107'
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
W=mod(R/'scripts/uwipl-v4.py','w4');V=W.V;E=W.E;load=E.json.loads
def read(p):return json.loads(p.read_text())
save=E.save;csvout=E.csvout

def vote(rows,length,raw=False):
 out=copy.deepcopy(rows);groups=collections.defaultdict(list);changed=0
 for r in out:groups[r['camera'],r['local_id']].append(r)
 for rs in groups.values():
  rs.sort(key=lambda r:r['frame']);last=None
  for k in range(0,len(rs),length):
   chunk=rs[k:k+length];key='raw_anchor_id' if raw else 'id';choice=last if len(chunk)<length and last is not None else collections.Counter(r[key] for r in chunk).most_common(1)[0][0]
   for r in chunk:changed+=r[key]!=choice;r['id']=choice
   last=choice
 return out,changed

def copy_metadata(src,dest):
 for name in ['summary.json','anchor_members.json','hungarian_unassigned.json']:save(dest/name,read(src/name))

def anchor(scope):
 truth=W.truth(scope)
 for seconds in [5,30]:
  n=seconds*10;src=V4/'experiments'/scope/'baseline'/f's{seconds}';base=read(src/'anchor_predictions.json')
  # Validate the same majority rule against both cached vote15 and vote30 outputs.
  for length in [15,30]:
   p,_=vote(base,length,True);old=V4/'experiments'/scope/('baseline' if length==15 else 'vote_30')/f's{seconds}'/'anchor_predictions.json';assert p==read(old)
  for length in [40,50]:
   dest=O/'experiments'/scope/f'vote_{length}'/f's{seconds}'
   if (dest/'metrics.json').exists():continue
   p,changed=vote(base,length,True);copy_metadata(src,dest);save(dest/'anchor_predictions.json',p);cfg=read(dest/'summary.json');cfg['config']['vote']=length;cfg.update(majority_changed_rows=changed,stcra=False,source_reused=str(src),new_neural_inference=False);save(dest/'summary.json',cfg);V.evaluate(dest,truth,n,p)
 save(O/'validation'/scope/'voting_parity.json',dict(status='PASS',vote15_and_30_predictions_exact=True,neural_inference_reused=True))

def stcra(scope):
 sys.path.insert(0,'/opt/uwipl/STCRA');st=mod(V3/'run_stcra_scout.py','frozen_stcra');fu=mod('/opt/uwipl/STCRA/file_utils.py','feet')
 st.cam_utils.HOMO={'S001':{k:np.asarray(v) for k,v in read(V3/'stcra_homographies.json')['H'].items()}}
 cal={c:read(Path('/data')/f'calibrations/cam_{c-1}.json') for c in range(1,9)}
 if scope=='development':
  kp=np.concatenate([np.load(R/'outputs/uwipl-study-20261101/sct_keypoints.npy'),np.load(V3/'pose_new/sct_keypoints.npy')]);posemap={(r['camera'],r['frame'],r['local_id']):r['pose_row'] for r in read(V3/'pose_mapping.json')}
 else:
  kp=np.load(V3/'holdout/pose/sct_keypoints.npy');posemap={(r['camera'],r['frame'],r['local_id']):r['pose_row'] for r in read(V3/'holdout/anchor/s30/anchor_predictions.json')}
 assert np.isfinite(kp).all();truth=W.truth(scope)
 # vote15 is a numerical parity control; vote30 is the requested new setting.
 for initial_vote in [15,30]:
  for seconds in [5,30]:
   src=V4/'experiments'/scope/('baseline' if initial_vote==15 else 'vote_30')/f's{seconds}';dest=O/'experiments'/scope/f'vote{initial_vote}_stcra432'/f's{seconds}';pred=read(src/'anchor_predictions.json');base={};state={}
   for r in pred:r['pose_row']=posemap[r['camera'],r['frame'],r['local_id']];assert r['pose_row'] not in base;base[r['pose_row']]=r
   if not (dest/'final_predictions.json').exists():
    copy_metadata(src,dest);save(dest/'anchor_predictions.json',pred)
    for cam in range(1,9):
     rs=[r for r in pred if r['camera']==cam];fp=dest/f'feet_c{cam:03d}.txt'
     with fp.open('w') as f:
      for r in rs:f.write(','.join(map(str,[r['frame'],r['id'],*r['bbox'],r['score'],-1,-1,-1,*kp[r['pose_row'],[15,16]].ravel().tolist()]))+'\n')
     ct=fu.load_tracking(str(fp),enable_feet=True) if rs else collections.defaultdict(dict);lookup=collections.defaultdict(list)
     for r in rs:lookup[r['frame'],r['id'],tuple(r['bbox'])].append(r['pose_row'])
     for tid,tr in ct.items():
      for frame,locs in tr.items():
       for loc in locs:
        idx=lookup[frame,tid,tuple(loc[:4])].pop(0);c=cal[cam];K=np.asarray(c['K'],float);dist=np.asarray(c['dist'],float);und=cv2.undistortPointsIter(np.asarray(loc[4:6]).reshape(1,1,2),K,dist,None,K,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(2);assert np.isfinite(und).all();loc[4:6]=und.tolist();loc.append(idx)
     state[f'c{cam:03d}']=ct
    def flat(ts):return {int(loc[7]):dict(base[int(loc[7])],id=int(tid)) for cam,tr in ts.items() for tid,fr in tr.items() for f,ls in fr.items() for loc in ls}
    rounds=[]
    for i,(distance,confidence) in enumerate(zip([4.,3.,2.],[.65,.7,.75]),1):
     before=flat(state);state,proposals,avg=st.id_reassignment(state,dis_thr=distance,conf_thr=confidence);after=flat(state);events=[dict(pose_row=k,old_id=r['id'],new_id=after[k]['id'] if k in after else None) for k,r in before.items() if k not in after or r['id']!=after[k]['id']];save(dest/f'round_{i}_events.json',events);rounds.append(dict(round=i,distance=distance,confidence=confidence,input_rows=len(before),output_rows=len(after),changed=len(events)))
    final=list(flat(state).values());save(dest/'final_predictions.json',final);save(dest/'rounds.json',rounds);cfg=read(dest/'summary.json');cfg.update(stcra_thresholds=[4,3,2],stcra_confidence=[.65,.7,.75],new_neural_inference=False,frozen_stcra=str(V3/'run_stcra_scout.py'));save(dest/'summary.json',cfg)
   final=read(dest/'final_predictions.json')
   if initial_vote==15:
    old=V4/'experiments'/scope/'stcra_on'/f's{seconds}'
    def canonical(p):return sorted([(r['camera'],r['frame'],r['local_id'],r['id'],r['bbox']) for r in p])
    assert canonical(final)==canonical(read(old/'final_predictions.json'))
    save(dest/'parity.json',dict(status='PASS',previous_V4_STCRA_predictions_exact=True))
   else:
    if not (dest/'metrics.json').exists():V.evaluate(dest,truth,seconds*10,final)
    for post in [15,30]:
     target=O/'experiments'/scope/f'vote30_stcra432_post{post}'/f's{seconds}'
     if (target/'metrics.json').exists():continue
     result,changed=vote(final,post);copy_metadata(dest,target);save(target/'anchor_predictions.json',pred);save(target/'final_predictions.json',result);cfg=read(target/'summary.json');cfg.update(postvote=post,postvote_changed_rows=changed);save(target/'summary.json',cfg);V.evaluate(target,truth,seconds*10,result)
 print('STCRA COMPLETE',scope,flush=True)

def filtered(rows,truth,n,min_height=None,ignore=False):
 if min_height is None:return rows,truth,[]
 gt={k:[g for g in gs if g['bbox'][3]>=min_height] for k,gs in truth.items()};removed=[]
 if not ignore:return rows,gt,removed
 groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[r['camera'],r['frame']].append((i,r))
 for key,rs in groups.items():
  high=gt.get(key,[]);small=[g for g in truth.get(key,[]) if g['bbox'][3]<min_height];bb=[r['bbox'] for i,r in rs];io=E.iou(bb,[g['bbox'] for g in high]);ri,ci=linear_sum_assignment(-((io>=.5)*1000+io));used={int(i) for i,j in zip(ri,ci) if io[i,j]>=.5};left=[i for i in range(len(rs)) if i not in used];io=E.iou([bb[i] for i in left],[g['bbox'] for g in small]);ri,ci=linear_sum_assignment(-((io>=.5)*1000+io))
  for i,j in zip(ri,ci):
   if io[i,j]>=.5:removed.append(dict(row_index=rs[left[i]][0],camera=key[0],frame=key[1],id=rs[left[i]][1]['id'],ignored_gt=small[j]['id'],iou=float(io[i,j])))
 excluded={r['row_index'] for r in removed};return [r for i,r in enumerate(rows) if i not in excluded],gt,removed

def metric(rows,truth,n,threshold=.5):
 metrics=[E.HOTA(),E.Identity({'PRINT_CONFIG':False,'THRESHOLD':threshold}),E.CLEAR({'PRINT_CONFIG':False,'THRESHOLD':threshold})];allres=[{} for m in metrics]
 for cam in range(1,9):
  data=E.metric_data(rows,truth,[(cam,f) for f in range(n)])
  for m,res in zip(metrics,allres):res[cam]=m.eval_sequence(data)
 h,identity,clear=[m.combine_sequences(res) for m,res in zip(metrics,allres)];vd=E.metric_data(rows,truth,[(cam,f) for f in range(n) for cam in range(1,9)]);vi=metrics[1].eval_sequence(vd);ix=int(np.argmin(abs(np.asarray(metrics[0].array_labels)-.3)))
 summary=dict(HOTA_2D=float(np.mean(h['HOTA'])),DetA_2D=float(np.mean(h['DetA'])),AssA_2D=float(np.mean(h['AssA'])),HOTA_at_0_3=float(h['HOTA'][ix]),DetA_at_0_3=float(h['DetA'][ix]),AssA_at_0_3=float(h['AssA'][ix]),IDF1_2D=float(identity['IDF1']),IDF1_view_aware=float(vi['IDF1']),IDSW_2D=int(clear['IDSW']),MOTA=float(clear['MOTA']),Recall=float(clear['CLR_Re']),Precision=float(clear['CLR_Pr']),rows_evaluated=len(rows),gt_boxes=sum(len(gs) for (c,f),gs in truth.items() if f<n),gt_unique_people=len({g['id'] for (c,f),gs in truth.items() if f<n for g in gs}),global_ids=len({r['id'] for r in rows}),iou_for_identity_clear=threshold)
 return summary,dict(HOTA=h,Identity=identity,CLEAR=clear,Identity_view=vi)

def reevaluate(scope):
 truth=W.truth(scope)
 for src in sorted((V4/'experiments'/scope).glob('*/s*')):
  if not (src/'metrics.json').exists():continue
  n=read(src/'summary.json')['nframes'];rows=read(src/('final_predictions.json' if (src/'final_predictions.json').exists() else 'anchor_predictions.json'));rows,duplicates=E.dedup(rows)
  for protocol,height,ignore,threshold in [('height_ge50_strict',50,False,.5),('height_ge50_ignore',50,True,.5),('iou_0.3',None,False,.3)]:
   dest=O/'reevaluation'/scope/src.parent.name/src.name/protocol
   if (dest/'metrics.json').exists():continue
   pred,gt,excluded=filtered(rows,truth,n,height,ignore);summary,detail=metric(pred,gt,n,threshold);summary.update(scope=scope,variant=src.parent.name,seconds=n//10,protocol=protocol,ignored_prediction_rows=len(excluded),duplicate_rows_removed=len(duplicates));save(dest/'metrics.json',dict(summary=summary,detail=detail));save(dest/'ignored_rows.json',excluded)
   if protocol=='iou_0.3':
    old=read(src/'metrics.json')['summary'];assert all(abs(summary[k]-old[k])<1e-12 for k in ['HOTA_2D','DetA_2D','AssA_2D'])
   print('REVAL',scope,src.parent.name,n,protocol,flush=True)
 # Protocol semantic checks on a small case, not scores fitted to real GT.
 t={(1,0):[dict(id=1,bbox=[0,0,10,60]),dict(id=2,bbox=[30,0,10,30])]};p=[dict(camera=1,frame=0,id=1,bbox=[0,0,10,60]),dict(camera=1,frame=0,id=2,bbox=[30,0,10,30])]
 pp,tt,excluded=filtered(p,t,1,50,True);assert len(pp)==1 and len(excluded)==1 and len(tt[1,0])==1
 pp,tt,excluded=filtered(p,t,1,50,False);assert len(pp)==2 and not excluded
 m,_=metric([dict(camera=1,frame=0,id=1,bbox=[4,0,10,60])],{(1,0):[dict(id=1,bbox=[0,0,10,60])]},1,.3);m5,_=metric([dict(camera=1,frame=0,id=1,bbox=[4,0,10,60])],{(1,0):[dict(id=1,bbox=[0,0,10,60])]},1,.5);assert m['IDF1_2D']==1 and m5['IDF1_2D']==0 and m['HOTA_2D']==m5['HOTA_2D']
 save(O/'validation'/scope/'reevaluation.json',dict(status='PASS',HOTA_mean_unchanged_by_identity_CLEAR_threshold=True,height_ignore_matching_test=True,height_strict_retains_all_predictions=True,iou03_vs05_semantic_test=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['anchor','stcra','reevaluate']);p.add_argument('--scope',required=True,choices=['development','holdout']);a=p.parse_args();globals()[a.action](a.scope)
