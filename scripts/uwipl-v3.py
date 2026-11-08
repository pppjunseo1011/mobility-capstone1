#!/usr/bin/env python3
"""Sequential UWIPL V3 development experiments; GT isolated to audit/selection/oracles."""
import argparse,collections,csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering

R=Path('/project');O=Path('/out');OLD=R/'outputs/uwipl-anchor-sweep-20261102';PREV=R/'outputs/uwipl-study-20261101'
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
E=mod(R/'scripts/evaluate-uwipl-study.py','eval_v3')
save=E.save;csvout=E.csvout
def load(p):return json.loads(p.read_text())
BASE=dict(area=1000,distance=80,score=.95,samples='11',vote=5,new=.7,high=.6)
def inputs(root=PREV/'inference30'):
 names=['osnet_ms_m_c','osnet_ibn_ms_m_c','osnet_ain_ms_m_c','osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0','osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0']
 ds=load(root/'uwipl/detection/detections.json');paths=[next((root/'uwipl/reid').glob(x+'*.npy')) for x in names]
 maps=[load(p.with_name(p.stem+'-rows.json')) for p in paths];assert all(x==maps[0] for x in maps)
 ft=np.concatenate([np.load(p) for p in paths],axis=1);assert np.isfinite(ft).all();keys={};off=0
 for d in ds:
  n=len(d['selected']);keys[int(d['camera'][4:])+1,d['index']]=(d,ft[off:off+n],off)
  for j in range(n):assert maps[0][off+j]['key']==d['key'] and maps[0][off+j]['box_index']==j
  off+=n
 assert off==len(ft);return ds,ft,keys
def truth(path=R/'outputs/osc-analysis-20261101/s30_th40/gt_selected.json',start=277):
 t=collections.defaultdict(list)
 for g in load(path):t[int(g['camera_id'][4:])+1,g['source_frame']-start].append(dict(id=g['track_id'],bbox=g['bbox_xywh'],world=g['world_xyz']))
 return t
def frame_set(n,s):return list(range(0,n,10)) if s=='1s' else sorted(set(np.linspace(0,n-1,int(s),dtype=int).tolist()))
def sct(ds,keys,cfg,tag):
 if cfg['new']==.7 and cfg['high']==.6 and tag=='development':return load(OLD/'sct_rows.json'),np.load(OLD/'sct_current_features.npy')
 dest=O/'sct'/tag/f"new_{cfg['new']}_high_{cfg['high']}"
 if (dest/'rows.json').exists():return load(dest/'rows.json'),np.load(dest/'features.npy')
 sys.path.insert(0,'/opt/uwipl/BoT-SORT');from tracker.bot_sort import BoTSORT
 native=mod('/opt/uwipl/BoT-SORT/tools/aic_track.py','native_v3');args=native.make_parser().parse_args(['/unused'])
 args.fps=10;args.track_low_thresh=.1;args.nms_thres=1;args.aspect_ratio_thresh=10;args.new_track_thresh=cfg['new'];args.track_high_thresh=cfg['high']
 rows=[];fts=[]
 for cam in range(1,9):
  tr=BoTSORT(args,frame_rate=10)
  for c,f in sorted(k for k in keys if k[0]==cam):
   d,feat,_=keys[c,f];det=np.asarray(d['selected'],float).reshape(-1,7)[:,:5]
   for t in tr.update(det.copy(),feat.copy()):
    if t.tlwh[2]*t.tlwh[3]<=args.min_box_area:continue
    rows.append(dict(camera=cam,frame=f,source_frame=d['source_frame'],local_id=int(t.track_id),bbox=t.tlwh.tolist(),score=float(t.score),input_file=d['file']))
    fts.append(t.curr_feat.copy())
  print('SCT',tag,cfg['new'],cfg['high'],cam,len(rows),flush=True)
 save(dest/'rows.json',rows);np.save(dest/'features.npy',np.asarray(fts));save(dest/'config.json',vars(args));return rows,np.asarray(fts)
def predict(cfg,n,keys,tracks,sft,dest,oracle_truth=None):
 start=time.monotonic();frames=frame_set(n,cfg['samples']);members=[];af=[]
 for cam in range(1,9):
  for f in frames:
   d,feat,off=keys[cam,f]
   for j,b in enumerate(d['selected']):
    if (b[2]-b[0])*(b[3]-b[1])>cfg['area'] and b[4]>cfg['score']:
     members.append(dict(camera=cam,frame=f,source_frame=d['source_frame'],detection_index=j,feature_row=off+j,bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]]));af.append(feat[j])
 af=np.asarray(af);assert len(af)>1
 if oracle_truth is None:
  estimate=AgglomerativeClustering(distance_threshold=cfg['distance'],n_clusters=None).fit(af);count=int(estimate.labels_.max()+1)
  labels=AgglomerativeClustering(n_clusters=count).fit(af).labels_
 else:
  matched=E.gt_matches(members,oracle_truth);ix=sorted(matched);members=[members[i] for i in ix];af=af[ix]
  gids=sorted({matched[i]['gt']['id'] for i in ix});labels=np.array([gids.index(matched[i]['gt']['id']) for i in ix]);count=len(gids)
 unit=af/np.linalg.norm(af,axis=1,keepdims=True);centers=np.stack([unit[labels==k].mean(0) for k in range(count)])
 for r,k in zip(members,labels):r['anchor_id']=int(k+1)
 cur=[(i,r) for i,r in enumerate(tracks) if r['frame']<n];lens=collections.Counter((r['camera'],r['local_id']) for _,r in cur)
 eligible=[(i,r) for i,r in cur if lens[r['camera'],r['local_id']]>=4];groups=collections.defaultdict(list)
 for i,r in eligible:groups[r['camera'],r['frame']].append((i,r))
 assigned=[];unassigned=[]
 for _,rs in sorted(groups.items()):
  feat=np.stack([sft[i] for i,r in rs]);feat/=np.linalg.norm(feat,axis=1,keepdims=True);cost=1-feat@centers.T;ri,ci=linear_sum_assignment(cost)
  for i,j in zip(ri,ci):assigned.append(dict(rs[i][1],raw_anchor_id=int(j+1),id=int(j+1),assignment_cost=float(cost[i,j])))
  for i in sorted(set(range(len(rs)))-set(ri)):unassigned.append(dict(rs[i][1]))
 groups=collections.defaultdict(list)
 for r in assigned:groups[r['camera'],r['local_id']].append(r)
 changed=0
 for rs in groups.values():
  rs.sort(key=lambda r:r['frame']);last=None
  for begin in range(0,len(rs),cfg['vote']):
   chunk=rs[begin:begin+cfg['vote']];choice=last if len(chunk)<cfg['vote'] and last is not None else collections.Counter(r['raw_anchor_id'] for r in chunk).most_common(1)[0][0]
   for r in chunk:r['id']=choice;changed+=choice!=r['raw_anchor_id']
   last=choice
 assigned.sort(key=lambda r:(r['camera'],r['frame'],r['local_id']))
 save(dest/'anchor_members.json',members);save(dest/'anchor_predictions.json',assigned);save(dest/'hungarian_unassigned.json',unassigned);np.save(dest/'anchor_mean_unit_features.npy',centers)
 save(dest/'summary.json',dict(config=cfg,nframes=n,fps=10,anchor_frames=frames,actual_sample_count=len(frames),anchors=count,anchor_observations=len(members),sct_rows=len(cur),eligible_rows=len(eligible),hungarian_unassigned=len(unassigned),output_rows=len(assigned),majority_changed_rows=int(changed),GT_used=oracle_truth is not None,evaluation_only=oracle_truth is not None,elapsed_seconds=time.monotonic()-start))
 return assigned
def evaluate(dest,t,n,pred=None):
 cfg=load(dest/'summary.json');pred=load(dest/'anchor_predictions.json') if pred is None else pred;members=load(dest/'anchor_members.json');matches=E.gt_matches(members,t)
 votes=collections.defaultdict(collections.Counter);gtanchors=collections.defaultdict(set)
 for i,p in matches.items():votes[members[i]['anchor_id']][p['gt']['id']]+=1;gtanchors[p['gt']['id']].add(members[i]['anchor_id'])
 audit=[]
 for aid in range(1,cfg['anchors']+1):
  v=votes[aid];audit.append(dict(anchor_id=aid,member_count=sum(m['anchor_id']==aid for m in members),gt_distribution=dict(v),distinct_gt=len(v),majority_gt=v.most_common(1)[0][0] if v else None,purity=max(v.values())/sum(v.values()) if v else None))
 save(dest/'anchor_gt_audit.json',dict(per_anchor=audit,gt_anchors={g:sorted(a) for g,a in gtanchors.items()},gt_usage='posthoc only unless experiment explicitly marked oracle'))
 m,detail,removed=E.metric_suite(pred,t,n);m.update(anchors=cfg['anchors'],anchor_gt_covered_people=len(gtanchors),gt_unique_people=len({g['id'] for (c,f),rs in t.items() if f<n for g in rs}),anchors_with_multiple_gt=sum(len(v)>1 for v in votes.values()),gt_people_fragmented_across_anchors=sum(len(v)>1 for v in gtanchors.values()),hungarian_unassigned=cfg['hungarian_unassigned'],anchor_observations=len(members),raw_rows=len(pred),seconds=n//10,majority_changed_rows=cfg['majority_changed_rows'])
 save(dest/'metrics.json',dict(summary=m,detail=detail));save(dest/'duplicate_removals.json',removed)
 pm=E.gt_matches(pred,t);csvout(dest/'gt_matching_results.csv',[dict(camera=r['camera'],frame=r['source_frame'],id=r['id'],matched_gt=pm[i]['gt']['id'] if i in pm else '',iou=pm[i]['iou'] if i in pm else '') for i,r in enumerate(pred)])
 csvout(dest/'predictions.csv',[dict(sequence='sequence_1',camera=f"cam_{r['camera']-1}",frame=r['source_frame'],model_frame=r['frame'],id=r['id'],x=r['bbox'][0],y=r['bbox'][1],width=r['bbox'][2],height=r['bbox'][3],score=r['score']) for r in pred])
 print('EVAL',dest.name,n,m['IDF1_view_aware'],m['HOTA_2D'],m['IDSW_2D'],flush=True);return m
def run_config(family,label,cfg,ds,keys,t):
 tracks,ft=sct(ds,keys,cfg,'development');results=[]
 for n in [50,300]:
  dest=O/'experiments'/family/label/f's{n//10}'
  if not (dest/'metrics.json').exists():
   predict(cfg,n,keys,tracks,ft,dest);m=evaluate(dest,t,n)
  else:m=load(dest/'metrics.json')['summary']
  results.append(m)
 return dict(label=label,config=cfg,mean_IDF1_view_aware=float(np.mean([m['IDF1_view_aware'] for m in results])),mean_HOTA_2D=float(np.mean([m['HOTA_2D'] for m in results])),sum_IDSW=sum(m['IDSW_2D'] for m in results))
def tune():
 ds,ft,keys=inputs();t=truth();selected=dict(BASE);history=[]
 families=[('V2',[('baseline',dict(BASE))])]
 for family in ['V2','E1','E2','E3','E4','E5']:
  if family=='V2':variants=[('baseline',dict(BASE))]
  elif family=='E1':variants=[(str(v),dict(selected,samples=str(v))) for v in [11,25,50,'1s']]
  elif family=='E2':variants=[(str(v),dict(selected,score=v)) for v in [.95,.9,.8,.7]]
  elif family=='E3':variants=[(str(v),dict(selected,distance=v)) for v in [60,80,100,120]]
  elif family=='E4':variants=[(str(v),dict(selected,vote=v)) for v in [5,10,15]]
  else:variants=[('unchanged',dict(selected)),('new_0.5_only',dict(selected,new=.5)),('high_0.4_only',dict(selected,high=.4))]
  ranking=[run_config(family,label,cfg,ds,keys,t) for label,cfg in variants]
  ranking.sort(key=lambda r:(-r['mean_IDF1_view_aware'],-r['mean_HOTA_2D'],r['sum_IDSW'],r['config']!=selected))
  selected=ranking[0]['config'];history.append(dict(family=family,ranking=ranking,selected_config=selected))
  save(O/'selection_history.json',dict(rule='equal-weight 5s/30s mean view-aware IDF1; ties mean HOTA2D, lower summed IDSW, unchanged setting',development_GT_used=True,development_clips_overlap=True,history=history,selected_config=selected))
  print('SELECTED',family,selected,flush=True)
  if family=='V2':
   for sec in [5,30]:
    new=O/f'experiments/V2/baseline/s{sec}';old=OLD/f'experiments/area_1000_distance_80/s{sec}'
    assert load(new/'anchor_members.json')==load(old/'anchor_members.json')
    aa=load(new/'anchor_predictions.json');bb=load(old/'anchor_predictions.json');assert len(aa)==len(bb)
    errors=[]
    for x,y in zip(aa,bb):
     errors.append(abs(x['assignment_cost']-y['assignment_cost']))
     assert {k:v for k,v in x.items() if k!='assignment_cost'}=={k:v for k,v in y.items() if k!='assignment_cost'}
    assert max(errors,default=0)<1e-6
    save(new/'v2_equivalence.json',dict(prediction_rows_and_IDs='exact equality',assignment_cost_max_absolute_error=max(errors,default=0),assignment_cost_tolerance=1e-6,note='float32 BLAS rounding; no changed assignment or metrics'))
    a=load(new/'metrics.json')['summary'];b=load(old/'anchor_metrics.json')['summary']
    for k in ['HOTA_2D','IDF1_2D','IDF1_view_aware','IDSW_2D']:assert a[k]==b[k],k
   save(O/'v2_equivalence.json',dict(status='PASS',members_prediction_boxes_IDs_metrics='exact equality with existing area1000/distance80 V2',assignment_costs='float32 absolute error <1e-6; per-window records retained'))
 rows=[]
 for p in sorted((O/'experiments').glob('*/*/s*/metrics.json')):
  rows.append(dict(family=p.parts[-4],variant=p.parts[-3],**load(p)['summary'],**load(p.parent/'summary.json')['config']))
 csvout(O/'experiment_comparison.csv',rows)

def diagnose():
 ds,ft,keys=inputs();t=truth();tracks=load(OLD/'sct_rows.json');allrows=[]
 for n in [50,300]:
  dest=O/'diagnostics'/f's{n//10}';dest.mkdir(parents=True,exist_ok=True);af=set(frame_set(n,'11'));v2=OLD/f'experiments/area_1000_distance_80/s{n//10}'
  members=load(v2/'anchor_members.json');am=E.gt_matches(members,t);covered={v['gt']['id'] for v in am.values()};ids={g['id'] for (c,f),rr in t.items() if f<n for g in rr}
  av=collections.defaultdict(collections.Counter)
  for i,v in am.items():av[members[i]['anchor_id']][v['gt']['id']]+=1
  majorities={v.most_common(1)[0][0] for v in av.values()}
  stages={}
  for key in ['detections','selected']:
   rows=[]
   for (c,f),(d,_,_) in keys.items():
    if f>=n:continue
    for b in d[key]:rows.append(dict(camera=c,frame=f,bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]],score=b[4]))
   stages['yolox_post_nms' if key=='detections' else 'selected_score_ge_0.3']=rows
  stages['botsort_output']=[r for r in tracks if r['frame']<n];stages['anchor_assigned']=load(v2/'anchor_predictions.json')
  stages['anchor_after_eval_duplicate_id_removal']=E.dedup(stages['anchor_assigned'])[0]
  gtcount=sum(len(rr) for (c,f),rr in t.items() if f<n);matchsets={};summ=[]
  for name,rows in stages.items():
   ma=E.gt_matches(rows,t);found={(rows[i]['camera'],rows[i]['frame'],v['gt']['id']) for i,v in ma.items()};matchsets[name]=found
   summ.append(dict(seconds=n//10,stage=name,rows=len(rows),gt_boxes=gtcount,matched_gt_boxes=len(found),recall=len(found)/gtcount,covered_gt_people=len({k[2] for k in found}),iou_threshold=.5))
  csvout(dest/'stage_gt_recall.csv',summ);allrows.extend(summ)
  obs=[];people=[]
  for gid in sorted(ids-majorities):
   appearances=[(c,f,g) for (c,f),rr in t.items() if f<n for g in rr if g['id']==gid];counts=collections.Counter()
   for c,f,g in appearances:
    if f not in af:continue
    d,_,_=keys[c,f];bb=d['detections'];ious=E.iou([g['bbox']],[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in bb])[0];j=int(np.argmax(ious)) if len(bb) else None
    b=None if j is None or ious[j]<.5 else bb[j];area=None if b is None else (b[2]-b[0])*(b[3]-b[1]);score=None if b is None else b[4]
    hits=[b for b,v in zip(bb,ious) if v>=.5];ap=[(b[2]-b[0])*(b[3]-b[1])>1000 for b in hits];sp=[b[4]>.95 for b in hits]
    reason='no_yolox_box_iou_ge_0.5' if not hits else ('passes_joint_filter' if any(a and s for a,s in zip(ap,sp)) else 'both_area_and_score' if not any(ap) and not any(sp) else 'area_only' if not any(ap) else 'score_only' if not any(sp) else 'area_and_score_pass_on_different_boxes')
    counts[reason]+=1;obs.append(dict(gt_id=gid,camera=c,frame=277+f,reason=reason,best_iou=0 if j is None else float(ious[j]),best_box_area=area,best_box_score=score,overlapping_detection_count=len(hits),joint_pass_count=sum(a and s for a,s in zip(ap,sp)),diagnostic='any overlapping box may pass; not one-to-one recall'))
   people.append(dict(gt_id=gid,covered_as_anchor_member=gid in covered,covered_as_anchor_majority=gid in majorities,total_gt_observations=len(appearances),appears_at_anchor_times=bool(counts),anchor_time_observations=sum(counts.values()),reason_counts=dict(counts)))
  save(dest/'uncovered_gt_people.json',people);csvout(dest/'uncovered_gt_observations.csv',obs)
  csvout(dest/'uncovered_gt_people.csv',[dict(p,reason_counts=json.dumps(p['reason_counts'])) for p in people])
  save(dest/'protocol.json',dict(anchor_covered_gt=len(covered),uncovered_gt=len(ids-covered),majority_covered_gt=len(majorities),majority_uncovered_gt=len(ids-majorities),anchor_frames=sorted(af),note='Uncovered-person audit includes all identities without a majority anchor, including member-only identities swallowed by mixing. YOLOX raw means saved post-NMS, objectness>0.1, clipped/truncated boxes, not logits. Every overlapping IoU>=0.5 box considered for filter diagnosis; same person may fail differently at different times. Stage recall is one-to-one IoU>=0.5.'))
 csvout(O/'stage_gt_recall.csv',allrows);print('DIAGNOSTICS COMPLETE',flush=True)

def pose_prepare():
 cfg=load(O/'selection_history.json')['selected_config'];ds,ft,keys=inputs();rows,sft=sct(ds,keys,cfg,'development')
 old=load(PREV/'pose_requests.json');lookup={(int(r['camera'][4:])+1,r['frame'],tuple(r['bbox_xywh'])):r['row'] for r in old}
 requests=[];mapping=[]
 for r in rows:
  key=(r['camera'],r['frame'],tuple(r['bbox']));idx=lookup.get(key)
  if idx is None:
   idx=len(old)+len(requests);lookup[key]=idx
   requests.append(dict(row=len(requests),camera=f"cam_{r['camera']-1}",camera_id=r['camera'],frame=r['frame'],source_frame=r['source_frame'],bbox_xywh=r['bbox'],score=r['score'],input_file=keys[r['camera'],r['frame']][0]['file']))
  mapping.append(dict(camera=r['camera'],frame=r['frame'],local_id=r['local_id'],pose_row=idx))
 save(O/'pose_new/pose_requests.json',requests);save(O/'pose_mapping.json',mapping)
 save(O/'pose_reuse.json',dict(previous_pool_rows=len(old),new_pose_boxes=len(requests),selected_sct_rows=len(rows),reused_selected_rows=sum(r['pose_row']<len(old) for r in mapping),matching='exact camera/frame/BBox equality; no GT'))
 print('POSE REQUESTS',len(requests),flush=True)

def e6():
 import copy,cv2
 sys.path.insert(0,'/opt/uwipl/STCRA')
 source=Path('/opt/uwipl/STCRA/run_stcra.py').read_text();start=source.index('    cam_weight = {');end=source.index('    avg_tracks =',start)
 patched=source[:start]+"    cam_weight = {f'c{i:03d}': 1.0 for i in range(1,9)}\n"+source[end:]
 p=O/'run_stcra_scout.py';p.write_text(patched);st=mod(p,'stcra_v3');fu=mod('/opt/uwipl/STCRA/file_utils.py','feet_v3')
 kp=np.concatenate([np.load(PREV/'sct_keypoints.npy'),np.load(O/'pose_new/sct_keypoints.npy')]);assert np.isfinite(kp).all()
 posemap={(r['camera'],r['frame'],r['local_id']):r['pose_row'] for r in load(O/'pose_mapping.json')}
 cfg=load(O/'selection_history.json')['selected_config'];winner=load(O/'selection_history.json')['history'][-1]['ranking'][0]['label'];t=truth();cal={};H={}
 for cam in range(1,9):
  c=load(Path('/data')/f'calibrations/cam_{cam-1}.json');cal[cam]=c;K,Rm,T=[np.array(c[k],float) for k in ['K','R','T']];H[f'c{cam:03d}']=K@np.column_stack([Rm[:,:2],T.reshape(3)])
 st.cam_utils.HOMO={'S001':H};save(O/'stcra_homographies.json',dict(H=H,assumption='single plane Z=0; undistorted feet; dataset world units treated as meters; not calibrated AIC pixel conversion'))
 ranked=[]
 for schedule,thresholds in [('off',[]),('2.5_2_1.5',[2.5,2.,1.5]),('4_3_2',[4.,3.,2.])]:
  scores=[]
  for n in [50,300]:
   src=O/f'experiments/E5/{winner}/s{n//10}';dest=O/f'experiments/E6/{schedule}/s{n//10}';dest.mkdir(parents=True,exist_ok=True)
   pred=load(src/'anchor_predictions.json');base={};ts={}
   for r in pred:r['pose_row']=posemap[r['camera'],r['frame'],r['local_id']];assert r['pose_row'] not in base;base[r['pose_row']]=r
   for name in ['summary.json','anchor_members.json','hungarian_unassigned.json']:save(dest/name,load(src/name))
   save(dest/'anchor_predictions.json',pred)
   for cam in range(1,9):
    rs=[r for r in pred if r['camera']==cam];fp=dest/f'feet_c{cam:03d}.txt'
    with fp.open('w') as f:
     for r in rs:f.write(','.join(map(str,[r['frame'],r['id'],*r['bbox'],r['score'],-1,-1,-1,*kp[r['pose_row'],[15,16]].ravel().tolist()]))+'\n')
    ct=fu.load_tracking(str(fp),enable_feet=True) if rs else collections.defaultdict(dict);lookup=collections.defaultdict(list)
    for r in rs:lookup[r['frame'],r['id'],tuple(r['bbox'])].append(r['pose_row'])
    for tid,tr in ct.items():
     for frame,locs in tr.items():
      for loc in locs:
       idx=lookup[frame,tid,tuple(loc[:4])].pop(0);c=cal[cam];K=np.array(c['K'],float);dist=np.array(c['dist'],float)
       und=cv2.undistortPointsIter(np.array(loc[4:6]).reshape(1,1,2),K,dist,None,K,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(2)
       assert np.isfinite(und).all();loc[4:6]=und.tolist();loc.append(idx)
    ts[f'c{cam:03d}']=ct
   rounds=[];events=[]
   def flat(state):return {int(loc[7]):dict(base[int(loc[7])],id=int(tid)) for cam,tr in state.items() for tid,fr in tr.items() for frame,ls in fr.items() for loc in ls}
   for i,(threshold,confidence) in enumerate(zip(thresholds,[.65,.7,.75]),1):
    before=flat(ts);ts,proposals,avg=st.id_reassignment(ts,dis_thr=threshold,conf_thr=confidence);after=flat(ts)
    ev=[dict(pose_row=k,camera=r['camera'],frame=r['frame'],old_id=r['id'],new_id=after[k]['id'] if k in after else None) for k,r in before.items() if k not in after or r['id']!=after[k]['id']]
    events.extend(dict(e,round=i) for e in ev);rounds.append(dict(round=i,threshold=threshold,confidence=confidence,input_rows=len(before),output_rows=len(after),reassigned=sum(e['new_id'] is not None for e in ev),deleted=sum(e['new_id'] is None for e in ev)))
    save(dest/f'round_{i}_proposals.json',proposals)
   final=pred if not thresholds else list(flat(ts).values());save(dest/'final_predictions.json',final);save(dest/'rounds.json',rounds);save(dest/'events.json',events)
   conf=load(dest/'summary.json');conf.update(stcra_thresholds=thresholds,interpolation=False,stage='anchor' if not thresholds else 'i-STCRA three rounds');save(dest/'summary.json',conf)
   scores.append(evaluate(dest,t,n,final))
  ranked.append(dict(label=schedule,mean_IDF1_view_aware=float(np.mean([m['IDF1_view_aware'] for m in scores])),mean_HOTA_2D=float(np.mean([m['HOTA_2D'] for m in scores])),sum_IDSW=sum(m['IDSW_2D'] for m in scores)))
 ranked.sort(key=lambda r:(-r['mean_IDF1_view_aware'],-r['mean_HOTA_2D'],r['sum_IDSW'],r['label']!='off'))
 save(O/'final_selection.json',dict(config=cfg,stcra=ranked[0]['label'],ranking=ranked,rule='same development ranking as E1..E5',GT_used_for_selection=True,holdout_not_used=True));print('E6 SELECTED',ranked[0],flush=True)

def oracle_anchors():
 ds,ft,keys=inputs();t=truth();cfg=load(O/'selection_history.json')['selected_config'];tracks,sft=sct(ds,keys,cfg,'development')
 for n in [50,300]:
  dest=O/f'oracles/GT_grouped_candidate_anchors/s{n//10}';predict(cfg,n,keys,tracks,sft,dest,oracle_truth=t);evaluate(dest,t,n)
 save(O/'oracles/GT_grouped_candidate_anchors/protocol.json',dict(evaluation_only=True,GT_used_in_inference=True,normal_results_comparable=False,definition='Same selected neural anchor candidates grouped by matched GT identity instead of Ward; unmatched candidates removed; real embeddings and Hungarian/vote retained',limitation='Not a perfect all-GT-person anchor oracle; people absent from eligible candidates remain uncovered. Diagnostic reference, not guaranteed mathematical upper bound.'))

def oracle_boxes():
 cfg=load(O/'selection_history.json')['selected_config'];ds,ft,keys=inputs(O/'oracles/GT_boxes/inference');tracks,sft=sct(ds,keys,cfg,'GT_boxes_oracle');t=truth()
 for n in [50,300]:
  dest=O/f'oracles/GT_boxes/s{n//10}';predict(cfg,n,keys,tracks,sft,dest);conf=load(dest/'summary.json');conf.update(GT_used=True,evaluation_only=True,GT_identity_used_for_tracking=False,oracle='GT detection boxes and confidence1, real crop features');save(dest/'summary.json',conf);evaluate(dest,t,n)

def holdout_truth():
 mapping={(r['camera']+1,r['source_frame']):r['index'] for r in load(O/'holdout_selection.json')['mapping']};t=collections.defaultdict(list)
 for g in load(O/'holdout/gt_selected.json'):
  cam=int(g['camera_id'][4:])+1;frame=mapping[cam,g['source_frame']];t[cam,frame].append(dict(id=g['track_id'],bbox=g['bbox_xywh'],world=g['world_xyz']))
 return t

def holdout_track():
 selected=load(O/'final_selection.json');cfg=selected['config'];ds,ft,keys=inputs(O/'holdout/inference')
 # Explicit missing capture slots advance tracker time without fabricating detections.
 for c in range(1,9):
  for f in range(300):
   if (c,f) not in keys:keys[c,f]=(dict(selected=[],source_frame=None,file=None),np.empty((0,2560),np.float32),-1)
 tracks,sft=sct(ds,keys,cfg,'holdout');assert all(r['source_frame'] is not None for r in tracks)
 requests=[]
 for i,r in enumerate(tracks):
  r['pose_row']=i;requests.append(dict(row=i,camera=f"cam_{r['camera']-1}",frame=r['frame'],source_frame=r['source_frame'],bbox_xywh=r['bbox'],score=r['score'],input_file=keys[r['camera'],r['frame']][0]['file']))
 save(O/'holdout/pose/pose_requests.json',requests)
 # Frozen selection copied before holdout metrics are computed. No holdout tuning.
 save(O/'holdout/frozen_configuration.json',dict(selected,selected_before_holdout_evaluation=True))
 t=holdout_truth()
 for n in [50,300]:
  dest=O/f'holdout/anchor/s{n//10}';predict(cfg,n,keys,tracks,sft,dest);evaluate(dest,t,n)
 baseline_tracks,baseline_ft=sct(ds,keys,BASE,'holdout')
 for n in [50,300]:
  dest=O/f'holdout/V2_baseline/s{n//10}';predict(BASE,n,keys,baseline_tracks,baseline_ft,dest);evaluate(dest,t,n)
 print('HOLDOUT SCT/ANCHOR DONE',len(tracks),flush=True)

def holdout_finish():
 import cv2
 sys.path.insert(0,'/opt/uwipl/STCRA');st=mod(O/'run_stcra_scout.py','stcra_holdout');fu=mod('/opt/uwipl/STCRA/file_utils.py','feet_holdout')
 H={k:np.array(v) for k,v in load(O/'stcra_homographies.json')['H'].items()};st.cam_utils.HOMO={'S001':H};cal={c:load(Path('/data')/f'calibrations/cam_{c-1}.json') for c in range(1,9)}
 kp=np.load(O/'holdout/pose/sct_keypoints.npy');assert np.isfinite(kp).all();sel=load(O/'holdout/frozen_configuration.json');thresholds={'off':[],'2.5_2_1.5':[2.5,2,1.5],'4_3_2':[4,3,2]}[sel['stcra']];t=holdout_truth()
 for n in [50,300]:
  src=O/f'holdout/anchor/s{n//10}';dest=O/f'holdout/final/s{n//10}';dest.mkdir(parents=True,exist_ok=True);pred=load(src/'anchor_predictions.json');base={r['pose_row']:r for r in pred};assert len(base)==len(pred)
  for name in ['summary.json','anchor_members.json','hungarian_unassigned.json','anchor_predictions.json']:save(dest/name,load(src/name))
  ts={}
  for cam in range(1,9):
   rs=[r for r in pred if r['camera']==cam];fp=dest/f'feet_c{cam:03d}.txt'
   with fp.open('w') as f:
    for r in rs:f.write(','.join(map(str,[r['frame'],r['id'],*r['bbox'],r['score'],-1,-1,-1,*kp[r['pose_row'],[15,16]].ravel().tolist()]))+'\n')
   ct=fu.load_tracking(str(fp),enable_feet=True) if rs else collections.defaultdict(dict);lookup=collections.defaultdict(list)
   for r in rs:lookup[r['frame'],r['id'],tuple(r['bbox'])].append(r['pose_row'])
   for tid,tr in ct.items():
    for frame,locs in tr.items():
     for loc in locs:
      idx=lookup[frame,tid,tuple(loc[:4])].pop(0);c=cal[cam];K=np.array(c['K'],float);dist=np.array(c['dist'],float)
      und=cv2.undistortPointsIter(np.array(loc[4:6]).reshape(1,1,2),K,dist,None,K,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(2);assert np.isfinite(und).all();loc[4:6]=und.tolist();loc.append(idx)
   ts[f'c{cam:03d}']=ct
  def flat(state):return {int(loc[7]):dict(base[int(loc[7])],id=int(tid)) for cam,tr in state.items() for tid,fr in tr.items() for f,ls in fr.items() for loc in ls}
  rounds=[];events=[]
  for i,(dist,conf) in enumerate(zip(thresholds,[.65,.7,.75]),1):
   before=flat(ts);ts,proposals,avg=st.id_reassignment(ts,dis_thr=dist,conf_thr=conf);after=flat(ts)
   ev=[dict(pose_row=k,source_frame=r['source_frame'],camera=r['camera'],old_id=r['id'],new_id=after[k]['id'] if k in after else None) for k,r in before.items() if k not in after or r['id']!=after[k]['id']]
   events.extend(dict(e,round=i) for e in ev);rounds.append(dict(round=i,distance=dist,confidence=conf,input_rows=len(before),output_rows=len(after),changed=len(ev)));save(dest/f'round_{i}_proposals.json',proposals)
  final=list(flat(ts).values()) if thresholds else pred;save(dest/'final_predictions.json',final);save(dest/'events.json',events);save(dest/'rounds.json',rounds)
  conf=load(dest/'summary.json');conf.update(stcra_thresholds=thresholds,interpolation=False,development_selected_no_holdout_tuning=True);save(dest/'summary.json',conf);evaluate(dest,t,n,final)
 print('HOLDOUT FINAL DONE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['tune','diagnose','pose_prepare','e6','oracle_anchors','oracle_boxes','holdout_track','holdout_finish']);a=p.parse_args();O.mkdir(parents=True,exist_ok=True);globals()[a.action]()
