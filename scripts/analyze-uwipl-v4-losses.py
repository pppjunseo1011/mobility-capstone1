#!/usr/bin/env python3
"""Posthoc attribution of detector changes, never used as tracker inputs."""
import collections,importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('v4','/project/scripts/uwipl-v4.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
O=M.O;V3=M.V3;rows=[];stages=[];errors=[]
def observed(ds,t,n):
 matched={};pred=[]
 for d in ds:
  c=int(d['camera'][4:])+1;f=d['index']
  if f>=n:continue
  gt=t.get((c,f),[]);bb=[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in d['selected']];ma=M.matches(bb,gt)
  for j,i in ma.items():matched[c,f,gt[j]['id']]=(gt[j],d['selected'][i])
 return matched
def found_count(pred,t,n):
 groups=collections.defaultdict(list)
 for r in pred:
  if r['frame']<n:groups[r['camera'],r['frame']].append(r)
 matched=set()
 for (c,f),rr in groups.items():
  gt=t.get((c,f),[]);ma=M.matches([r['bbox'] for r in rr],gt)
  matched.update((c,f,gt[j]['id']) for j in ma)
 return matched
for scope in ['development','holdout']:
 t=M.truth(scope);old=M.load(M.paths(scope)/'uwipl/detection/detections.json')
 for n in [50,300]:
  baseline=observed(old,t,n);gtcount=sum(len(rr) for (c,f),rr in t.items() if f<n)
  for variant in ['baseline','D1_1088x1920','D1_1280x2304','D2_tiles2x2','D3_nms0.8']:
   ds=old if variant=='baseline' else M.load(O/f'inference/{scope}/{variant}/uwipl/detection/detections.json');current=observed(ds,t,n)
   for c in range(9):
    for band in ['all','lt50','50to100','ge100']:
     def inside(key,value):
      h=value[0]['bbox'][3];return (not c or key[0]==c) and (band=='all' or (h<50 if band=='lt50' else 50<=h<100 if band=='50to100' else h>=100))
     before={k for k,v in baseline.items() if inside(k,v)};after={k for k,v in current.items() if inside(k,v)};gained=after-before;lost=before-after
     passes={k for k in gained if (current[k][1][2]-current[k][1][0])*(current[k][1][3]-current[k][1][1])>1000 and current[k][1][4]>.95}
     rows.append(dict(scope=scope,variant=variant,seconds=n//10,camera=c,gt_height=band,baseline_matched=len(before),current_matched=len(after),gained_gt_observations=len(gained),lost_gt_observations=len(lost),gained_pass_area_score=len(passes),gained_at_11_anchor_times=sum(k[1] in M.V.frame_set(n,'11') for k in gained),gained_pass_area_score_and_anchor_time=sum(k[1] in M.V.frame_set(n,'11') for k in passes)))
   ec=collections.Counter()
   for d in ds:
    c=int(d['camera'][4:])+1;f=d['index']
    if f>=n:continue
    gt=t.get((c,f),[]);bb=[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in d['selected']];ma=M.matches(bb,gt);used=set(ma.values());iou=M.E.iou(bb,[g['bbox'] for g in gt])
    ec.update(predictions=len(bb),gt_boxes=len(gt),true_positives=len(ma),false_positives=len(bb)-len(ma))
    for i in set(range(len(bb)))-used:
     ec['unmatched_prediction_overlaps_accounted_GT' if len(gt) and iou[i].max()>=.5 else 'unmatched_prediction_no_GT_iou_ge_0.5']+=1
   errors.append(dict(scope=scope,variant=variant,seconds=n//10,**{k:ec[k] for k in ['predictions','gt_boxes','true_positives','false_positives','unmatched_prediction_overlaps_accounted_GT','unmatched_prediction_no_GT_iou_ge_0.5']}))
   sctpath=V3/f'sct/{scope}/new_0.7_high_0.4/rows.json' if variant=='baseline' else O/f'sct/{scope}_{variant}/new_0.7_high_0.4/rows.json'
   pred=M.load(O/f'experiments/{scope}/{variant}/s{n//10}/anchor_predictions.json')
   for name,found in [('selected_detection',set(current)),('botsort',found_count(M.load(sctpath),t,n)),('anchor',found_count(pred,t,n)),('anchor_after_duplicate_id_removal',found_count(M.E.dedup(pred)[0],t,n))]:
    stages.append(dict(scope=scope,variant=variant,seconds=n//10,stage=name,gt_boxes=gtcount,matched_gt_boxes=len(found),recall=len(found)/gtcount,covered_gt_people=len({k[2] for k in found})))
M.csvout(O/'detection_error_breakdown.csv',errors);M.csvout(O/'detection_gain_loss.csv',rows);M.csvout(O/'stage_gt_recall.csv',stages)
print('GAIN_LOSS',len(rows),'STAGE_RECALL',len(stages),flush=True)
