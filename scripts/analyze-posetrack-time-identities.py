#!/usr/bin/env python3
"""Posthoc analysis of original IDs and Missing recoveries; no tracking input."""
import collections,csv,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];OLD=R/'outputs/posetrack-mesh-20261104';O=R/'outputs/posetrack-time-study-20261106/association'
def read(p):return list(csv.DictReader(p.open()))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def write(p,rs):
 if not rs:return
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def consensus(rs):
 ids=[str(r['matched_gt_id']) for r in rs if r.get('matched_gt_id') not in ['',None]]
 return ids[0] if ids and len(set(ids))==1 else None

def main():
 O.mkdir(exist_ok=True);neural={r['observation_id']:r for r in read(O/'neural_observation_gt.csv')};p=OLD/'C_30s';pred=read(p/'posthoc_gt_matches.csv');byid=collections.defaultdict(list);byobs=collections.defaultdict(list)
 for r in pred:byid[int(r['global_id'])].append(r);byobs[r['observation_id']].append(r)
 targets={};requested=[]
 for tid in [28,38]:
  rr=byid[tid];write(O/f'global_id_{tid}_analysis.csv',rr);first=min(int(r['model_frame']) for r in rr);initial=[r for r in rr if int(r['model_frame'])==first];seed=consensus(initial)
  bad=next((r for r in rr if seed is not None and r['matched_gt_id'] and str(r['matched_gt_id'])!=seed),None)
  if bad:requested.append(bad['observation_id'])
  targets[tid]=dict(predicted_id=tid,first_output_frame=first,first_output_source_frame=277+first,initial_GT_consensus=seed,
    GT_counts=dict(collections.Counter(r['matched_gt_id'] or 'unmatched' for r in rr)),first_inconsistent_observation=bad,
    inference='GT inconsistency is posthoc IoU>=.5 evidence; first actual identity error is not guaranteed if initial cluster/GT match is ambiguous')
 print('IDENTITY_TARGETS',[(k,v['initial_GT_consensus'],v['first_inconsistent_observation']) for k,v in targets.items()],flush=True)
 # Entire candidate set, rather than only the accepted edge, at the first conflict.
 pairs=[]
 for r in csv.DictReader((p/'affinity_pairs.csv').open()):
  if r['observation_id'] in requested:pairs.append(r)
 write(O/'affinity_examples.csv',pairs);write(O/'hungarian_matches.csv',[r for r in pairs if r['hungarian_selected']=='True'])
 events=json.loads((p/'events.json').read_text());selected=[];births={};react=[];decision=None
 snaps=json.loads((p/'track_snapshots.json').read_text());snap_byframe={s['frame']:s for s in snaps}
 for e in events:
  typ=e['event'];tid=e.get('track_id')
  if typ=='initialization' and tid in [28,38] and tid not in births:births[tid]=e
  if typ=='missing_reid_decision':decision=e
  if tid in [28,38] and typ!='state_aware_reid':selected.append(e)
  if typ=='reactivation':
   f=e['frame'];rr=[r for r in byid[tid] if int(r['model_frame'])<f];last=max((int(r['model_frame']) for r in rr),default=None)
   before=[] if last is None else [neural[r['observation_id']] for r in rr if int(r['model_frame'])==last];after=[neural[oid] for oid in e['observations'].values() if oid in neural]
   strict_g0,g1=consensus(before),consensus(after);historical=[r for r in rr if r['matched_gt_id']];g0=consensus(historical);status='unresolved_mixed_or_unmatched_GT' if g0 is None or g1 is None else ('correct' if g0==g1 else 'incorrect')
   snap=next(t for t in snap_byframe[f]['before'] if t['id']==tid);enter=f
   while enter>0:
    prev=next((t for t in snap_byframe[enter-1]['after'] if t['id']==tid),None)
    if prev is None or prev['state']!=3:break
    enter-=1
   assert decision is not None and decision['best_track_id']==tid and decision['accepted']
   row=dict(frame=f,source_frame=277+f,track_id=tid,new_candidate_id=decision['new_id'],status=status,previous_GT=g0,new_GT=g1,
     strict_last_output_GT=strict_g0,historical_GT_set=sorted({r['matched_gt_id'] for r in historical}),last_GT_supported_frame=max((int(r['model_frame']) for r in historical),default=None),previous_GT_set=sorted({r['matched_gt_id'] for r in before}),new_GT_set=sorted({r['matched_gt_id'] for r in after}),
     last_output_frame=last,detection_gap_frames=None if last is None else f-last,missing_status_frames=f-enter,
     update_age_before=snap['age'],feature_bank_count_before=snap['feat_count'],similarity=max(decision['similarities']),threshold=decision['threshold'],
     new_observation_ids=list(e['observations'].values()),previous_observations=[r['observation_id'] for r in before],
     matching_path='match_with_miss_tracks: max feature-bank cosine >0.5; no geometry/Hungarian acceptance gate in this recovery function')
   react.append(row)
 for tid,v in targets.items():
  v['initialization_event']=births.get(tid);v['candidate_pairs_at_first_conflict']=[r for r in pairs if v['first_inconsistent_observation'] and r['observation_id']==v['first_inconsistent_observation']['observation_id']]
 save(O/'global_id_analysis.json',targets);save(O/'id_28_38_events.json',selected)
 write(O/'feature_bank_events.csv',[e for e in selected if e['event'] in ['feature_bank','feature_bank_gate']]);write(O/'missing_reactivation.csv',react)
 recovery_summary=dict(collections.Counter(r['status'] for r in react));recovery_summary.update(total=len(react),policy='Previous identity: unique GT consensus of annotated prior trajectory; reference frame gap recorded. New identity: actual reactivated inputs including unconfirmed targets, IoU>=.5. Mixed/unmatched evidence unresolved. Strict immediate predecessor GT also retained. Conditional accuracy must not be presented as accuracy over all 14 events.')
 save(O/'missing_reactivation_summary.json',recovery_summary);save(O/'missing_reactivation_detail.json',react)
 # Feature-bank/state histories for requested original IDs.
 write(O/'state_history.csv',[r for r in read(p/'state_history.csv') if int(r['track_id']) in [28,38]])
 # Clustering evidence if a conflict is born before any Hungarian association.
 save(O/'initialization_examples.json',[e for e in events if e['event']=='new_track_clustering' and any(x in e['observation_ids'] for x in requested)])
 print('RECOVERY',recovery_summary,flush=True)
if __name__=='__main__':main()
