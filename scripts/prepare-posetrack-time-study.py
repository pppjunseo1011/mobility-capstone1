#!/usr/bin/env python3
"""Timestamp-only alignment of cached neural observations. Never reads GT."""
import collections,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R/'datasets/scout/extracted/scout';OLD=R/'outputs/posetrack-mesh-20261104';O=R/'outputs/posetrack-time-study-20261106'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def stat(a):return {'count':len(a),'max':float(max(a)),'mean':float(np.mean(a)),'p95':float(np.percentile(a,95)),'over_50ms':sum(x>.050000001 for x in a)}
def main():
 O.mkdir(exist_ok=True);T=json.loads((D/'timestamps/timestamps_sequence_1.json').read_text());keys=[json.loads((D/f'calibrations/cam_{c}.json').read_text())['view_id'].rsplit('_',1)[0] for c in range(8)]
 origin=float(np.median([T['277'][key] for key in keys]));grid=origin+np.arange(300)*.1;rows=[];selected={};percam=[]
 for cam,key in enumerate(keys):
  seen={};candidates=collections.defaultdict(list);camrows=[]
  for source in range(277,577):
   ts=float(T[str(source)][key]);slot=int(np.floor((ts-origin)/.1+.5));error=abs(ts-(origin+slot*.1));previous=seen.get(ts);duplicate=previous is not None;equal=None
   if duplicate:
    paths=[D/f'images/sequence_1/cam_{cam}/image_{f}.jpg' for f in [previous,source]]
    equal=hashlib.sha256(paths[0].read_bytes()).digest()==hashlib.sha256(paths[1].read_bytes()).digest();assert equal,'Duplicate timestamp with distinct images needs manual policy'
   else:seen[ts]=source
   status='candidate'
   if duplicate:status='duplicate_timestamp_identical_image'
   elif not 0<=slot<300:status='outside_common_30s_window'
   elif error>.025000001:status='outside_25ms_grid_tolerance'
   r=dict(sequence='sequence_1',camera=cam,timestamp_key=key,source_frame=source,timestamp_seconds=ts,common_time_index=slot,
    common_timestamp_seconds=origin+slot*.1,offset_seconds=ts-(origin+slot*.1),duplicate_of_source=previous,image_identical_to_duplicate=equal,status=status)
   rows.append(r);camrows.append(r)
   if status=='candidate':candidates[slot].append(r)
  for slot,rr in candidates.items():
   winner=min(rr,key=lambda r:(abs(r['offset_seconds']),r['source_frame']));winner['status']='selected';selected[cam,slot]=winner
   for r in rr:
    if r is not winner:r['status']='same_time_bin_collision'
  percam.append(dict(camera=cam,timestamp_key=key,source_frames=300,selected_frames=sum(r['status']=='selected' for r in camrows),
    missing_slots=300-sum(r['status']=='selected' for r in camrows),duplicate_timestamp_frames=sum(r['duplicate_of_source'] is not None for r in camrows),
    statuses=dict(collections.Counter(r['status'] for r in camrows))))
 before=[];after=[];comparison=[];slots=[]
 for slot in range(300):
  src=277+slot;old=[float(T[str(src)][k]) for k in keys];now=[selected[c,slot]['timestamp_seconds'] for c in range(8) if (c,slot) in selected]
  a=max(old)-min(old);b=max(now)-min(now);assert b<=.050000001
  before.append(a);after.append(b);comparison.append(dict(common_time_index=slot,nominal_source_frame=src,before_skew_seconds=a,after_skew_seconds=b,available_cameras=len(now)))
  for cam in range(8):slots.append(selected.get((cam,slot),dict(sequence='sequence_1',camera=cam,timestamp_key=keys[cam],source_frame=None,timestamp_seconds=None,common_time_index=slot,common_timestamp_seconds=float(grid[slot]),offset_seconds=None,duplicate_of_source=None,image_identical_to_duplicate=None,status='missing_observation')))
 write(O/'frame_mapping_all_sources.csv',rows);write(O/'frame_mapping.csv',slots);write(O/'timestamp_before_after.csv',comparison)
 save(O/'timestamp_policy.json',dict(origin_seconds=origin,grid_period_seconds=.1,grid_count=300,tolerance_seconds=.025,source_range=[277,576],camera_keys=keys,
  timestamp_unit='dataset seconds; median adjacent unique delta approximately 0.1; absolute epoch/time-of-day not independently established',
  before=stat(before),after=stat(after),per_camera=percam,GT_used=False,
  missing_policy='No synthetic detection or duplicate reuse. Missing camera/time has no evaluable image/GT; shared valid-slot mask used for all comparisons.',
  duplicate_policy='Same camera/timestamp: earliest source retained; all repetitions verified JPEG bytes identical. Nearest unique candidate per grid slot within 25ms. No extrapolation.'))
 I=O/'aligned_input';I.mkdir(exist_ok=True);shutil.copytree(OLD/'full_input30/calibration',I/'calibration',dirs_exist_ok=True)
 original=json.loads((OLD/'full_input30/observations.json').read_text());new=[];used={(r['camera'],r['source_frame']):r for r in rows if r['status']=='selected'}
 for cam in range(8):
  p=OLD/f'full_input30/input/camera_{cam+1:04d}';dest=I/f'input/camera_{cam+1:04d}';dest.mkdir(parents=True,exist_ok=True)
  rr=[r for r in original if r['camera']==cam and (cam,r['source_frame']) in used];rr.sort(key=lambda r:(used[cam,r['source_frame']]['common_time_index'],r['input_row']))
  ix=[r['input_row'] for r in rr]
  for name in ['detection','pose','reid']:
   a=np.load(p/(name+'.npy'))[ix].copy()
   if name!='reid':a[:,0]=[used[cam,r['source_frame']]['common_time_index'] for r in rr]
   assert np.isfinite(a).all();np.save(dest/(name+'.npy'),a)
  for i,r in enumerate(rr):
   q=dict(r);m=used[cam,r['source_frame']];q.update(input_row=i,old_model_frame=r['frame'],frame=m['common_time_index'],timestamp_seconds=m['timestamp_seconds'],common_timestamp_seconds=m['common_timestamp_seconds']);new.append(q)
 save(I/'observations.json',new);save(I/'preparation_summary.json',dict(input_rows=len(new),original_rows=len(original),excluded_rows=len(original)-len(new),counts=dict(collections.Counter(r['camera'] for r in new)),GT_used=False,policy='../timestamp_policy.json'))
 print(json.dumps(json.loads((O/'timestamp_policy.json').read_text()),indent=2));print('INPUT',len(new))
if __name__=='__main__':main()
