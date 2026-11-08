#!/usr/bin/env python3
"""Freeze timestamp-selected holdout inputs; annotations are exported separately."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v3-20261107';D=R/'datasets/scout/extracted/scout'
selection=json.loads((O/'holdout_selection.json').read_text());rows=sorted(selection['mapping'],key=lambda r:(r['camera'],r['index']));frames=[]
for r in rows:
 c=r['camera'];f=r['source_frame'];i=r['index'];src=D/f'images/sequence_1/cam_{c}/image_{f}.jpg';rel=f'cam_{c}/frame_{i:06d}.jpg';target=O/'holdout/inference/input'/rel
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():shutil.copyfile(src,target)
 digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==hashlib.sha256(src.read_bytes()).hexdigest()
 frames.append(dict(index=i,source_frame=f,source_path=str(src.relative_to(D)),timestamp_seconds=r['timestamp_seconds'],timestamp_duplicate=False,camera=f'cam_{c}',model_camera_id=c+1,key=f'cam_{c}_{i:06d}',file=rel,sha256=digest))
root=O/'holdout';(root/'inference/input/manifest.json').write_text(json.dumps(dict(sequence='sequence_1',frames=frames,gt_used=False,selection='timestamp-only; holdout_selection.json'),indent=2)+'\n')
for stage in ['detection','reid']:(root/'inference/uwipl'/stage).mkdir(parents=True,exist_ok=True)
skews=[]
for i in range(300):
 ts=[r['timestamp_seconds'] for r in rows if r['index']==i];skews.append(max(ts)-min(ts));assert skews[-1]<=.050000001
summary=dict(frames=len(frames),nframes=300,per_camera=selection['counts'],max_skew_seconds=max(skews),missing_slots=2400-len(rows),GT_used_for_selection=False,source_ranges={f'cam_{c}':[min(r['source_frame'] for r in rows if r['camera']==c),max(r['source_frame'] for r in rows if r['camera']==c)] for c in range(8)},note='Timestamp consistency verified, not independent hardware synchronization. Unavailable camera/time slots are explicitly absent and excluded from evaluation GT.')
(root/'input_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'frame_mapping.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
# Separate post-selection GT export. These boxes are never used by inference preparation.
sys.path.insert(0,str(R/'adapters'));from scout_adapter.annotations import iter_ground_truth
manifest=dict(sequence='sequence_1',cameras=[dict(id=f'cam_{c}',frames=[dict(source_frame=r['source_frame']) for r in rows if r['camera']==c]) for c in range(8)])
gt=list(iter_ground_truth(D,manifest,'mot'));(root/'gt_selected.json').write_text(json.dumps(gt,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True);print('GT exported separately:',len(gt),flush=True)
