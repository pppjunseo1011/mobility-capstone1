#!/usr/bin/env python3
"""Small, sufficient original-event extracts for offline matching analysis."""
import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];OLD=R/'outputs/posetrack-mesh-20261104';O=R/'outputs/posetrack-time-study-20261106';A=O/'association'
def load(p):return json.loads(p.read_text())
def main():
 ids=load(A/'global_id_analysis.json');rec=load(A/'missing_reactivation_detail.json');cases={r['first_inconsistent_observation']['observation_id'] for r in ids.values()};obsids=set(cases)
 tracks={28,38}
 for r in rec:tracks.add(r['track_id']);obsids.update(r['new_observation_ids']);obsids.update(r['previous_observations'])
 events=load(OLD/'C_30s/events.json');selected=[]
 for e in events:
  if (e['event']=='state_aware_reid' and e.get('observation_id') in cases) or e['event'] in ['missing_reid_decision','reactivation'] or (e.get('track_id') in tracks and e['event'] in ['initialization','feature_bank','feature_bank_gate','occlusion_switch_decision']):selected.append(e)
 (A/'review_candidate_reid_and_recovery_events.json').write_text(json.dumps(selected,indent=2)+'\n');del events
 obs=load(OLD/'full_input30/observations.json');lookup={f"cam{r['camera']}_row{r['camera_row']}":r for r in obs};vectors={}
 for c in range(8):
  feat=np.load(OLD/f'full_input30/input/camera_{c+1:04d}/reid.npy',mmap_mode='r')
  for oid in obsids:
   r=lookup[oid]
   if r['camera']==c:
    v=feat[r['input_row']].copy();vectors[oid]=v/np.linalg.norm(v)
 np.savez_compressed(A/'review_observation_embeddings.npz',**vectors)
 (A/'review_observation_metadata.json').write_text(json.dumps([lookup[oid] for oid in sorted(obsids)],indent=2)+'\n')
 frames={r['frame'] for r in rec}|{int(r['first_inconsistent_observation']['model_frame']) for r in ids.values()}
 snaps=load(OLD/'C_30s/track_snapshots.json');selected=[dict(frame=s['frame'],before=[t for t in s['before'] if t['id'] in tracks],after=[t for t in s['after'] if t['id'] in tracks]) for s in snaps if any(abs(s['frame']-f)<=1 for f in frames)]
 (A/'review_track_snapshots.json').write_text(json.dumps(selected,indent=2)+'\n');print('Review evidence ready',len(vectors),'observation vectors',len(selected),'frame snapshots')
if __name__=='__main__':main()
