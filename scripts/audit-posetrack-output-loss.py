#!/usr/bin/env python3
"""Disjoint observation-level output ledger for original five-second Mesh run."""
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];OLD=R/'outputs/posetrack-mesh-20261104';P=OLD/'C_final';O=R/'outputs/posetrack-time-study-20261106/association'
def read(p):return list(csv.DictReader(p.open()))
def main():
 global P
 ap=argparse.ArgumentParser();ap.add_argument('--run-directory',type=Path,default=P);ap.add_argument('--input-directory',type=Path,default=OLD/'full_input');ap.add_argument('--tag',default='original_5s');args=ap.parse_args();P=args.run_directory;input_root=args.input_directory
 obs=json.loads((input_root/'observations.json').read_text());byid={f"cam{r['camera']}_row{r['camera_row']}":r for r in obs};out={r['observation_id'] for r in read(P/'global_predictions.csv')}
 ev=json.loads((P/'events.json').read_text());snap={r['frame']:r['after'] for r in json.loads((P/'track_snapshots.json').read_text())};clusters={};accepted={};initial={}
 for e in ev:
  if e['event']=='accepted_update':accepted[e['observation_id']]=e['track_id']
  if e['event']=='initialization':
   for oid in e['observations'].values():initial[oid]=e['track_id']
  if e['event']=='new_track_clustering':
   for cl in e['clusters']:
    ids=[e['observation_ids'][i] for i in cl]
    for oid in ids:clusters[oid]=ids
 matrix={};keypoints={}
 for c in range(8):keypoints[c]=np.load(input_root/f'input/camera_{c+1:04d}/pose.npy')[:,6:].reshape(-1,17,3)
 frameobs=collections.defaultdict(list)
 for oid,r in byid.items():frameobs[r['camera'],r['frame']].append(oid)
 def gate(oid):
  r=byid[oid];key=(r['camera'],r['frame'])
  if key not in matrix:
   z=np.load(P/f'matrices/frame_{key[1]:03d}_cam_{key[0]}.npz');ids=z['observation_ids'].tolist();iou=z['iou'];b=np.array([byid[x]['bbox_undistorted'][:4] for x in ids]);area=np.prod(b[:,2:]-b[:,:2],axis=1)
   inter=np.maximum(0,np.minimum(b[:,None,2:],b[None,:,2:])-np.maximum(b[:,None,:2],b[None,:,:2])).prod(2)
   ovr=inter/area[:,None] if iou.size else np.zeros((len(ids),len(ids)));matrix[key]=(ids,iou,ovr)
  ids,iou,ovr=matrix[key];i=ids.index(oid);kp=keypoints[r['camera']][r['input_row']]
  return {'confidence':bool(r['raw_bbox'][4]>.9),'pose_main_joints':bool((kp[[5,6,11,12,13,14,15,16],2]>.5).all()),
   'existing_track_iou':bool(np.sum(iou[i]>.15)<1),'detection_overlap':bool(np.sum(ovr[i]>.3)<2)}
 rows=[]
 for oid,r in byid.items():
  reason='output';tid=accepted.get(oid,initial.get(oid));gates=None
  if oid not in out:
   if tid is not None:
    tracks=[t for t in snap[r['frame']] if oid in t['observation_ids'].values()]
    t=next((t for t in tracks if t['observation_ids'].get(str(r['camera']))==oid),None)
    if t is None:reason='accepted_observation_replaced_or_target_absent'
    elif t['age_bbox'][r['camera']] is None:reason='geometry_height_correction_removed_view'
    elif t['state']==1:reason='unconfirmed_target_warmup'
    elif t['state']==3:reason='target_missing'
    elif t['state']==4:reason='target_deleted'
    else:reason='final_output_gate_or_view_not_valid'
   elif oid in clusters:
    gates={x:gate(x) for x in clusters[oid]};eligible=list(gates.values());reason='unresolved_initialization'
    for name in ['confidence','pose_main_joints','existing_track_iou','detection_overlap']:
     eligible=[g for g in eligible if g[name]]
     if not eligible:reason='no_cluster_seed_'+name;break
    # All gate reconstructions must explain failure, not invent an upstream drop.
    assert reason!='unresolved_initialization',(oid,gates)
   else:reason='unresolved_not_accepted_or_clustered'
  rows.append(dict(observation_id=oid,camera=r['camera'],source_frame=r['source_frame'],frame=r['frame'],output=oid in out,
   direct_terminal_reason=reason,accepted_or_initial_track=tid,initialization_gate_evidence=json.dumps(gates),
   bbox_mesh_hit=r['mesh_world'] is not None))
 with (O/(args.tag+'_output_ledger.csv')).open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 counts=collections.Counter(r['direct_terminal_reason'] for r in rows);assert counts['output']==len(out) and len(rows)==len(obs)
 if args.tag=='original_5s':assert counts['output']==4265 and len(rows)-counts['output']==1304
 summary={'input':len(rows),'output':counts['output'],'no_output':len(rows)-counts['output'],'disjoint_terminal_reasons':dict(counts),
  'upstream_detection_filters':'All provided observations were already detector-selected; no additional upstream filtering by mesh adapter in this run',
  'interpretation':'A failed Hungarian edge is not a terminal loss if initialization can recover it. Cluster seed failures are assigned first failed gate in native AND order; full flags retained. Warmup is normal native filtering, not automatically a bug. Mesh ray miss and rejected height are separate.',
  'verification':'Every original observation has one ledger row, sums reconcile; reconstructed seed gates use saved IoU matrices and exact overlap formula, no GT'}
 (O/(args.tag+'_output_loss_summary.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
