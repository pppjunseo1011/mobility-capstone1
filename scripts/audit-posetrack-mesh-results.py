#!/usr/bin/env python3
"""Post-run integrity/event summaries; never influences tracking decisions."""
import collections,csv,json,math,shutil
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-mesh-20261104'
def main():
    audit={}
    for name in ['B_final','C_final','C_30s']:
        p=O/name
        if not (p/'metrics.json').exists():continue
        rows=list(csv.DictReader((p/'predictions_raw.csv').open()))
        world_invalid=sum(not all(r[k] and math.isfinite(float(r[k])) for k in ['world_x','world_y']) for r in rows)
        assert world_invalid==json.loads((p/'metrics.json').read_text())['summary']['missing_world_rows']
        assert all(math.isfinite(float(r[k])) for r in rows for k in ['x','y','width','height','confidence'])
        assert all(float(r['width'])>0 and float(r['height'])>0 for r in rows)
        events=json.loads((p/'events.json').read_text());kinds=collections.Counter(r['event'] for r in events)
        selected=[r for r in events if r['event'] in ['missing_reid_decision','reactivation','switch_view','occlusion_self_similarity']]
        (p/'identity_lifecycle_events.json').write_text(json.dumps(selected,indent=2)+'\n')
        solver=[json.loads(x) for x in (p/'solver.jsonl').read_text().splitlines()]
        assert all(x['status']==0 and (x['mip_gap'] is None or x['mip_gap']<=1e-9) for x in solver)
        mesh=json.loads((p/'mesh_events.json').read_text());reasons=collections.Counter(x['reason'] for x in mesh if x['event']=='mesh_height_gate' and x['rejected'])
        audit[name]={'output_rows':len(rows),'world_invalid_rows':world_invalid,'all_bbox_finite_positive':True,
          'event_counts':dict(kinds),'accepted_missing_reid_decisions':sum(x['event']=='missing_reid_decision' and x.get('accepted',False) for x in selected),
          'solver_calls':len(solver),'all_solver_calls_optimal':True,'rejected_height_reasons':dict(reasons),
          'lifecycle_note':'events report algorithm actions, not GT-correct re-identification; detailed state/bank analysis deferred'}
    timeline=list(csv.DictReader((R/'outputs/uwipl-study-20261101/timestamp_audit.csv').open()))
    shutil.copy2(R/'outputs/uwipl-study-20261101/timestamp_audit.csv',O/'timestamp_audit.csv')
    audit['time_alignment']={str(n):{'frames':n,'max_cross_camera_skew_seconds':max(float(x['camera_skew_seconds']) for x in timeline[:n]),
      'frames_with_skew_over_50ms':sum(float(x['camera_skew_seconds'])>.05 for x in timeline[:n])} for n in [50,300]}
    # Best single sloped plane: distinguish height offset from non-planarity.
    with (R/'datasets/scout/extracted/scout/meshes/mesh_ground_only.ply').open('rb') as f:
        count=None
        while True:
            line=f.readline().decode('ascii').strip()
            if line.startswith('element vertex '):count=int(line.split()[-1])
            if line=='end_header':break
        vertices=np.frombuffer(f.read(count*12),dtype='<f4').reshape(-1,3).astype(float)
    a=np.c_[vertices[:,:2],np.ones(len(vertices))];coef=np.linalg.lstsq(a,vertices[:,2],rcond=None)[0]
    residual=np.abs(a@coef-vertices[:,2])/np.sqrt(1+np.dot(coef[:2],coef[:2]))
    audit['global_plane_fit']={'vertices':len(vertices),'z_equals_ax_by_c':coef.tolist(),
      'orthogonal_residual_median':float(np.median(residual)),'orthogonal_residual_p95':float(np.percentile(residual,95)),
      'orthogonal_residual_max':float(residual.max()),'note':'Mesh vertex fit only; unweighted by triangle area; not GT error'}
    samples=[]
    for cam in range(8):
        p=O/f'full_input/input/camera_{cam+1:04d}';d=np.load(p/'detection.npy');k=np.load(p/'pose.npy')[d[:,0]==0,6:].reshape(-1,17,3)
        P=np.array(json.loads((O/f'full_input/calibration/camera_{cam+1:04d}/calibration.json').read_text())['camera projection matrix'])
        center=-np.linalg.solve(P[:,:3],P[:,3]);rays=np.einsum('ij,nkj->nki',np.linalg.inv(P[:,:3]),np.concatenate([k[:,:,:2],np.ones(k.shape[:2]+(1,))],axis=2))
        rays/=np.linalg.norm(rays,axis=2,keepdims=True);samples.append((center,k,rays))
    distances=[]
    for i,(ci,ki,ri) in enumerate(samples):
        for cj,kj,rj in samples[i+1:]:
            cross=np.cross(ri[:,None],rj[None]);dist=np.abs(cross@(ci-cj))/(np.linalg.norm(cross,axis=3)+1e-6)
            distances.extend(dist[(ki[:,None,:,2]>.7)&(kj[None,:,:,2]>.7)].tolist())
    audit['epipolar_distance_first_frame']={'confident_joint_pairs':len(distances),'median':float(np.median(distances)),
      'p95':float(np.percentile(distances,95)),'max':float(max(distances)),
      'note':'All cross-camera detection candidates at source frame 277; contains different people, not a localization accuracy metric; SCOUT coordinate units'}
    (O/'execution_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
