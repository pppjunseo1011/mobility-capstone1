#!/usr/bin/env python3
"""Audit measured mesh failures without GT or replacing absent intersections."""
import collections,csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/opt/posetrack/track');import Tracker.PoseTracker as M
from util.camera import Camera
spec=importlib.util.spec_from_file_location('geo','/project/scripts/posetrack-mesh-geometry.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
O=Path('/out');OLD=Path('/project/outputs/posetrack-mesh-20261104')
def main():
 obs=json.loads((OLD/'full_input30/observations.json').read_text());cams=[Camera(str(OLD/f'full_input30/calibration/camera_{c+1:04d}/calibration.json')) for c in range(8)];g=G.install(M,cams,'/data',obs,'C_mesh')
 poses=[np.load(OLD/f'full_input30/input/camera_{c+1:04d}/pose.npy')[:,6:].reshape(-1,17,3) for c in range(8)];rows=[];counts=collections.Counter()
 for r in obs:
  c=r['camera'];b=np.array(r['bbox_undistorted']);k=poses[c][r['input_row']];p=g.ground(c,b,k,False);eligible=bool((k[[15,16],2]>.7).all());a=g.ground(c,b,k,True) if eligible else None
  uv=r['bottom_undistorted'];ray=g.inv[c]@np.r_[uv,1.];ray/=np.linalg.norm(ray);lo,hi=g.mesh.nodes[0][:2]
  reason='hit' if p is not None else ('ray_misses_mesh_bounding_volume' if g.mesh._box_distance(cams[c].pos,ray,lo,hi,float('inf')) is None else 'ray_crosses_bounding_volume_but_no_surface_triangle')
  rows.append(dict(camera=c,source_frame=r['source_frame'],observation_id=f"cam{c}_row{r['camera_row']}",bbox_status=reason,
   ankle_confidence_pass=eligible,ankle_hit=a is not None,bbox_failed_but_ankle_hit=p is None and a is not None,
   bbox_ground_xyz=None if p is None else p.tolist(),ankle_ground_xyz=None if a is None else a.tolist(),ray_z=float(ray[2]),ray_norm=float(np.linalg.norm(ray))))
  counts[c,reason]+=1
 out=O/'geometry';out.mkdir(exist_ok=True)
 with (out/'mesh_intersections_all.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 failures=[r for r in rows if r['bbox_status']!='hit']
 with (out/'mesh_intersection_failures.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(failures)
 summary={'original_30s_observations':len(rows),'bbox_failures':len(failures),'bbox_failures_first5s':sum(r['source_frame']<=326 for r in failures),
  'per_camera':[dict(camera=c,input=sum(r['camera']==c for r in rows),bbox_failures=sum(r['camera']==c for r in failures),
    bbox_failed_ankle_available=sum(r['camera']==c and r['bbox_failed_but_ankle_hit'] for r in rows),failure_reasons={k[1]:v for k,v in counts.items() if k[0]==c}) for c in range(8)],
  'classification_limit':'Bounding-volume/surface classification is numerical, not proof of physical cause. No triangle hit may reflect mesh holes, ground occlusion, off-ground box bottom, or calibration error; do not assert a cause without image/survey evidence.',
  'existing_policy':'ankle if both views confidence>.7, then bbox fallback if needed; no hit contributes neutral ground term. No mesh coordinate fabricated.',
  'height_policy_issue':'finite triangulated joint without vertical mesh coverage was labeled invalid height; absence of ground reference is not measured invalid height. Candidate ablation abstains ONLY in this case, keeps finite-height thresholds and rejects nonfinite joints.', 'GT_used':False}
 (out/'mesh_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 # Audit prior rejected heights supporting the controlled candidate.
 ev=json.loads((OLD/'C_30s/mesh_events.json').read_text());unknown=[r for r in ev if r['event']=='mesh_height_gate' and r['ground_z'] is None]
 finite=[r for r in unknown if np.isfinite(r['world_xyz']).all()]
 (out/'old_unknown_height_events.json').write_text(json.dumps(finite,indent=2)+'\n');print('UNKNOWN_HEIGHT',len(unknown),'FINITE',len(finite))
if __name__=='__main__':main()
