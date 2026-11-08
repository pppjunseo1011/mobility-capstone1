#!/usr/bin/env python3
"""Geometry consistency tests independent of GT and tracking accuracy."""
import importlib.util,json,sys,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,'/opt/posetrack/track');sys.path.insert(0,'/project/adapters')
import Tracker.PoseTracker as M
from util.camera import Camera
from scout_adapter.geometry import GroundMesh
spec=importlib.util.spec_from_file_location('mesh_extension','/project/scripts/posetrack-mesh-geometry.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
O=Path('/out');I=O/'full_input'
def main():
    obs=json.loads((I/'observations.json').read_text());cams=[Camera(str(I/f'calibration/camera_{v+1:04d}/calibration.json')) for v in range(8)]
    g=G.install(M,cams,'/data',obs,'B_no_ground');ds=[]
    for v in range(8):
        p=I/f'input/camera_{v+1:04d}';d=np.load(p/'detection.npy');k=np.load(p/'pose.npy');f=np.load(p/'reid.npy')
        ix=np.flatnonzero(d[:,0]==0)[:3];ds.append([M.Detection_Sample(d[i,2:],k[i,6:].reshape(17,3),f[i]/np.linalg.norm(f[i]),v,0) for i in ix])
    tracks=[]
    for j in range(3):
        t=M.PoseTrack(cams)
        for v in range(8):
            s=ds[v][j%len(ds[v])];t.keypoints_mv[v]=s.keypoints_2d;t.bbox_mv[v]=s.bbox;t.age_bbox[v]=j%2;t.age_2D[v]=j%2
        tracks.append(t)
    tracker=M.PoseTracker(cams);native=M.PoseTracker.compute_epi_homo_aff(tracker,ds,tracks);custom=g.affinity(tracker,ds,tracks)
    error=max(float(np.max(np.abs(a-b))) for a,b in zip(native[0],custom[0]));assert error<1e-7
    assert all(np.all(a==0) for a in custom[1])
    g.mode='C_mesh';c=g.affinity(tracker,ds,tracks)
    assert all(np.array_equal(a,b) for a,b in zip(custom[0],c[0]))
    ground_rows=[]
    for r in obs:
        v=r['camera'];uv=r['bottom_undistorted'];p=g.point(v,*map(float,uv),0.)
        # Rectify ankles using the same stored input, without consulting GT.
        k=np.load(I/f'input/camera_{v+1:04d}/pose.npy',mmap_mode='r')[r['input_row'],6:].reshape(17,3)
        valid=bool((k[[15,16],2]>.7).all());a=g.ground(v,np.array(r['bbox_undistorted']),k,True) if valid else None
        ground_rows.append({'camera':v,'source_frame':r['source_frame'],'observation_id':f"cam{v}_row{r['camera_row']}",
          'bbox_mesh_status':'hit' if p is not None else 'no_mesh_intersection','ankle_confidence_pass':valid,
          'ankle_mesh_status':'not_selected_low_confidence' if not valid else ('hit_lifted_surface' if a is not None else 'no_mesh_intersection'),
          'bbox_ground_xyz':None if p is None else p.tolist(),'ankle_ground_xyz':None if a is None else a.tolist(),
          'ankle_world_xyz':None if a is None else (a+[0,0,.15]).tolist(),'ankle_lift_world_z':.15})
    with (O/'geometry_observations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(ground_rows[0]));w.writeheader();w.writerows(ground_rows)
    # Exact flat mesh test at nonzero z, with ankle offset: use arbitrary camera.
    vs=np.array([[-100,-100,2],[100,-100,2],[100,100,2],[-100,100,2]],float)
    mesh=GroundMesh(vs,np.array([[0,1,2],[0,2,3]]));origin=np.array([0.,0.,10.]);ray=np.array([1.,2.,-4.]);lift=.15
    hit=mesh.intersect(origin-[0,0,lift],ray)['world'];expected=origin+ray*((2+lift-10)/ray[2])-[0,0,lift]
    np.testing.assert_allclose(hit,expected,atol=1e-10)
    # Original ROI's local-plane positions vs mesh in the same SCOUT frame.
    old=json.loads(Path('/project/outputs/posetrack-flat-20261101/observations.json').read_text())
    lookup={(r['camera'],r['camera_row']):r for r in obs};errs=[]
    for r in old:
        q=lookup[r['camera'],r['camera_row']]
        assert q['mesh_world'] is not None
        np.testing.assert_allclose(q['mesh_world'],r['mesh_world'],atol=1e-6)
        errs.append(np.linalg.norm(np.array(q['mesh_world'])-r['plane_world']))
    report={'status':'PASS','epipolar_native_Cpp_max_abs_error':error,'epipolar_B_C_identical':True,
      'nonzero_plane_ankle_offset_test':'PASS, contact differs from ankle by 0.15 world Z',
      'ROI_reused_mesh_points_equal':len(old),'ROI_mesh_vs_plane_distance_p95':float(np.percentile(errs,95)),
      'ROI_mesh_vs_plane_distance_max':float(max(errs)),
      'GT_used':False,'limits':'numerical consistency only; mesh geometry accuracy and ankle offset physical validity not proved'}
    patched=O/'C_final/PoseTracker.patched.py'
    if patched.exists():
        ps=importlib.util.spec_from_file_location('patched_tracker_verification',patched);PM=importlib.util.module_from_spec(ps);ps.loader.exec_module(PM)
        a,b=PM.PoseTrack(cams),PM.PoseTrack(cams);a.id=1;b.id=2
        for field in ['age_2D','keypoints_mv','bbox_mv']:
            getattr(a,field)[0]=1;getattr(b,field)[0]=2
        a.switch_view(b,0)
        for field in ['age_2D','keypoints_mv','bbox_mv']:
            assert np.all(getattr(a,field)[0]==2) and np.all(getattr(b,field)[0]==1)
        report['numpy_view_swap_regression']='PASS for age_2D, keypoints_mv, bbox_mv; distinct rows preserved'
    (O/'geometry_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
