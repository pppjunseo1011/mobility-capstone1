#!/usr/bin/env python3
"""Audit actual PoseTrack module outputs and SCOUT geometry, without tracker/GT.

Run in the existing PoseTrack image. Candidate homographies are diagnostics,
never native calibration.json. Output files are separate from source data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, '/project/adapters')
from scout_adapter.geometry import CameraGeometry, GroundMesh, read_ground_ply


def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csvout(path, rows):
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def stats(a):
    a = np.asarray(a, float)
    return {'n': len(a), 'median': float(np.median(a)) if len(a) else None,
            'p95': float(np.percentile(a, 95)) if len(a) else None,
            'max': float(np.max(a)) if len(a) else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    inf, out = a.inference, a.out
    det = json.loads((inf/'posetrack/detection/detections.json').read_text())
    poses = {x['key']: x for x in json.loads((inf/'posetrack/pose/poses.json').read_text())}
    rows = json.loads((inf/'posetrack/reid/aic24-rows.json').read_text())
    features = np.load(inf/'posetrack/reid/aic24.npy', allow_pickle=False)
    assert features.shape == (len(rows), 2048) and np.isfinite(features).all()
    norms = np.linalg.norm(features, axis=1)
    assert (norms > 0).all()
    rowidx = {(x['key'], x['box_index']): i for i, x in enumerate(rows)}
    assert len(rowidx) == len(rows)
    frames, alignment, identities, audits, diagnostics = [], [], [], [], []
    cammap = []
    points_by_cam = {}
    for c in range(8):
        cam = f'cam_{c}'; native = f'camera_{c+1:04d}'
        cammap.append({'sequence':'sequence_1', 'scout_camera':cam, 'native_folder':native,
                       'native_camera_id':c+1, 'tracker_view_index':c})
        ds = sorted([d for d in det if d['camera'] == cam], key=lambda d:d['index'])
        assert len(ds) == 50 and [d['source_frame'] for d in ds] == list(range(277,327))
        nd = np.loadtxt(inf/f'posetrack/pose/{cam}-det7.txt', delimiter=',', ndmin=2)
        np_ = np.loadtxt(inf/f'posetrack/pose/{cam}-pose57.txt', ndmin=2)
        ids, point_records = [], []
        offset = 0
        for d in ds:
            po = poses[d['key']]
            boxes = np.asarray(d['selected'], float).reshape(-1,7)[:, :5]
            kp = np.asarray(po['keypoints'], float).reshape(-1,17,3)
            assert len(kp) == len(boxes)
            np.testing.assert_allclose(boxes, np.asarray(po['bbox_xyxy_score']).reshape(-1,5))
            frames.append({k: d[k] for k in ['camera','model_camera_id','index','source_frame','timestamp_seconds','source_path','sha256']})
            for j, (box, keypoints) in enumerate(zip(boxes, kp)):
                ii = rowidx[(d['key'],j)]; rr = rows[ii]
                assert rr['camera'] == cam and rr['source_frame'] == d['source_frame']
                np.testing.assert_allclose(nd[offset], [d['index'],1,*box], atol=1e-5)
                np.testing.assert_allclose(np_[offset], [d['index'],*box,*keypoints.ravel()], atol=1e-5)
                assert np.isfinite(keypoints).all()
                ids.append(ii)
                identities.append({'camera':cam,'native_camera':native,'source_frame':d['source_frame'],
                    'internal_frame':d['index'],'detection_index':j,'camera_row':offset,'embedding_global_row':ii,
                    'x1':box[0],'y1':box[1],'x2':box[2],'y2':box[3],'objectness':box[4],
                    'pose_valid_joints_gt_05':int((keypoints[:,2]>.5).sum()),'embedding_dim':2048,
                    'embedding_norm':float(norms[ii])})
                if d['index'] in [0,10,20,30,40,49]:
                    point_records.append({'source_frame':d['source_frame'],'internal_frame':d['index'],
                        'detection_index':j,'camera_row':offset,'pixel':[(box[0]+box[2])/2,box[3]]})
                offset += 1
        assert offset == len(nd) == len(np_)
        for typ, src in [('detection',inf/f'posetrack/pose/{cam}-det7.txt'),('pose',inf/f'posetrack/pose/{cam}-pose57.txt')]:
            dst=out/f'03_intermediate/result/{typ}/scene_001/{native}.txt'
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        dst=out/f'03_intermediate/result/reid/scene_001/{native}.npy'
        dst.parent.mkdir(parents=True,exist_ok=True)
        # Preserve native raw features. Native tracking loader normalizes them.
        np.save(dst,features[ids], allow_pickle=False)
        np.testing.assert_array_equal(np.load(dst,allow_pickle=False),features[ids])
        counts=[len(d['selected']) for d in ds]
        alignment.append({'camera':cam,'native_camera':native,'frames':len(ds),'rows':offset,
            'max_people_per_frame':max(counts),'min_people_per_frame':min(counts),
            'det_cols':7,'pose_cols':57,'reid_dim':2048,'nan_inf':0,'alignment_status':'PASS'})
        points_by_cam[cam]=point_records
    assert len(identities)==len(rows)==5569
    js(out/'02_adapter/camera_map.json',cammap)
    js(out/'02_adapter/frame_map.json',frames)
    csvout(out/'03_intermediate/alignment_audit.csv',alignment)
    csvout(out/'03_intermediate/row_correspondence.csv',identities)
    skew=[]
    for i in range(50):
        ts=[f['timestamp_seconds'] for f in frames if f['index']==i]
        skew.append({'internal_frame':i,'source_frame':i+277,'max_skew_seconds':max(ts)-min(ts)})
    csvout(out/'02_adapter/timestamp_audit.csv',skew)
    print('ALIGNMENT PASS',len(identities),flush=True)

    # One common plane fitted only to mesh vertices, never person GT.
    vertices, faces = read_ground_ply(a.data/'meshes/mesh_ground_only.ply')
    center=vertices.mean(axis=0)
    _, axes=np.linalg.eigh(np.cov(vertices-center,rowvar=False))
    normal=axes[:,0]
    if normal[2]<0: normal=-normal
    ex=np.array([1.,0.,0.]);ex-=normal*np.dot(ex,normal);ex/=np.linalg.norm(ex)
    ey=np.cross(normal,ex)
    W=np.eye(4);W[:3,:3]=np.column_stack([ex,ey,normal]);W[:3,3]=center
    residual=np.abs((vertices-center)@normal)
    js(out/'02_adapter/plane_diagnostic.json',{'status':'BLOCKED','ready_for_tracking':False,
        'meaning':'Mesh-only common PCA plane; NOT a validated ground model or calibration.json',
        'plane_world_to_scout':W.tolist(),'normal':normal.tolist(),
        'absolute_residual_dataset_units':stats(residual),'metric_scale_independently_verified':False})
    mesh=GroundMesh(vertices,faces)
    for c in range(8):
        cam=f'cam_{c}'
        raw=json.loads((a.data/f'calibrations/{cam}.json').read_text())
        g=CameraGeometry(raw,1920,1080)
        P=g.k@np.column_stack([g.r,g.t])
        H0=P[:,[0,1,3]]
        Pplane=P@W;Hplane=Pplane[:,[0,1,3]]
        js(out/f'02_adapter/raw_calibrations/{cam}.json',raw)
        js(out/f'02_adapter/candidate_matrices_DO_NOT_USE/{cam}.json',{
            'status':'BLOCKED','ready_for_tracking':False,'image_domain':'undistorted pixels only',
            'P_scout_world_3x4':P.tolist(),'H_z0_candidate_3x3':H0.tolist(),
            'P_plane_world_3x4':Pplane.tolist(),'H_mesh_pca_candidate_3x3':Hplane.tolist(),
            'note':'Diagnostic candidates, no native calibration.json emitted; raw images retain lens distortion.'})
        cr=[]
        for pr in points_by_cam[cam]:
            pix=np.array(pr['pixel'],float)
            rec={k:v for k,v in pr.items() if k!='pixel'}
            rec.update(camera=cam,u=float(pix[0]),v=float(pix[1]),status='',
                mesh_x=None,mesh_y=None,mesh_z=None,distortion_shift_px=None,
                z0_vs_mesh_world_error=None,pca_vs_mesh_world_error=None,
                raw_z0_vs_mesh_world_error=None,pca_surface_projection_error_px=None,
                mesh_roundtrip_px=None)
            if not (0<=pix[0]<1920 and 0<=pix[1]<1080):
                rec['status']='bbox_bottom_outside_image';cr.append(rec);continue
            hit=g.ground_point(pix,mesh)
            rec['status']=hit['status']
            if hit['world'] is not None:
                world=np.array(hit['world'])
                direction,_=g.ray(pix);local=g.r@direction; ideal=g.k@local;ideal=ideal[:2]/ideal[2]
                wh=np.r_[world,1.]
                uv=P@wh;uv=uv[:2]/uv[2]
                np.testing.assert_allclose(uv,ideal,atol=1e-4)
                def unproject(H,xy):
                    x=np.linalg.solve(H,np.r_[xy,1.]);return x[:2]/x[2]
                z0=np.r_[unproject(H0,ideal),0.]
                pca=(W@np.r_[unproject(Hplane,ideal),0.,1.])[:3]
                raw0=np.r_[unproject(H0,pix),0.]
                wp=np.linalg.solve(W,wh);wp[2]=0.;onplane=(W@wp)[:3]
                plane_projection=g.project(onplane)
                rec.update(mesh_x=float(world[0]),mesh_y=float(world[1]),mesh_z=float(world[2]),
                    distortion_shift_px=float(np.linalg.norm(ideal-pix)),
                    z0_vs_mesh_world_error=float(np.linalg.norm(z0-world)),
                    pca_vs_mesh_world_error=float(np.linalg.norm(pca-world)),
                    raw_z0_vs_mesh_world_error=float(np.linalg.norm(raw0-world)),
                    pca_surface_projection_error_px=float(np.linalg.norm(plane_projection-pix)),
                    mesh_roundtrip_px=float(hit['reprojection_error_px']))
            cr.append(rec)
        diagnostics.extend(cr)
        valid=[r for r in cr if r['mesh_x'] is not None]
        au={'camera':cam,'status':'BLOCKED','sampled_bbox_points':len(cr),'mesh_hits':len(valid),
            'P_rank':int(np.linalg.matrix_rank(P)),'H_z0_invertible':bool(np.linalg.matrix_rank(H0)==3),
            'H_pca_invertible':bool(np.linalg.matrix_rank(Hplane)==3),'H_z0_condition':float(np.linalg.cond(H0))}
        for k in ['distortion_shift_px','z0_vs_mesh_world_error','pca_vs_mesh_world_error','raw_z0_vs_mesh_world_error','pca_surface_projection_error_px','mesh_roundtrip_px']:
            for q,v in stats([r[k] for r in valid]).items():au[k+'_'+q]=v
        audits.append(au)
        print('GEOMETRY',cam,len(cr),len(valid),au['pca_vs_mesh_world_error_p95'],flush=True)
    csvout(out/'02_adapter/calibration_audit.csv',audits)
    csvout(out/'02_adapter/ground_geometry_diagnostic.csv',diagnostics)
    js(out/'02_adapter/audit_summary.json',{'module_alignment':'PASS','native_tracking_readiness':'BLOCKED',
        'detections':len(identities),'frames':len(frames),'max_camera_skew_seconds':max(x['max_skew_seconds'] for x in skew),
        'geometry_sample_rule':'all detection bbox bottoms at internal frames 0,10,20,30,40,49; no top-k',
        'geometry_points':len(diagnostics),'geometry_hits':sum(r['mesh_x'] is not None for r in diagnostics),
        'reference':'SCOUT ground mesh, NOT independent person world GT',
        'gt_used':False,'tracking_run':False,
        'blockers':['Raw pixel coordinates retain lens distortion absent from native P/H.',
                    'SCOUT ground is nonplanar; z=0 and mesh PCA plane are diagnostic candidates only.',
                    'Metric scale and 0.15 / 1.5 native physical thresholds not independently validated.']})
    print('DONE: native tracking BLOCKED; module alignment PASS',flush=True)


if __name__=='__main__': main()
