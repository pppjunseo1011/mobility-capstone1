#!/usr/bin/env python3
"""Rectify every cached observation, audit SCOUT calibration; never reads GT."""
import argparse, collections, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from scout_adapter.geometry import CameraGeometry, GroundMesh, read_ground_ply

def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def stats(x):
    a=np.asarray(x,float)
    return dict(n=len(a), median=float(np.median(a)), p95=float(np.percentile(a,95)), max=float(a.max())) if len(a) else dict(n=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();O=args.out;O.mkdir(parents=True,exist_ok=True)
    vs,fs=read_ground_ply(args.data/'meshes/mesh_ground_only.ply');mesh=GroundMesh(vs,fs)
    records=[];audit=[];hashes={}
    for c in range(8):
        name=f'camera_{c+1:04d}';cal=args.data/f'calibrations/cam_{c}.json'
        g=CameraGeometry(json.loads(cal.read_text()),1920,1080);P=g.k@np.column_stack([g.r,g.t]);H=P[:,[0,1,3]]
        save(O/f'calibration/{name}/calibration.json',{'camera projection matrix':P.tolist(),'homography matrix':H.tolist()})
        paths=[args.native/f'{stage}/scene_001/{name}{ext}' for stage,ext in [('detection','.txt'),('pose','.txt'),('reid','.npy')]]
        for p in [cal,*paths]:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        det=np.loadtxt(paths[0],delimiter=',',ndmin=2);pose=np.loadtxt(paths[1],ndmin=2);feat=np.load(paths[2]);assert len(det)==len(pose)==len(feat)
        assert np.isfinite(det).all() and np.isfinite(pose).all() and np.isfinite(feat).all()
        assert np.all(np.linalg.norm(feat,axis=1)>0)
        outdet=[];outpose=[];errors=[];plane_errors=[];raw_ideal=[];ray_errors=[]
        def und(p):return cv2.undistortPointsIter(np.array(p,float).reshape(-1,1,2),g.k,g.dist,None,g.k,(3,100,1e-12)).reshape(-1,2)
        for i,(dd,pp) in enumerate(zip(det,pose)):
            assert dd[0]==pp[0];np.testing.assert_allclose(dd[2:],pp[1:6],atol=1e-5)
            x1,y1,x2,y2,score=dd[2:];kp=pp[6:].reshape(17,3).copy();original=kp[:,:2].copy();kp[:,:2]=und(original)
            tt=np.linspace(0,1,17);xx=x1+(x2-x1)*tt;yy=y1+(y2-y1)*tt
            edge=np.concatenate([np.c_[xx,np.full(17,y1)],np.c_[xx,np.full(17,y2)],np.c_[np.full(17,x1),yy],np.c_[np.full(17,x2),yy]])
            b=und(edge);bbox=np.r_[b.min(0),b.max(0),score];assert bbox[2]>bbox[0] and bbox[3]>bbox[1]
            ideal=(np.linalg.inv(g.k)@np.c_[kp[:,:2],np.ones(17)].T).T
            re=cv2.projectPoints(ideal.reshape(-1,1,3),np.zeros(3),np.zeros(3),g.k,g.dist)[0].reshape(-1,2)
            errors.extend(np.linalg.norm(re-original,axis=1));raw_ideal.extend(np.linalg.norm(kp[:,:2]-original,axis=1))
            # Camera-native pseudoinverse rays must describe the same lines as R.T K^-1.
            q=np.linalg.pinv(P)@np.c_[kp[:,:2],np.ones(17)].T;q=q[:3]/q[3]-g.center[:,None];q/=np.linalg.norm(q,axis=0)
            true=g.r.T@ideal.T;true/=np.linalg.norm(true,axis=0)
            ray_errors.extend(np.linalg.norm(np.cross(q.T,true.T),axis=1))
            raw_bottom=np.array([(x1+x2)/2,y2]);bottom=und([raw_bottom])[0]
            # Boxes clipped to x2=width/y2=height have a valid boundary ray even
            # though the strict raw-pixel adapter excludes this coordinate.
            ray=g.r.T@np.linalg.solve(g.k,np.r_[bottom,1.]);mh=mesh.intersect(g.center,ray)
            hit={'status':'no_ground_intersection','world':None,'reprojection_error_px':None}
            if mh is not None:
                error=float(np.linalg.norm(g.project(mh['world'])-raw_bottom));assert error<1e-3
                hit.update(status='hit',world=mh['world'].tolist(),reprojection_error_px=error)
            hp=np.linalg.solve(H,np.r_[bottom,1]);hp=hp[:2]/hp[2]
            err=None
            if hit['status']=='hit':
                err=float(np.linalg.norm(hp-np.array(hit['world'])[:2]));plane_errors.append(err)
            rec={'camera':c,'native_camera':c+1,'camera_row':i,'input_row':i,'frame':int(dd[0]),'source_frame':277+int(dd[0]),
                 'raw_bbox':dd[2:].tolist(),'bbox_undistorted':bbox.tolist(),'keypoints_raw':pp[6:].reshape(17,3).tolist(),
                 'bottom_undistorted':bottom.tolist(),'mesh_world':hit['world'],'mesh_status':hit['status'],
                 'mesh_reprojection_error_px':hit.get('reprojection_error_px'),'z0_plane_xy':hp.tolist(),'z0_mesh_xy_error':err,
                 'outside_image_bottom':bool(not(0<=raw_bottom[0]<1920 and 0<=raw_bottom[1]<1080))}
            records.append(rec);outdet.append([dd[0],dd[1],*bbox]);outpose.append([dd[0],*bbox,*kp.ravel()])
        p=O/f'input/{name}';p.mkdir(parents=True,exist_ok=True)
        np.save(p/'detection.npy',np.array(outdet).reshape(-1,7));np.save(p/'pose.npy',np.array(outpose).reshape(-1,57));np.save(p/'reid.npy',feat)
        a={'camera':c,'rows':len(det),'center_world':g.center.tolist(),'det_R':float(np.linalg.det(g.r)),
           'R_orthogonality_error':float(np.linalg.norm(g.r@g.r.T-np.eye(3))),'K_condition':float(np.linalg.cond(g.k)),
           'H_z0_condition':float(np.linalg.cond(H)),'redistortion_error_px':stats(errors),'distortion_displacement_px':stats(raw_ideal),
           'ray_line_cross_norm':stats(ray_errors),'z0_mesh_xy_error_dataset_units':stats(plane_errors)}
        assert max(errors)<1e-3 and max(ray_errors)<1e-6
        audit.append(a);print('PREPARED',c,len(det),flush=True)
    save(O/'observations.json',records);save(O/'calibration_audit.json',audit);save(O/'input_sha256.json',hashes)
    save(O/'preparation_summary.json',{'input_rows':len(records),'counts':dict(collections.Counter(r['camera'] for r in records)),
      'mesh_status':dict(collections.Counter(r['mesh_status'] for r in records)), 'roi_filter':False,'gt_used':False,
      'mesh_z_range':[float(vs[:,2].min()),float(vs[:,2].max())],
      'bbox_policy':'dense boundary rectification; all finite input rows retained, original raw bottom ray preserved separately',
      'interpretation':'numerical calibration consistency, not independent physical calibration accuracy'})
if __name__=='__main__':main()
