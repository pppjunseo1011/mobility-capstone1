#!/usr/bin/env python3
"""Select a mesh-only local plane and rectify cached SCOUT observations; no GT."""
import collections,csv,json,sys
from pathlib import Path
import cv2,numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/project/adapters')
from scout_adapter.geometry import CameraGeometry,GroundMesh,read_ground_ply
R=Path('/project');O=Path('/out');D=Path('/data')
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(p,rs):
    with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def main():
    old=R/'outputs/posetrack-transfer-20261101/handoff'
    diag=[r for r in csv.DictReader((old/'02_adapter/ground_geometry_diagnostic.csv').open()) if r['mesh_x']]
    xyz=np.array([[float(r[k]) for k in ['mesh_x','mesh_y','mesh_z']] for r in diag]);cams=np.array([r['camera'] for r in diag])
    vertices,faces=read_ground_ply(D/'meshes/mesh_ground_only.ply');tree=cKDTree(vertices[:,:2]);candidates=[]
    for side in [6.,8.,10.,12.,16.]:
      centers=np.unique(np.round(xyz[:,:2]/3)*3,axis=0)
      for center in centers:
        lo=center-side/2;hi=center+side/2
        ids=tree.query_ball_point(center,side/2,p=np.inf)
        if len(ids)<100:continue
        vs=vertices[ids];origin=vs.mean(0);_,axes=np.linalg.eigh(np.cov(vs-origin,rowvar=False));normal=axes[:,0]
        if normal[2]<0:normal=-normal
        residual=np.abs((vs-origin)@normal)
        if normal[2]<.98 or np.percentile(residual,95)>.035 or residual.max()>.15:continue
        mask=((xyz[:,:2]>=lo)&(xyz[:,:2]<=hi)).all(1);counts=collections.Counter(cams[mask])
        if len(counts)<2 or min(sorted(counts.values(),reverse=True)[:2])<5:continue
        ex=np.array([1.,0.,0.]);ex-=normal*np.dot(ex,normal);ex/=np.linalg.norm(ex);ey=np.cross(normal,ex)
        W=np.eye(4);W[:3,:3]=np.column_stack([ex,ey,normal]);W[:3,3]=origin
        errs=[]
        for idx in np.flatnonzero(mask):
          row=diag[idx];g=CameraGeometry(json.loads((D/f'calibrations/{row["camera"]}.json').read_text()),1920,1080)
          ray,_=g.ray([float(row['u']),float(row['v'])]);den=np.dot(normal,ray)
          hit=g.center+ray*(np.dot(normal,origin-g.center)/den);errs.append(float(np.linalg.norm(hit-xyz[idx])))
        if np.percentile(errs,95)>.20 or max(errs)>.50:continue
        candidates.append({'bounds_scout_xy':[lo.tolist(),hi.tolist()],'side':side,'mesh_vertices':len(ids),'mesh_p95':float(np.percentile(residual,95)),
          'mesh_max':float(residual.max()),'ray_plane_mesh_p95':float(np.percentile(errs,95)),'ray_plane_mesh_max':max(errs),
          'sample_counts':dict(counts),'sample_total':int(mask.sum()),'W_plane_to_scout':W.tolist()})
    candidates.sort(key=lambda c:(-sum(sorted(c['sample_counts'].values(),reverse=True)[1:]),-c['sample_total'],c['mesh_p95']))
    js(O/'plane_candidates.json',candidates)
    assert candidates,'No suitable flat multiview ROI under declared thresholds'
    choice=candidates[0];W=np.array(choice['W_plane_to_scout']);lo,hi=np.array(choice['bounds_scout_xy']);normal=W[:3,2];origin=W[:3,3]
    js(O/'plane.json',dict(choice,selection='mesh p95<=0.035, max<=0.15; sampled ray-plane p95<=0.20, max<=0.50; maximize multiview observations; NO GT',
       scope='Approximate local plane for internal algorithm tracing, not global SCOUT calibration',coordinate_unit='SCOUT dataset unit, interpreted as metric; no independent survey',
       threshold_policy='All original PoseTrack thresholds and frame-age values retained; no FPS retuning'))
    print('CHOSEN',choice,flush=True)
    mesh=GroundMesh(vertices,faces);records=[];native=old/'03_intermediate/result';allobs=[];exclude=collections.Counter()
    for cam in range(8):
      name=f'camera_{cam+1:04d}';raw=json.loads((D/f'calibrations/cam_{cam}.json').read_text());g=CameraGeometry(raw,1920,1080)
      P=g.k@np.column_stack([g.r,g.t])@W;H=P[:,[0,1,3]]
      cal=O/f'calibration/{name}/calibration.json';js(cal,{'camera projection matrix':P.tolist(),'homography matrix':H.tolist()})
      det=np.loadtxt(native/f'detection/scene_001/{name}.txt',delimiter=',',ndmin=2)
      pose=np.loadtxt(native/f'pose/scene_001/{name}.txt',ndmin=2);feat=np.load(native/f'reid/scene_001/{name}.npy')
      assert len(det)==len(pose)==len(feat)
      keep=[];newdet=[];newpose=[]
      def und(points):return cv2.undistortPointsIter(np.array(points,float).reshape(-1,1,2),g.k,g.dist,None,g.k,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
      for i,(dd,pp) in enumerate(zip(det,pose)):
        x1,y1,x2,y2,score=dd[2:];pixel=np.array([(x1+x2)/2,y2]);reason='outside_plane_roi';rec={'camera':cam,'camera_row':i,'frame':int(dd[0]),'source_frame':277+int(dd[0]),'kept':False,'reason':reason,'mesh_plane_error':None}
        if not (0<=pixel[0]<1920 and 0<=pixel[1]<1080):rec['reason']='outside_image';records.append(rec);continue
        ray,error=g.ray(pixel)
        if ray is None:rec['reason']='undistortion_failed';records.append(rec);continue
        t=np.dot(normal,origin-g.center)/np.dot(normal,ray);p=g.center+t*ray
        if t<=0 or not ((p[:2]>=lo)&(p[:2]<=hi)).all():records.append(rec);continue
        hit=g.ground_point(pixel,mesh)
        if hit['status']!='hit':rec['reason']='mesh_'+hit['status'];records.append(rec);continue
        world=np.array(hit['world']);e=float(np.linalg.norm(world-p));rec['mesh_plane_error']=e
        if not ((world[:2]>=lo)&(world[:2]<=hi)).all() or e>.25:rec['reason']='mesh_roi_or_plane_error';records.append(rec);continue
        kp=pp[6:].reshape(17,3).copy();kp[:,:2]=und(kp[:,:2])
        # Rectified axis-aligned box encloses 17 samples per edge, not just corners.
        tedge=np.linspace(0,1,17);xx=x1+(x2-x1)*tedge;yy=y1+(y2-y1)*tedge
        edge=np.concatenate([np.c_[xx,np.full(17,y1)],np.c_[xx,np.full(17,y2)],np.c_[np.full(17,x1),yy],np.c_[np.full(17,x2),yy]])
        b=und(edge);bbox=np.r_[b.min(0),b.max(0),score]
        # BBox rectification changes its bottom centre; audit resulting ground location.
        q=np.linalg.solve(H,[(bbox[0]+bbox[2])/2,bbox[3],1]);q=q[:2]/q[2];qb=(W@np.r_[q,0,1])[:3]
        shift=float(np.linalg.norm(qb-world))
        if shift>.30:rec['reason']='rectified_bbox_ground_shift';records.append(rec);continue
        rec.update(kept=True,reason='kept');records.append(rec);keep.append(i)
        newdet.append([dd[0],dd[1],*bbox]);newpose.append([dd[0],*bbox,*kp.ravel()])
        allobs.append({'camera':cam,'native_camera':cam+1,'camera_row':i,'input_row':len(keep)-1,'frame':int(dd[0]),'source_frame':277+int(dd[0]),
          'raw_bbox':dd[2:].tolist(),'bbox_undistorted':bbox.tolist(),'keypoints_raw':pp[6:].reshape(17,3).tolist(),
          'mesh_world':world.tolist(),'plane_world':p.tolist(),'mesh_plane_error':e,'rectified_bbox_ground_shift':shift})
      path=O/f'input/{name}';path.mkdir(parents=True,exist_ok=True)
      np.save(path/'detection.npy',np.array(newdet).reshape(-1,7));np.save(path/'pose.npy',np.array(newpose).reshape(-1,57));np.save(path/'reid.npy',feat[keep])
      print('CAM',cam,'kept',len(keep),'of',len(det),flush=True)
    assert allobs
    csvout(O/'selection_audit.csv',records);js(O/'observations.json',allobs)
    counts=collections.Counter(r['camera'] for r in allobs);errors=[r['mesh_plane_error'] for r in allobs]
    js(O/'preparation_summary.json',{'input_rows':len(records),'kept':len(allobs),'counts':dict(counts),'excluded':dict(collections.Counter(r['reason'] for r in records if not r['kept'])),
      'mesh_plane_p95':float(np.percentile(errors,95)),'mesh_plane_max':max(errors),'frame_start':277,'frame_end':326,'gt_used':False,
      'bbox_policy':'dense boundary rectification, axis-aligned envelope; keep only ground shift <=0.30 dataset units',
      'plane_row_gate':0.25,'roi_selected_using_gt':False})
if __name__=='__main__':main()
