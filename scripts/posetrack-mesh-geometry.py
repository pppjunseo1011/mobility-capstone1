"""SCOUT geometry extension. No GT; B disables ground affinity, C uses mesh.

P and rays remain in the original SCOUT world frame. Ground distances use 3D
Euclidean dataset units; 1.5 is retained as an un-tuned original threshold.
Ankles intersect a surface lifted 0.15 along world Z, then return ground contact.
This is a model assumption, not measured ankle height. Bbox rays use the actual
raw bottom centre after rectification, not the envelope's displaced centre.
"""
import collections, functools, importlib.util, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/project/adapters')
from scout_adapter.geometry import GroundMesh, read_ground_ply

class MeshGeometry:
    def __init__(self,cameras,data,observations,mode):
        self.cameras=cameras;self.mode=mode;self.events=[];self.counts=collections.Counter();self.frame=-1
        vs,fs=read_ground_ply(Path(data)/'meshes/mesh_ground_only.ply');self.mesh=GroundMesh(vs,fs);self.top=float(vs[:,2].max()+10)
        self.bottom={(r['camera'],tuple(r['bbox_undistorted'][:4])):np.array(r['bottom_undistorted']) for r in observations}
        self.inv=[np.linalg.inv(c.project_mat[:,:3]) for c in cameras]

    @functools.lru_cache(maxsize=150000)
    def point(self,v,u,y,lift=0.):
        cam=self.cameras[v];ray=self.inv[v]@np.array([u,y,1.]);ray/=np.linalg.norm(ray)
        origin=cam.pos-np.array([0.,0.,lift]);hit=self.mesh.intersect(origin,ray)
        self.counts['ray_query']+=1
        if hit is None:self.counts['no_mesh_hit']+=1;return None
        p=hit['world'];lifted=p+np.array([0.,0.,lift]);q=cam.project_mat@np.r_[lifted,1.]
        if not np.isfinite(p).all() or q[2]<=0 or np.linalg.norm(q[:2]/q[2]-[u,y])>1e-4:
            self.counts['invalid_reprojection']+=1;return None
        return np.asarray(p)

    def ground(self,v,b,k,ankle):
        if ankle:uv=k[[15,16],:2].mean(0);lift=.15
        else:uv=self.bottom.get((v,tuple(b[:4])),np.array([(b[0]+b[2])/2,b[3]]));lift=0.
        return self.point(v,float(uv[0]),float(uv[1]),lift)

    def pair(self,vi,bi,ki,vj,bj,kj):
        ankle=bool((ki[[15,16],2]>.7).all() and (kj[[15,16],2]>.7).all())
        a=self.ground(vi,bi,ki,ankle);b=self.ground(vj,bj,kj,ankle);kind='ankle_lift_0.15' if ankle else 'bbox_ground'
        if (a is None or b is None) and ankle:
            a=self.ground(vi,bi,ki,False);b=self.ground(vj,bj,kj,False);kind='bbox_after_ankle_miss'
        if a is None or b is None:return None,kind
        return float(np.linalg.norm(a-b)),kind

    @functools.lru_cache(maxsize=150000)
    def floor(self,x,y):
        hit=self.mesh.intersect([x,y,self.top],[0,0,-1])
        return None if hit is None else float(hit['world'][2])

    def invalid_height(self,track,joint,j):
        z=None
        if np.isfinite(joint).all():z=self.floor(float(joint[0]),float(joint[1]))
        h=None if z is None else float(joint[2]-z)
        bad=h is None or h < -1 or h > (1 if j in track.feet_idx else 2.5)
        self.events.append({'event':'mesh_height_gate','frame':self.frame,'track_id':track.id,'joint':j,
          'world_xyz':joint[:3].tolist(),'ground_z':z,'relative_height':h,'rejected':bad,
          'reason':'no_vertical_mesh_hit_or_invalid_joint' if h is None else 'relative_height_bounds'})
        return bad

    def output(self,track):
        for comb in track.output_priority:
            if all(track.age_3D[comb]==0):
                p=np.mean(track.keypoints_3d[comb,:3],axis=0)
                if np.isfinite(p).all() and self.floor(float(p[0]),float(p[1])) is not None:
                    track.output_cord=np.r_[p[:2],3];return track.output_cord
        for ankle in [True,False]:
            for v in track.valid_views:
                k=track.keypoints_mv[v]
                if ankle and not ((k[[15,16],2]>.7).all() and (track.age_2D[v][[15,16]]==0).all()):continue
                p=self.ground(v,track.bbox_mv[v],k,ankle)
                if p is not None:
                    track.output_cord=np.r_[p[:2],2 if ankle else 1];return track.output_cord
        self.events.append({'event':'output_world_unavailable','frame':self.frame,'track_id':track.id})
        track.output_cord=np.array([np.nan,np.nan,0.]);return track.output_cord

    def affinity(self,tracker,dets,tracks):
        # Same ray-line distance, confidence/age masks and decay as native C++.
        # No H evaluation is needed in this extension.
        rays=tracker.CalcJointRays(dets);ep=[];hg=[]
        target_rays={(j,v):tracker_module.aic_cpp.compute_joints_rays(t.keypoints_mv[v],c.project_inv,c.pos)
                     for j,t in enumerate(tracks) for v,c in enumerate(self.cameras) if t.age_bbox[v]<2}
        for v,ds in enumerate(dets):
            e=np.zeros((len(ds),len(tracks)));h=e.copy()
            for i,s in enumerate(ds):
                for j,t in enumerate(tracks):
                    es=0.;en=1e-5;hs=0.;hn=1e-5
                    for w,cam in enumerate(self.cameras):
                        if w==v or t.age_bbox[w]>=2:continue
                        k=t.keypoints_mv[w];mask=(k[:,2]>.7)&(s.keypoints_2d[:,2]>.7)&(t.age_2D[w]<=1)
                        cross=np.cross(rays[v][i],target_rays[j,w]);distance=np.abs(cross@(self.cameras[v].pos-cam.pos))/(np.linalg.norm(cross,axis=1)+1e-6)
                        decay=np.exp(-t.age_2D[w][mask]);total=float(np.sum((1-distance[mask]/tracker.thred_epi)*decay))
                        if total!=0:es+=total/(decay.sum()+1e-5);en+=1
                        if self.mode=='C_mesh':
                            d,kind=self.pair(v,s.bbox,s.keypoints_2d,w,t.bbox_mv[w],k)
                            if d is not None:hs+=1-d/tracker.thred_homo;hn+=1
                            self.counts['affinity_'+('valid' if d is not None else 'missing')]+=1
                    e[i,j]=es/en;h[i,j]=hs/hn
            ep.append(e);hg.append(h)
        return ep,hg

    def init_affinities(self,tracker,ds):
        counts=[len(s) for s in ds];offset=np.r_[0,np.cumsum(counts)];n=int(offset[-1]);e=np.full((n,n),-10000.);h=e.copy();rays=tracker.CalcJointRays(ds)
        for v in range(8):
            for w in range(v+1,8):
                for i,a in enumerate(ds[v]):
                    for j,b in enumerate(ds[w]):
                        valid=(a.keypoints_2d[:,2]>.7)&(b.keypoints_2d[:,2]>.7);ids=np.flatnonzero(valid)
                        scores=np.zeros(17)
                        scores[ids]=tracker_module.epipolar_3d_score_norm(self.cameras[v].pos,rays[v][i][ids],self.cameras[w].pos,rays[w][j][ids],tracker.thred_epi)
                        weights=a.keypoints_2d[:,2]*b.keypoints_2d[:,2]
                        e[offset[v]+i,offset[w]+j]=np.sum(scores*weights)/(np.sum(valid*weights)+1e-5)
                        d,kind=self.pair(v,a.bbox,a.keypoints_2d,w,b.bbox,b.keypoints_2d) if self.mode=='C_mesh' else (None,'disabled')
                        h[offset[v]+i,offset[w]+j]=0 if d is None else 1-d/tracker.thred_homo
                        self.counts['init_'+kind]+=1
        return e,h

def install(M,cameras,data,observations,mode):
    global tracker_module
    tracker_module=M;geo=MeshGeometry(cameras,data,observations,mode)
    for c in cameras:c.scout_geometry=geo
    return geo
