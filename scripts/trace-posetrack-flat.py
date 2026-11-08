#!/usr/bin/env python3
"""Run unchanged native PoseTrack on rectified local-plane inputs; trace decisions.

Wrappers record native values and verify a separate geometry explanation against
the native C++ outputs. No GT, synthetic detections or forced states are used.
"""
import collections,csv,hashlib,inspect,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/opt/posetrack/track')
import Tracker.PoseTracker as M
from util.camera import Camera
O=Path('/out');FRAME=-1;EVENTS=[];PAIRS=[];BANK_VECTORS={};SNAPS=[];OUTPUT=[];STATELOG=[];GEO=[]
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v)) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in r.items()})
def emit(kind,**kw):EVENTS.append(dict(frame=FRAME,source_frame=FRAME+277,event=kind,**clean(kw)))
def sid(s):return getattr(s,'observation_id',None)
def snapshot(t):
    return {'id':t.id,'state':t.state,'age':t.update_age,'valid_views':list(t.valid_views),'feat_count':t.feat_count,
      'view_states':[list(x.state) for x in t.track2ds],'age_bbox':t.age_bbox.copy(),'dura_bbox':t.dura_bbox.copy(),
      'oc_state':list(t.oc_state),'observation_ids':dict(getattr(t,'observation_ids',{}))}

ORIG_REID=M.PoseTracker.compute_reid_aff
def reid(self,dets,tracks):
    result=ORIG_REID(self,dets,tracks)
    for v,ds in enumerate(dets):
      for i,s in enumerate(ds):
       for j,t in enumerate(tracks):
        states=t.track2ds[v].state
        reason='active'
        if s.bbox[-1]<.88:reason='detection_score_lt_0.88'
        elif not states:reason='empty_state_history'
        elif 2 not in states and 3 not in states:reason='no_Occluded_or_Missing_in_last_10'
        sims=t.feat_bank@s.reid_feat
        if reason=='active' and not np.any(sims>0):reason='no_positive_bank_similarity'
        emit('state_aware_reid',camera=v,observation_id=sid(s),track_id=t.id,score=s.bbox[-1],history=states,
          active=bool(result[1][v][i,j]==1),reason=reason,As=result[0][v][i,j],weight=result[1][v][i,j],
          max_cosine_all_bank=float(sims.max()),bank_count=t.feat_count,threshold=.5)
    self._audit_reid=([a.copy() for a in result[0]],[a.copy() for a in result[1]])
    return result
M.PoseTracker.compute_reid_aff=reid

def plane_point(inv,xy):
    p=inv@np.r_[xy,1.];return p[:2]/p[2]
ORIG_GEO=M.PoseTracker.compute_epi_homo_aff
def geometry(self,dets,tracks):
    result=ORIG_GEO(self,dets,tracks);rays=self.CalcJointRays(dets)
    for v,ds in enumerate(dets):
     for i,s in enumerate(ds):
      for j,t in enumerate(tracks):
        contrib=[];epi_sum=0.;epi_n=1e-5;h_sum=0.;h_n=1e-5
        for vj,cam in enumerate(self.cameras):
          if v==vj or t.age_bbox[vj]>=2:continue
          joints=t.keypoints_mv[vj];valid=(joints[:,2]>.7)&(s.keypoints_2d[:,2]>.7)&(t.age_2D[vj]<=1)
          tr=M.aic_cpp.compute_joints_rays(joints,cam.project_inv,cam.pos)
          cross=np.cross(rays[v][i],tr);dist=np.abs(cross@(self.cameras[v].pos-cam.pos))/(np.linalg.norm(cross,axis=1)+1e-6)
          weights=np.exp(-t.age_2D[vj][valid]);es=float(np.sum((1-dist[valid]/.2)*weights));en=float(weights.sum()+1e-5)
          if es!=0:epi_sum+=es/en;epi_n+=1
          feet=bool((s.keypoints_2d[[15,16],2]>.7).all() and (joints[[15,16],2]>.7).all())
          if feet:
            p1=plane_point(self.cameras[v].homo_feet_inv,s.keypoints_2d[[15,16],:2].mean(0));p2=plane_point(cam.homo_feet_inv,joints[[15,16],:2].mean(0))
          else:
            b=s.bbox;bt=t.bbox_mv[vj];p1=plane_point(self.cameras[v].homo_inv,[(b[0]+b[2])/2,b[3]]);p2=plane_point(cam.homo_inv,[(bt[0]+bt[2])/2,bt[3]])
          hd=float(np.linalg.norm(p1-p2));hs=1-hd/1.5;h_sum+=hs;h_n+=1
          contrib.append({'target_camera':vj,'valid_joint_indices':np.flatnonzero(valid),'ray_distances':dist[valid],
             'joint_age':t.age_2D[vj][valid],'epipolar_contribution':es/en,'foot_mode':'ankle_z0.15' if feet else 'bbox_z0',
             'sample_plane_xy':p1,'track_plane_xy':p2,'ground_distance':hd,'homography_contribution':hs})
        np.testing.assert_allclose([epi_sum/epi_n,h_sum/h_n],[result[0][v][i,j],result[1][v][i,j]],atol=1e-7,rtol=1e-7)
        GEO.append({'frame':FRAME,'camera':v,'observation_id':sid(s),'track_id':t.id,'Aepi_raw':result[0][v][i,j],
           'Ah_raw':result[1][v][i,j],'contributions':contrib})
    self._audit_geo=([a.copy() for a in result[0]],[a.copy() for a in result[1]])
    return result
M.PoseTracker.compute_epi_homo_aff=geometry

ORIG_ASSIGN=M.linear_sum_assignment
def assign(cost):
    caller=inspect.currentframe().f_back;L=caller.f_locals
    ret=ORIG_ASSIGN(cost)
    if caller.f_code.co_name!='mv_update_wo_pred':return ret
    v=L['v'];tracks=L['avail_tracks'];ds=L['detection_sample_list_mv'][v];self=L['self'];rows,cols=ret
    selected=set(zip(rows.tolist(),cols.tolist()));A=-cost
    path=O/f'matrices/frame_{FRAME:03d}_cam_{v}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,Ab=L['aff_box'][v],Aepi_raw=self._audit_geo[0][v],Ah_raw=self._audit_geo[1][v],
       Aepi_clamped=L['aff_epi'][v],Ah_clamped=L['aff_homo'][v],As=L['aff_reid'][v],reid_weight=L['reid_weight'][v],
       norm=L['norm'],final_affinity=A,cost=cost,hungarian_rows=rows,hungarian_cols=cols,
       track_ids=np.array([t.id for t in tracks]),observation_ids=np.array([sid(s) for s in ds]),iou=L['iou_sv'])
    states=[]
    for j,t in enumerate(tracks):
      if j not in cols:
        best=float(L['iou_sv'][:,j].max()) if ds else None
        states.append({'track_id':t.id,'proposed_state':'Occluded' if best is not None and best>.5 else 'Missing','max_iou':best,'column_assigned':False})
    emit('hungarian',camera=v,matrix=str(path.relative_to(O)),track_ids=[t.id for t in tracks],observation_ids=[sid(s) for s in ds],
      rows=rows,cols=cols,unassigned_state_conditions=states)
    for i,s in enumerate(ds):
     for j,t in enumerate(tracks):
      PAIRS.append({'frame':FRAME,'source_frame':FRAME+277,'camera':v,'observation_id':sid(s),'track_id':t.id,
        'Ab':L['aff_box'][v][i,j],'Aepi_raw':self._audit_geo[0][v][i,j],'Ah_raw':self._audit_geo[1][v][i,j],
        'Aepi':L['aff_epi'][v][i,j],'Ah':L['aff_homo'][v][i,j],'As':L['aff_reid'][v][i,j],
        'reid_weight':L['reid_weight'][v][i,j],'norm':L['norm'][i,j],'final_score':A[i,j],
        'hungarian_selected':(i,j) in selected,'accepted_by_score':(i,j) in selected and A[i,j]>0,'iou':L['iou_sv'][i,j]})
    return ret
M.linear_sum_assignment=assign

def bankwrap(name):
    original=getattr(M.PoseTrack,name)
    def wrapped(self,*args,**kwargs):
        before=self.feat_bank.copy();count=self.feat_count;state=getattr(self,'state',None)
        result=original(self,*args,**kwargs)
        changed=np.flatnonzero(np.any(self.feat_bank!=before,axis=1));keys=[]
        for idx in changed:
            key=f'event_{len(EVENTS)}_slot_{idx}';BANK_VECTORS[key]=self.feat_bank[idx].copy();keys.append(key)
        emit('feature_bank',method=name,track_id=self.id,before_count=count,after_count=self.feat_count,changed_slots=changed,
          vector_keys=keys,bank_changed=bool(len(changed)),state_before=state,state_after=self.state)
        return result
    setattr(M.PoseTrack,name,wrapped)
for _name in ['single_view_init','multi_view_init','multi_view_3D_update','reactivate']:bankwrap(_name)

def initwrap(name):
    old=getattr(M.PoseTrack,name)
    def wrapped(self,samples,*args,**kw):
        ss=[samples] if name=='single_view_init' else samples
        result=old(self,samples,*args,**kw);self.observation_ids={s.cam_id:sid(s) for s in ss}
        emit('initialization',method=name,track_id=self.id,observations=self.observation_ids,state=self.state)
        return result
    setattr(M.PoseTrack,name,wrapped)
for _name in ['single_view_init','multi_view_init']:initwrap(_name)
OLD_UPDATE=M.PoseTrack.single_view_2D_update
def update(self,v,s,*args,**kw):
    before=snapshot(self);r=OLD_UPDATE(self,v,s,*args,**kw);self.observation_ids[v]=sid(s)
    emit('accepted_update',track_id=self.id,camera=v,observation_id=sid(s),before=before,after=snapshot(self));return r
M.PoseTrack.single_view_2D_update=update
OLD_REACT=M.PoseTrack.reactivate
def react(self,new):
    r=OLD_REACT(self,new);self.observation_ids=dict(new.observation_ids);emit('reactivation',track_id=self.id,observations=self.observation_ids);return r
M.PoseTrack.reactivate=react
OLD_SWITCH=M.PoseTrack.switch_view
def switch(self,other,v):
    emit('switch_view',track_id=self.id,other_id=other.id,camera=v)
    r=OLD_SWITCH(self,other,v);a=self.observation_ids.get(v);b=other.observation_ids.get(v)
    self.observation_ids[v]=b;other.observation_ids[v]=a;return r
M.PoseTrack.switch_view=switch

OLD_DETECTED=M.PoseTrack2D.init_with_det
def detected(self,sample):
    caller=inspect.currentframe().f_back;L=caller.f_locals;t=L.get('self')
    before=list(self.state);r=OLD_DETECTED(self,sample)
    emit('view_state_detected',track_id=getattr(t,'id',L.get('id')),camera=sample.cam_id,observation_id=sid(sample),
      before_history=before,after_history=self.state,assigned_state=1,reason=caller.f_code.co_name)
    return r
M.PoseTrack2D.init_with_det=detected

# Capture native assignment points and conditions without changing source lines.
SOURCE=Path(M.__file__).read_text().splitlines()
def trace(frame,event,arg):
    if frame.f_code.co_filename!=M.__file__:return None
    if event!='line':return trace
    L=frame.f_locals;line=SOURCE[frame.f_lineno-1].strip();name=frame.f_code.co_name
    if line.startswith('avail_tracks[') and 'state=[state]+' in line:
        index=L['i'] if 'avail_tracks[i]' in line else L['col'];t=L['avail_tracks'][index]
        emit('view_state_assignment',track_id=t.id,camera=L['v'],assigned_state=L['state'],source_line=frame.f_lineno,
          reason=('matched_detection_also_overlaps_unassigned_track' if 'avail_tracks[col]' in line else
             ('unassigned_column_max_iou_gt_0.5' if L['state']==2 else 'unassigned_column_no_iou_gt_0.5')),
          iou_column=L['iou_sv'][:,index])
    if line.startswith('avail_tracks[i]') and 'state=[Track2DState.Missing]+' in line:
        emit('view_state_assignment',track_id=L['avail_tracks'][L['i']].id,camera=L['v'],assigned_state=3,
             source_line=frame.f_lineno,reason='no_detections_in_this_camera',iou_column=[])
    if name=='multi_view_3D_update' and line.startswith('if all(sample.keypoints_2d[self.upper_body'):
        t=L['self'];v=L['v'];s=L['sample'];bank=L['bank'];sim=bank@s.reid_feat
        emit('feature_bank_gate',track_id=t.id,camera=v,upper_body_pass=bool((s.keypoints_2d[t.upper_body,-1]>.5).all()),
          score=s.bbox[4],score_pass=bool(s.bbox[4]>.9),iou_above015=int(np.sum(t.iou_mv[v]>.15)),overlap_above03=int(np.sum(t.ovr_mv[v]>.3)),
          max_cosine=None if not len(sim) else float(sim.max()),novelty_threshold=.6,source_line=frame.f_lineno,
          store_condition=bool((s.keypoints_2d[t.upper_body,-1]>.5).all() and s.bbox[4]>.9 and np.sum(t.iou_mv[v]>.15)<2
            and np.sum(t.ovr_mv[v]>.3)<2 and (t.feat_count==0 or sim.max()<.6)))
    if name=='match_with_miss_tracks' and line=='if reid_sim[t_id]>0.5:':
        emit('missing_reid_decision',new_id=L['new_track'].id,missing_ids=[t.id for t in L['miss_tracks']],similarities=L['reid_sim'],
          best_track_id=L['miss_tracks'][L['t_id']].id,accepted=bool(L['reid_sim'][L['t_id']]>.5),threshold=.5)
    if name=='multi_view_3D_update' and line=='if self_sim > 0.5:':
        emit('occlusion_self_similarity',track_id=L['self'].id,camera=L['v'],As_self=L['self_sim'],keep_own_id=bool(L['self_sim']>.5))
    if name=='multi_view_3D_update' and line=='if reid_sim[max_idx] > self_sim and reid_sim[max_idx]>0.5:':
        emit('occlusion_switch_decision',track_id=L['self'].id,camera=L['v'],self_sim=L['self_sim'],other_ids=[t.id for t in L['oc_tracks']],
          other_similarities=L['reid_sim'],switch=bool(L['reid_sim'][L['max_idx']]>max(L['self_sim'],.5)))
    if name=='multi_view_3D_update' and line=='if (joint_3d[2]<-1 or joint_3d[2]>2.5) or (j_idx in self.feet_idx and (joint_3d[2]<-1 or joint_3d[2]>1)):':
        z=L['joint_3d'][2];t=L['self'];j=L['j_idx']
        emit('height_gate',track_id=t.id,joint=j,z=z,invalid=bool(z<-1 or z>2.5 or (j in t.feet_idx and (z<-1 or z>1))))
    return trace

OLD_BIP=M.GLPKSolver.solve
def solve(self,aff,rtn_matrix=False):
    L=inspect.currentframe().f_back.f_locals;r=OLD_BIP(self,aff,rtn_matrix)
    if 'detection_sample_list_mv' in L:
      ds=L['detection_sample_list_mv'];flat=[sid(s) for ss in ds for s in ss]
      emit('new_track_clustering',observation_ids=flat,clusters=r[0],affinity=aff,geometry_epi=L.get('aff_epi'),geometry_homo=L.get('aff_homo'))
    return r
M.GLPKSolver.solve=solve

def main():
    global FRAME
    cams=[Camera(str(O/f'calibration/camera_{c+1:04d}/calibration.json')) for c in range(8)]
    obs=json.loads((O/'observations.json').read_text());lookup={(r['camera'],r['input_row']):r for r in obs}
    data=[]
    for c in range(8):
        p=O/f'input/camera_{c+1:04d}';d=np.load(p/'detection.npy');po=np.load(p/'pose.npy');f=np.load(p/'reid.npy')
        if len(f):f=f/np.linalg.norm(f,axis=1,keepdims=True)
        assert len(d)==len(po)==len(f);data.append((d,po,f))
    tracker=M.PoseTracker(cams);start=time.monotonic();sys.settrace(trace)
    for FRAME in range(50):
      detections=[]
      for c,(d,po,f) in enumerate(data):
        cur=[]
        for i in np.flatnonzero(d[:,0]==FRAME):
          s=M.Detection_Sample(d[i,2:].copy(),po[i,6:].reshape(17,3).copy(),f[i].copy(),c,FRAME)
          s.observation_id=f'cam{c}_row{lookup[c,i]["camera_row"]}';cur.append(s)
        detections.append(cur)
      before=[snapshot(t) for t in tracker.tracks];tracker.mv_update_wo_pred(detections,FRAME)
      after=[snapshot(t) for t in tracker.tracks];SNAPS.append({'frame':FRAME,'before':before,'after':after})
      for t in tracker.tracks:
        for v in range(8):
          STATELOG.append({'frame':FRAME,'source_frame':FRAME+277,'camera':v,'track_id':t.id,'global_state':t.state,
             'view_history':list(t.track2ds[v].state),'age_bbox':None if not np.isfinite(t.age_bbox[v]) else float(t.age_bbox[v]),'feature_count':t.feat_count})
      for arr in tracker.output(FRAME):
        r=arr[0];tid=int(r[1]);cam=int(r[0])-1;t=next(t for t in tracker.tracks if t.id==tid)
        OUTPUT.append({'frame':FRAME,'source_frame':FRAME+277,'camera':cam,'global_id':tid,'observation_id':t.observation_ids.get(cam),
           'x':r[3],'y':r[4],'w':r[5],'h':r[6],'plane_x':r[7],'plane_y':r[8],'output_method_code':int(r[9]),'bbox_space':'undistorted'})
      print('FRAME',FRAME,'dets',[len(d) for d in detections],'tracks',len(tracker.tracks),'outputs',len(OUTPUT),flush=True)
    sys.settrace(None)
    image_source=Path(M.__file__).read_bytes();host_source=Path('/project/repos/posetrack/track/Tracker/PoseTracker.py').read_bytes()
    assert image_source==host_source
    js(O/'source_verification.json',{'source_sha256':hashlib.sha256(image_source).hexdigest(),
        'image_and_host_tracker_equal':True,'commit':'ba727e6bc6e45e44cd6e006b73e00864892173c2',
        'docker_image':'scout-mtmc/posetrack-all:ba727e6',
        'docker_image_id':'sha256:da99db46dd7d4cfbbd936899c7f3941f065df1725132d62417d0d5194d05bf3d'})
    js(O/'events.json',EVENTS);js(O/'geometry_details.json',GEO);js(O/'track_snapshots.json',SNAPS)
    csvout(O/'affinity_pairs.csv',PAIRS);csvout(O/'state_history.csv',STATELOG);csvout(O/'global_predictions.csv',OUTPUT)
    np.savez_compressed(O/'feature_bank_updates.npz',**BANK_VECTORS)
    counts=collections.Counter(e['event'] for e in EVENTS);reid=[e for e in EVENTS if e['event']=='state_aware_reid']
    js(O/'summary.json',{'status':'PASS','frames':50,'input_rows':len(obs),'output_rows':len(OUTPUT),'output_ids':len({r['global_id'] for r in OUTPUT}),
       'tracks_created':len(tracker.tracks),'event_counts':dict(counts),'affinity_pairs':len(PAIRS),'reid_active_pairs':sum(e['active'] for e in reid),
       'reid_inactive_reasons':dict(collections.Counter(e['reason'] for e in reid if not e['active'])),
       'accepted_hungarian':sum(r['accepted_by_score'] for r in PAIRS),'feature_bank_changed_events':sum(e['event']=='feature_bank' and e['bank_changed'] for e in EVENTS),
       'geometry_cpp_equivalence':'PASS all observed pairs, atol=rtol=1e-7','elapsed_seconds':time.monotonic()-start,
       'gt_used':False,'thresholds_changed':False,'tracking_source_modified':False,'global_id_accuracy_evaluated':False})
    print('DONE',json.loads((O/'summary.json').read_text()),flush=True)
if __name__=='__main__':main()
