#!/usr/bin/env python3
"""Read-only OSC forensics; replay native local correction and audit every exclusion."""
import collections
import copy
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

sys.path.insert(0,'/model')
import scpt
import mcpt
import pose
from osc_pilot import iou_matrix
from osc_evaluate import update_2d, metric

W=Path('/work');O=Path('/analysis');O.mkdir(exist_ok=True)
def load(p):return json.loads((W/p).read_text())
def save(name,obj):(O/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
    if not rows:return
    with (O/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def counts(data):
    ids=collections.Counter(r['OfflineID'] for r in data.values() if r['OfflineID']>=0)
    return {'rows':len(data),'assigned':sum(ids.values()),'ids':len(ids),'singletons':sum(v==1 for v in ids.values())}

manifest=load('inputs/export/manifest.json');frames=manifest['frame_ids']
params=load('tracking/repeat_01/settings.json')['parameters']
reps=load('tracking/repeat_01/representative_nodes_scene1.json')
rawglobal=load('tracking/repeat_01/whole_tracking_results.json')
fixedglobal=load('tracking/repeat_01/fixed_whole_tracking_results.json')
gt=collections.defaultdict(list)
for row in load('gt_selected.json'):gt[(row['camera_id'],row['source_frame'])].append(row)
stage_rows=[];tracklets=[];removals=[];similarities=[];events=[];camera_metrics=[]
length_hist={};failure_cases=[]

for cam in manifest['cameras']:
    cid=cam['model_camera_id'];name=cam['id'];key=str(cid)
    raw=load('tracking/repeat_01/camera%03d_tracking_results.json'%cid)
    fixed=load('tracking/repeat_01/fixed_camera%03d_tracking_results.json'%cid)
    snms=scpt.sequential_non_maximum_suppression(copy.deepcopy(raw),**params)
    warp=scpt.separate_warp_tracklet(copy.deepcopy(snms),**params)
    assert warp==fixed, 'Native correction replay differs'
    for stage,rows in [('raw_scpt',raw),('after_sequential_nms',snms),('after_warp',warp)]:
        stage_rows.append(dict(camera=name,stage=stage,**counts(rows)))
    length_hist[name]=dict(collections.Counter(len(v['all_serials']) for v in reps[key].values()))
    for lid,v in reps[key].items():
        n=len(v['all_serials']);score=v['representative_node']['score']
        fs=[fixed[s]['Frame'] for s in v['all_serials']]
        tracklets.append(dict(camera=name,local_id=int(lid),length=n,span=max(fs)-min(fs)+1,
            start_source=cam['frames'][min(fs)-1]['source_frame'],end_source=cam['frames'][max(fs)-1]['source_frame'],
            pose_condition=score,short_fail=n<params['short_track_th'],pose_fail=score>params['keypoint_condition_th'],
            sequential_rejection='length' if n<params['short_track_th'] else 'pose' if score>params['keypoint_condition_th'] else 'eligible',
            representative_serial=v['representative_node']['serial']))
    pk=pose.PoseKeypoints(str(W/('inputs/Pose/scene_001/camera_%04d/camera_%04d_out_keypoint.json'%(cid,cid))))
    pk.assign_serial_from_tracking_dict(rawglobal[key])
    for serial,row in rawglobal[key].items():
        if serial in fixedglobal[key]:continue
        had='GlobalOfflineID' in row
        lid=row['OfflineID']
        if not had:
            if lid<0:reason='local_unassigned_after_snms'
            else:
                v=reps[key][str(lid)]
                reason='short_track' if len(v['all_serials'])<params['short_track_th'] else 'representative_pose'
                assert reason=='short_track' or v['representative_node']['score']>params['keypoint_condition_th']
        else:
            kp=pk.get_keypoints(serial);b=row['Coordinate'];width=b['x2']-b['x1'];height=b['y2']-b['y1']
            if kp is None:reason='post_missing_pose'
            elif width/height>3 or height/width>5:reason='post_aspect_ratio'
            else:
                noise=mcpt.eval_noise_level(kp['Keypoints'])
                assert noise>=2 and not (noise==2 and min(width,height)<100)
                reason='post_pose_noise_'+str(noise)
        removals.append(dict(camera=name,serial=serial,local_id=lid,global_id=row.get('GlobalOfflineID',-1),
                             source_frame=cam['frames'][row['Frame']-1]['source_frame'],had_global_id=had,reason=reason))

    # Evaluate the three observed local stages with exactly the same motmetrics matching.
    for stage,data in [('raw_scpt',raw),('after_sequential_nms',snms),('after_warp',fixed)]:
        acc=mm.MOTAccumulator(auto_id=True);by=collections.defaultdict(list)
        for serial,row in data.items():
            if row['OfflineID']<0:continue
            b=row['Coordinate'];src=cam['frames'][row['Frame']-1]['source_frame']
            by[src].append(dict(native_track_id=row['OfflineID'],x=b['x1'],y=b['y1'],width=b['x2']-b['x1'],height=b['y2']-b['y1']))
        for f in frames:update_2d(acc,gt[(name,f)],by[f])
        m=metric(acc,['idf1','num_switches','recall','mota'])
        camera_metrics.append(dict(camera=name,stage=stage,**{k:float(v) for k,v in m.items()}))
        if stage=='after_warp':
            prior={};chosen=0
            for (fi,ei),ev in acc.events.iterrows():
                if ev['Type'] not in ('MATCH','SWITCH'):continue
                gid,hid=int(ev['OId']),int(ev['HId']);src=frames[int(fi)]
                if ev['Type']=='SWITCH':
                    previous=prior.get(gid)
                    item=dict(camera=name,gt_id=gid,source_frame=src,new_local_id=hid,
                              previous_source=previous[0] if previous else None,previous_local_id=previous[1] if previous else None)
                    events.append(item)
                    if previous and chosen<2:
                        panels=[]
                        for source,pid in ((previous[0],previous[1]),(src,hid)):
                            f=cam['frames'][source-frames[0]]
                            image=cv2.imread('/inference/input/'+name+'/frame_%06d.jpg'%f['index'])
                            assert image is not None
                            for truth in gt[(name,source)]:
                                if truth['track_id']==gid:
                                    x,y,w,h=map(int,truth['bbox_xywh']);cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
                            for pred in by[source]:
                                if pred['native_track_id']==pid:
                                    x,y,w,h=[int(pred[k]) for k in ('x','y','width','height')];cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
                            image=cv2.resize(image,(960,540));cv2.rectangle(image,(0,0),(960,45),(0,0,0),-1)
                            cv2.putText(image,f'{name} frame {source} GT {gid} Local {pid}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                            panels.append(image)
                        path='failure_%s_%d.jpg'%(name,chosen)
                        cv2.imwrite(str(O/path),np.concatenate(panels,axis=0))
                        failure_cases.append(dict(item,image=path));chosen+=1
                prior[gid]=(src,hid)

    # Independent IoU matching for appearance diagnostics, not tracking input or evaluation substitution.
    byframe=collections.defaultdict(list)
    for serial,row in raw.items():byframe[row['Frame']].append((serial,row))
    last={}
    for model_frame,items in sorted(byframe.items()):
        source=cam['frames'][model_frame-1]['source_frame'];truth=gt[(name,source)]
        gb=[[g['bbox_xywh'][0],g['bbox_xywh'][1],g['bbox_xywh'][0]+g['bbox_xywh'][2],g['bbox_xywh'][1]+g['bbox_xywh'][3]] for g in truth]
        boxes=[[r['Coordinate'][k] for k in ('x1','y1','x2','y2')] for _,r in items]
        iou=iou_matrix(gb,boxes);ii,jj=linear_sum_assignment(np.where(iou>=.5,1-iou,1e6))
        for i,j in zip(ii,jj):
            if iou[i,j]<.5:continue
            gid=truth[i]['track_id'];serial,row=items[j]
            feature=np.load(row['NpyPath']).reshape(-1)
            if gid in last and model_frame-last[gid][0]==1:
                previous,prev_feature,prev_id=last[gid];cos=float(np.dot(feature,prev_feature)/(np.linalg.norm(feature)*np.linalg.norm(prev_feature)))
                similarities.append(dict(camera=name,gt_id=gid,source_frame=source,cosine=cos,below_scpt_gate=cos<.9,
                    raw_id_changed=prev_id!=row['OfflineID'],period_boundary=(model_frame//30!=(model_frame-1)//30)))
            last[gid]=(model_frame,feature,row['OfflineID'])
    print('FORENSICS',name,flush=True)

csvsave('local_stage_counts.csv',stage_rows);csvsave('tracklets.csv',tracklets)
csvsave('global_row_exclusions.csv',removals);csvsave('local_stage_metrics.csv',camera_metrics)
csvsave('idsw_events.csv',events);csvsave('adjacent_gt_appearance.csv',similarities)
save('length_histogram.json',length_hist);save('failure_cases.json',failure_cases)
camera_summary={}
for cam in manifest['cameras']:
    name=cam['id'];t=[r for r in tracklets if r['camera']==name];removed=[r for r in removals if r['camera']==name]
    camera_summary[name]={'tracklets':len(t),'max_length':max(r['length'] for r in t),
        'length_bins':dict(collections.Counter('1' if r['length']==1 else '2' if r['length']==2 else '3-9' if r['length']<10 else '10-19' if r['length']<20 else '20-29' if r['length']<30 else '30-39' if r['length']<40 else '40+' for r in t)),
        'sequential_rejection':dict(collections.Counter(r['sequential_rejection'] for r in t)),
        'joint_short_pose':dict(collections.Counter(str(r['short_fail'])+'_'+str(r['pose_fail']) for r in t)),
        'row_exclusions':dict(collections.Counter(r['reason'] for r in removed))}
summary={'native_correction_replay_exact':True,'local_ids_total':len(tracklets),
         'raw_local_ids_total':sum(r['ids'] for r in stage_rows if r['stage']=='raw_scpt'),
         'global_input_rows':sum(len(v) for v in rawglobal.values()),
         'raw_global_assigned':sum('GlobalOfflineID' in r for v in rawglobal.values() for r in v.values()),
         'global_unassigned':sum(not r['had_global_id'] for r in removals),
         'post_removed_assigned':sum(r['had_global_id'] for r in removals),
         'final_global_rows':sum(len(v) for v in fixedglobal.values()),
         'row_exclusions':dict(collections.Counter(r['reason'] for r in removals)),
         'camera_summary':camera_summary,'appearance_pairs':len(similarities),
         'appearance_below_gate':sum(r['below_scpt_gate'] for r in similarities),
         'raw_id_changed_pairs':sum(r['raw_id_changed'] for r in similarities),
         'raw_changed_and_below_gate':sum(r['raw_id_changed'] and r['below_scpt_gate'] for r in similarities),
         'total_idsw':len(events),'caution':'IoU association is diagnostic, not guaranteed same-person crop correspondence or causal proof.'}
assert summary['global_input_rows']==summary['global_unassigned']+summary['post_removed_assigned']+summary['final_global_rows']
save('summary.json',summary)
print(json.dumps(summary,indent=2))
