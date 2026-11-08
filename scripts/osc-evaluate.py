#!/usr/bin/env python3
"""Explicit, separate 2D and multi-view pilot protocols; no fitting to GT."""
import collections
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import motmetrics as mm
from scipy.spatial.distance import cdist
from scout_adapter.results import FIELDS, canonical_row, normalize_results, export_ground_truth
from scout_adapter.common import write_csv
from osc_pilot import iou_matrix

WORK, OUT = Path('/work'), Path('/evaluation')


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')


def metric(acc, fields):
    return mm.metrics.create().compute(acc, metrics=fields, name='pilot').to_dict(orient='index')['pilot']


def update_2d(acc, gt, pred):
    gb=[[r['bbox_xywh'][0],r['bbox_xywh'][1],r['bbox_xywh'][0]+r['bbox_xywh'][2],r['bbox_xywh'][1]+r['bbox_xywh'][3]] for r in gt]
    pb=[[r['x'],r['y'],r['x']+r['width'],r['y']+r['height']] for r in pred]
    iou=iou_matrix(gb,pb)
    cost=np.where(iou>=.5,1-iou,np.nan)
    acc.update([r['track_id'] for r in gt],[r['native_track_id'] for r in pred],cost)


def self_test():
    g={'track_id':1,'bbox_xywh':[0,0,10,10]}
    p={'native_track_id':7,'x':0,'y':0,'width':10,'height':10}
    acc=mm.MOTAccumulator(auto_id=True);update_2d(acc,[g],[p]);update_2d(acc,[g],[p])
    assert metric(acc,['idf1'])['idf1']==1
    acc=mm.MOTAccumulator(auto_id=True);update_2d(acc,[g],[]);update_2d(acc,[],[p])
    assert metric(acc,['idf1'])['idf1']==0, 'Cross-camera boxes must never match'
    acc=mm.MOTAccumulator(auto_id=True);update_2d(acc,[g],[p]);update_2d(acc,[g],[dict(p,native_track_id=8)])
    assert metric(acc,['idf1'])['idf1']==.5, 'Global identity split must be penalized'
    assert iou_matrix([[0,0,10,10]],[[10,10,20,20]]).item()==0


def main():
    self_test()
    manifest=json.loads((WORK/'inputs/export/manifest.json').read_text())
    cams={c['model_camera_id']:c for c in manifest['cameras']}
    frame_ids=manifest['frame_ids']; start,end=frame_ids[0],frame_ids[-1]
    frame_lookup={(c['id'],f['source_frame']):f for c in cams.values() for f in c['frames']}
    vis_frames=(start,frame_ids[len(frame_ids)//2-1],end)
    gt=json.loads((WORK/'gt_selected.json').read_text())
    truth=collections.defaultdict(list)
    for g in gt:truth[(g['camera_id'],g['source_frame'])].append(g)
    track=WORK/'tracking/repeat_01'
    local, omitted=[],[]
    for cid,cam in cams.items():
        data=json.loads((track/('fixed_camera%03d_tracking_results.json'%cid)).read_text())
        original=json.loads((WORK/('inputs/native/camera%03d.json'%cid)).read_text())
        for serial,r in data.items():
            assert r['Frame']==original[serial]['Frame'] and r['Coordinate']==original[serial]['Coordinate']
            if r['OfflineID']<0:
                omitted.append({'camera':cid,'serial':serial,'stage':'local','reason':'unassigned_local_id'});continue
            b=r['Coordinate']; f=cam['frames'][r['Frame']-1]
            row=canonical_row(manifest,cam,f,r['OfflineID'],[b['x1'],b['y1'],b['x2']-b['x1'],b['y2']-b['y1']],None,None,
                              world_xy=list(r['WorldCoordinate'].values()) if r['WorldCoordinate'] else None,model='yachiyo',scope='camera')
            local.append(row)
    write_csv(OUT/'local_predictions.csv',FIELDS,local)
    whole=json.loads((track/'fixed_whole_tracking_results.json').read_text())
    raw=json.loads((track/'whole_tracking_results.json').read_text())
    global_audit={'raw_rows':sum(len(v) for v in raw.values()),'fixed_rows':sum(len(v) for v in whole.values()),
                  'raw_unassigned_global_rows':sum(r.get('GlobalOfflineID',-1)<0 for v in raw.values() for r in v.values()),
                  'native_correction_removed':[]}
    for cam, rows in raw.items():
        global_audit['native_correction_removed'].extend({'camera':int(cam),'serial':s,'had_global_id':r.get('GlobalOfflineID',-1)>=0} for s,r in rows.items() if s not in whole[cam])
    for cam,rows in whole.items():
        originals=json.loads((WORK/('inputs/native/camera%03d.json'%int(cam))).read_text())
        for serial,r in rows.items():
            assert r['Frame']==originals[serial]['Frame'] and r['Coordinate']==originals[serial]['Coordinate']
            assert r.get('GlobalOfflineID',-1)>=0, 'Unassigned global ID must not be silently invented'
    # Use the pre-existing strict adapter; duplicates or malformed rows fail closed.
    normalize_results(WORK/'inputs/export',track/'fixed_whole_tracking_results.json',OUT/'global_common',
                      'yachiyo-json','sequence')
    with (OUT/'global_common/predictions.csv').open() as stream:
        global_rows=list(csv.DictReader(stream))
    for r in global_rows:
        for k in ('frame','native_track_id'):r[k]=int(r[k])
        for k in ('x','y','width','height'):r[k]=float(r[k])
    for rows in (local,global_rows):
        keys=[(r['camera'],r['frame'],r['native_track_id']) for r in rows]
        assert len(keys)==len(set(keys)), 'Duplicate camera/frame/ID'
        assert all(r['frame'] in frame_ids for r in rows)
    save(OUT/'csv_integrity.json',{'status':'PASS','local_rows':len(local),'global_rows':len(global_rows),
                                 'local_unassigned':omitted,'global_audit':global_audit,
                                 'bbox_frame_preserved':True,'duplicates':0})
    gt_rows=[]
    for g in gt:
        cam=next(c for c in cams.values() if c['id']==g['camera_id'])
        gt_rows.append(canonical_row(manifest,cam,frame_lookup[(cam['id'],g['source_frame'])],g['track_id'],g['bbox_xywh'],g['score'],g['world_xyz'],kind='ground_truth',label_source=g['label_source']))
    write_csv(OUT/'ground_truth.csv',FIELDS,gt_rows)
    fields=['idf1','idp','idr','mota','motp','precision','recall','num_objects','num_predictions','num_matches','num_false_positives','num_misses','num_switches']
    metrics={}
    for label,rows in [('local_scpt',local),('global_mcpt_2d',global_rows)]:
        preds=collections.defaultdict(list)
        for r in rows:preds[(r['camera'],r['frame'])].append(r)
        accs,names=[],[]
        for cam in cams.values():
            acc=mm.MOTAccumulator(auto_id=True)
            for frame in frame_ids:update_2d(acc,truth[(cam['id'],frame)],preds[(cam['id'],frame)])
            events=acc.events.reset_index()
            events['source_frame']=events['FrameId'].map(lambda i:frame_ids[int(i)])
            events.to_csv(OUT/(label+'_'+cam['id']+'_events.csv'),index=False)
            accs.append(acc);names.append(cam['id'])
        metrics[label]=mm.metrics.create().compute_many(accs,names=names,metrics=fields,generate_overall=True).to_dict(orient='index')
    save(OUT/'metrics_2d.json',{'protocol':'Per-camera 2D MOT, IoU>=0.5, all selected frames incl empty; identities assigned independently per camera for OVERALL',
                              'pilot_only':True,'motp_is_one_minus_iou':True,'hota_computed':False,'results':metrics})
    preds=collections.defaultdict(list)
    for r in global_rows:preds[(r['camera'],r['frame'])].append(r)
    acc=mm.MOTAccumulator(auto_id=True)
    for frame in frame_ids:
        for cam in cams.values():update_2d(acc,truth[(cam['id'],frame)],preds[(cam['id'],frame)])
    save(OUT/'metrics_multiview_identity.json',{'protocol':'View-aware global IDF1: independent camera/frame bbox matching, one identity assignment across all views; not official SCOUT world protocol',
         'pilot_only':True,'no_cross_camera_pixel_matching':True,'mota_not_reported_for_view_serialization':True,
         'metrics':metric(acc,['idf1','idp','idr','num_objects','num_predictions'])})
    predicted=json.loads((WORK/'inputs/predicted_world.json').read_text())
    world_lookup={(str(r['camera_id']),r['serial']):r for r in predicted}
    g_world,p_world=collections.defaultdict(list),collections.defaultdict(list)
    for r in gt:
        if r['world_xyz'] is not None:g_world[(r['source_frame'],r['track_id'])].append(r['world_xyz'])
    missing_world=0
    for cam,rows in whole.items():
        for serial,row in rows.items():
            r=world_lookup[(cam,serial)]
            if r['world'] is None:missing_world+=1;continue
            p_world[(r['frame'],row['GlobalOfflineID'])].append(r['world'])
    def by_frame(values):
        grouped=collections.defaultdict(list)
        for (frame,identity),points in values.items():grouped[frame].append((identity,np.median(points,axis=0)))
        return grouped
    g_by,p_by=by_frame(g_world),by_frame(p_world)
    scores={}
    for threshold in (.5,1.,2.):
        acc=mm.MOTAccumulator(auto_id=True)
        for frame in frame_ids:
            gs,ps=g_by[frame],p_by[frame]
            distance=cdist(np.array([p[1][:2] for p in gs]).reshape(-1,2),np.array([p[1][:2] for p in ps]).reshape(-1,2))
            distance[distance>threshold]=np.nan
            acc.update([p[0] for p in gs],[p[0] for p in ps],distance)
        scores[str(threshold)]=metric(acc,fields)
    save(OUT/'metrics_world_diagnostic.json',{'protocol':'Per source frame and ID, median fuse views; Euclidean world XY gating in DATASET UNITS',
        'official_benchmark':False,'physical_units_verified':False,'timestamp_alignment_newly_validated':False,
        'gt_world_source':'SCOUT MOT annotations','missing_world_prediction_observations':missing_world,
        'world_gt_id_frames':len(g_world),'world_pred_id_frames':len(p_world),'threshold_sensitivity':scores})
    vis=OUT/'visualizations';vis.mkdir(exist_ok=True)
    for label,rows in [('local',local),('global',global_rows)]:
        for cid,cam in cams.items():
            for frame in vis_frames:
                img=cv2.imread(str(WORK/'inputs/Images'/cam['id']/('%06d.jpg'%(frame_lookup[(cam['id'],frame)]['index']+1))))
                for row in rows:
                    if row['camera']!=cam['id'] or row['frame']!=frame:continue
                    identity=row['native_track_id'];rng=np.random.RandomState(identity%65536)
                    color=tuple(int(v) for v in rng.randint(50,256,3))
                    x,y,w,h=[row[k] for k in ('x','y','width','height')]
                    cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),color,2)
                    cv2.putText(img,('%s ID %d'%(label,identity)),(int(x),max(15,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
                cv2.imwrite(str(vis/('%s_%s_%d.jpg'%(label,cam['id'],frame))),img)
    save(OUT/'evaluation_report.json',{'status':'PASS','protocol_self_tests':4,'gt_rows':len(gt),'local_rows':len(local),
        'global_rows':len(global_rows),'global_ids':len({r['native_track_id'] for r in global_rows}),
        'source_frame_start':start,'source_frame_end':end,'pilot_only':True,
        'local_idf1':metrics['local_scpt']['OVERALL']['idf1'],
        'multiview_idf1':json.loads((OUT/'metrics_multiview_identity.json').read_text())['metrics']['idf1']})
    print((OUT/'evaluation_report.json').read_text())


if __name__=='__main__':main()
