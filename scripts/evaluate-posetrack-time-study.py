#!/usr/bin/env python3
"""Common raw-pixel, aligned-camera evaluation. GT used only after tracking."""
import argparse,collections,csv,importlib.util,json,math
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('metrics',Path(__file__).with_name('scout-common-evaluation-metrics.py'));E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
def read(p):return list(csv.DictReader(p.open()))
def save(p,x):E.save(p,x)
def write(p,rs):E.csvout(p,rs)
def keycam(x):return int(str(x).replace('cam_',''))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path);ap.add_argument('--native-run',type=Path);ap.add_argument('--observations',type=Path)
 ap.add_argument('--frame-map',type=Path,default=Path('/out/frame_mapping.csv'));ap.add_argument('--gt',type=Path,default=Path('/project/outputs/osc-analysis-20261101/s30_th40/gt_selected.json'))
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--config',required=True);ap.add_argument('--model',default='PoseTrack');ap.add_argument('--allow-native-source-remap',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 mapping=read(a.frame_map);valid={(int(r['camera']),int(r['source_frame'])):r for r in mapping if r['status']=='selected'};n=max(int(r['common_time_index']) for r in mapping)+1
 common=[];excluded=0
 if a.native_run:
  obs=json.loads(a.observations.read_text());lookup={f"cam{r['camera']}_row{r['camera_row']}":r for r in obs}
  for r in read(a.native_run/'global_predictions.csv'):
   o=lookup[r['observation_id']];assert int(r['camera'])==o['camera'];bb=o['bbox_undistorted']
   np.testing.assert_allclose([float(r[k]) for k in ['x','y','w','h']],[bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]],atol=1e-7)
   k=(o['camera'],o['source_frame']);m=valid.get(k)
   if m is None:excluded+=1;continue
   idx=int(m['common_time_index'])
   if not a.allow_native_source_remap:assert int(r['frame'])==idx and int(r['source_frame'])==o['source_frame']
   b=o['raw_bbox'];common.append(dict(sequence_id='sequence_1',camera_id=f"cam_{o['camera']}",source_frame_id=o['source_frame'],
    timestamp_seconds=float(m['timestamp_seconds']),common_time_index=idx,common_timestamp_seconds=float(m['common_timestamp_seconds']),
    x=b[0],y=b[1],width=b[2]-b[0],height=b[3]-b[1],global_id=int(r['global_id']),local_id='',detection_confidence=b[4],
    model_name=a.model,experiment_configuration_id=a.config,observation_id=r['observation_id'],world_x=r['plane_x'],world_y=r['plane_y'],world_method=r['output_method_code']))
  write(a.out/'common_predictions.csv',common)
 else:common=read(a.predictions)
 rows=[];seen=set()
 for r in common:
  assert r['sequence_id']=='sequence_1';cam=keycam(r['camera_id']);src=int(r['source_frame_id']);m=valid[cam,src];idx=int(r['common_time_index'])
  assert idx==int(m['common_time_index']);assert abs(float(r['timestamp_seconds'])-float(m['timestamp_seconds']))<1e-8
  bbox=[float(r[k]) for k in ['x','y','width','height']];assert np.isfinite(bbox).all() and min(bbox[2:])>0
  rows.append(dict(camera=cam+1,frame=idx,id=int(r['global_id']),bbox=bbox,score=float(r['detection_confidence']) if r['detection_confidence'] else -1))
 truth=collections.defaultdict(list);truth_export=[]
 for g in json.loads(a.gt.read_text()):
  cam=keycam(g['camera_id']);m=valid.get((cam,int(g['source_frame'])))
  if m:
   idx=int(m['common_time_index']);truth[cam+1,idx].append(dict(id=g['track_id'],bbox=g['bbox_xywh'],world=g['world_xyz']));truth_export.append(dict(g,common_time_index=idx,timestamp_seconds=float(m['timestamp_seconds'])))
 metrics,detail,removed=E.metric_suite(rows,truth,n);ids=detail['combined_2D']['Identity'];gi=detail['Identity_view_aware']
 metrics.update(IDP_2D=float(ids['IDP']),IDR_2D=float(ids['IDR']),IDP_view_aware=float(gi['IDP']),IDR_view_aware=float(gi['IDR']),
  input_observations=None if not a.observations else sum((r['camera'],r['source_frame']) in valid for r in json.loads(a.observations.read_text())),
  output_rows=len(rows),gt_rows=sum(len(x) for x in truth.values()),available_camera_time_slots=len(valid),total_camera_time_slots=8*n,excluded_original_output_rows=excluded,config=a.config)
 if metrics['input_observations']:metrics['output_fraction']=len(rows)/metrics['input_observations'];metrics['IDSW_per_1000_output']=metrics['IDSW_2D']/len(rows)*1000
 cam_metrics=[]
 for cam in range(8):
  data=E.metric_data(rows,truth,[(cam+1,f) for f in range(n)]);h=E.HOTA().eval_sequence(data);i=E.Identity({'PRINT_CONFIG':False}).eval_sequence(data);c=E.CLEAR({'PRINT_CONFIG':False}).eval_sequence(data)
  cam_metrics.append(dict(camera=f'cam_{cam}',input_observations=None if not a.observations else sum(r['camera']==cam and (r['camera'],r['source_frame']) in valid for r in json.loads(a.observations.read_text())),
    output_rows=sum(r['camera']==cam+1 for r in rows),global_ids=len({r['id'] for r in rows if r['camera']==cam+1}),GT_rows=data['num_gt_dets'],
    HOTA=float(np.mean(h['HOTA'])),DetA=float(np.mean(h['DetA'])),AssA=float(np.mean(h['AssA'])),IDF1=float(i['IDF1']),IDP=float(i['IDP']),IDR=float(i['IDR']),IDSW=int(c['IDSW'])))
 write(a.out/'per_camera_metrics.csv',cam_metrics);save(a.out/'metrics.json',dict(summary=metrics,detail=detail,duplicate_removals=removed))
 matches=E.gt_matches(rows,truth);evidence=[]
 for j,r in enumerate(common):
  m=matches.get(j);evidence.append(dict(r,matched_gt_id='' if m is None else m['gt']['id'],gt_iou='' if m is None else m['iou']))
 write(a.out/'gt_matching_results.csv',evidence);save(a.out/'selected_gt.json',truth_export)
 save(a.out/'protocol.json',dict(sequence='sequence_1',cameras=list(range(8)),grid_count=n,valid_camera_time_slots=len(valid),
  GT_usage='posthoc only',camera_missing='No image/GT invented for unavailable capture slots; same explicit valid mask for all runs. Valid images with zero detections retain GT and count FN.',
  bbox='original distorted image pixels xywh',HOTA='TrackEval 0.05..0.95 IoU thresholds; camera-local sequences combined',
  IDF1_IDSW='TrackEval IoU>=0.5; IDSW sum per-camera temporal CLEAR',shared_IDF1='shared identity namespace, separate camera/time steps; NOT official SCOUT 3D',
  interpretation='Association ablation with reused PoseTrack neural inputs. End-to-end comparison with other neural front ends must be labeled separately.',
  original_comparison='Original output remapped/filtered to SAME valid slots and GT; legacy full-source-frame score retained separately',config=a.config))
 print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
