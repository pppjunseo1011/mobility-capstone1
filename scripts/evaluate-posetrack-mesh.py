#!/usr/bin/env python3
"""Posthoc only: map output to original observations and evaluate B/C equally."""
import argparse,collections,csv,importlib.util,json
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('metrics','/project/scripts/evaluate-uwipl-study.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
def csvout(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/out'));ap.add_argument('--run',required=True);ap.add_argument('--input',required=True);ap.add_argument('--frames',type=int,default=50);a=ap.parse_args()
    work=a.root/a.run;observations=json.loads((a.root/a.input/'observations.json').read_text());obs={f"cam{x['camera']}_row{x['camera_row']}":x for x in observations}
    source=list(csv.DictReader((work/'global_predictions.csv').open()));raw=[];rows=[]
    for r in source:
        o=obs[r['observation_id']];assert int(r['source_frame'])==o['source_frame'] and int(r['camera'])==o['camera']
        expected=o['bbox_undistorted'];np.testing.assert_allclose([float(r['x']),float(r['y']),float(r['w']),float(r['h'])],
          [expected[0],expected[1],expected[2]-expected[0],expected[3]-expected[1]],atol=1e-7)
        b=o['raw_bbox'];bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]]
        raw.append({'sequence':'sequence_1','camera':f"cam_{o['camera']}",'source_frame':o['source_frame'],'model_frame':o['frame'],
          'local_id':'','global_id':int(r['global_id']),'x':bbox[0],'y':bbox[1],'width':bbox[2],'height':bbox[3],
          'confidence':b[4],'world_x':r['plane_x'],'world_y':r['plane_y'],'world_method':r['output_method_code'],'observation_id':r['observation_id']})
        rows.append({'camera':o['camera']+1,'frame':o['frame'],'id':int(r['global_id']),'bbox':bbox,'score':b[4]})
    csvout(work/'predictions_raw.csv',raw)
    gt=json.loads(Path('/project/outputs/osc-analysis-20261101/s30_th40/gt_selected.json').read_text());truth=collections.defaultdict(list)
    for r in gt:
        f=r['source_frame']-277
        if 0<=f<a.frames:truth[(int(r['camera_id'].split('_')[1])+1,f)].append({'id':r['track_id'],'bbox':r['bbox_xywh'],'world':r['world_xyz']})
    metrics,detail,removed=E.metric_suite(rows,truth,a.frames);matches=E.gt_matches(rows,truth)
    evidence=[]
    for i,p in enumerate(rows):
        m=matches.get(i);evidence.append(dict(raw[i],matched_gt_id=None if m is None else m['gt']['id'],gt_iou=None if m is None else m['iou']))
    csvout(work/'posthoc_gt_matches.csv',evidence)
    input_counts=collections.Counter(x['camera'] for x in observations);camrows=[]
    for cam in range(8):
        rr=[r for r in raw if r['camera']==f'cam_{cam}'];camrows.append({'camera':f'cam_{cam}','input_observations':input_counts[cam],
          'output_rows':len(rr),'predicted_global_ids':len({r['global_id'] for r in rr}),
          'output_fraction':len(rr)/input_counts[cam] if input_counts[cam] else None,'empty_output_reason':None if rr else 'Inspect initialization/height/state events; no forced ID'})
    csvout(work/'per_camera.csv',camrows)
    expected={(x['camera'],x['frame'],f"cam{x['camera']}_row{x['camera_row']}") for x in observations}
    output_keys={(int(r['camera'].split('_')[1]),r['model_frame'],r['observation_id']) for r in raw}
    assert output_keys<=expected
    metrics.update(input_rows=len(observations),output_rows=len(rows),output_fraction=len(rows)/len(observations),
      no_output_observations=len(expected-output_keys),frames=a.frames,gt_rows=sum(len(x) for x in truth.values()),
      missing_world_rows=sum(not (r['world_x'] and r['world_y']) or not np.isfinite([float(r['world_x']),float(r['world_y'])]).all() for r in raw),
      protocol='TrackEval kernels, per-camera 2D IoU; separate view-aware global identity; NOT official SCOUT benchmark',gt_used_for_tracking=False)
    E.save(work/'metrics.json',{'summary':metrics,'detail':detail,'evaluation_duplicate_removals':removed})
    E.save(work/'protocol.json',{'raw_bbox':'original observation lookup, verified against rectified tracker bbox',
      'roi':None,'cameras':'cam_0..7','source_frames':[277,276+a.frames],'IDSW':'sum per-camera temporal CLEAR at IoU>=0.5',
      'HOTA':'per-camera sequences, thresholds 0.05..0.95','global_identity':'view-aware protocol; separate camera/frame observations; not world-distance benchmark',
      'GT':'provided MOT annotation export; no size/confidence filtering','missing_detections':'all selected frames including empty predictions',
      'local_id':'not separately exposed by native PoseTrack, left blank','world_unit':'SCOUT dataset units; metric scale not independently surveyed'})
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
