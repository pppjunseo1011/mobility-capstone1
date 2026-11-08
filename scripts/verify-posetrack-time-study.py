#!/usr/bin/env python3
"""Verify data lineage, timestamp scope, equations, and converter on actual output."""
import csv,json,hashlib,subprocess,tempfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-time-study-20261106';OLD=R/'outputs/posetrack-mesh-20261104'
def read(p):return list(csv.DictReader(p.open()))
def main():
 result={};oldobs=json.loads((OLD/'full_input30/observations.json').read_text());obs=json.loads((O/'aligned_input/observations.json').read_text());source={(r['camera'],r['camera_row']):r for r in oldobs};rows=read(O/'frame_mapping.csv');mapping={(int(r['camera']),int(r['source_frame'])):r for r in rows if r['status']=='selected'}
 assert len(mapping)==2359 and len({(r['camera'],r['common_time_index']) for r in rows if r['status']=='selected'})==2359
 hashes={}
 for c in range(8):
  rr=[r for r in obs if r['camera']==c];ix=[source[c,r['camera_row']]['input_row'] for r in rr]
  for name in ['detection','pose','reid']:
   p=O/f'aligned_input/input/camera_{c+1:04d}/{name}.npy';new=np.load(p);old=np.load(OLD/f'full_input30/input/camera_{c+1:04d}/{name}.npy')[ix]
   assert np.isfinite(new).all()
   if name=='reid':np.testing.assert_array_equal(new,old)
   else:
    np.testing.assert_array_equal(new[:,1:],old[:,1:]);np.testing.assert_array_equal(new[:,0],[int(mapping[c,r['source_frame']]['common_time_index']) for r in rr])
   hashes[str(p.relative_to(O))]=hashlib.sha256(p.read_bytes()).hexdigest()
 result['lineage']='PASS: exact same features, keypoints, boxes, scores; only selected rows and frame indices changed';result['input_observations']=len(obs)
 for run in ['B_timestamp','C_timestamp','C_height_abstain']:
  p=O/run;summary=json.loads((p/'summary.json').read_text());assert summary['status']=='PASS' and summary['frames']==300
  first=[l for l in (O/(run+'.log')).read_text().splitlines() if l.startswith('FRAME ')];retry=[l for l in (O/(run+'-retry.log')).read_text().splitlines() if l.startswith('FRAME ')]
  assert first==retry and len(first)==300
  solver=[json.loads(l) for l in (p/'solver.jsonl').read_text().splitlines()];assert all(r['status']==0 for r in solver)
  metrics=json.loads((O/'evaluation'/run/'metrics.json').read_text())['summary'];assert metrics['duplicate_id_rows_removed_for_eval']==0 and metrics['gt_rows']==51500
  mx=0.;count=0
  for r in csv.DictReader((p/'affinity_pairs.csv').open()):
   v={k:float(r[k]) for k in ['Ab','Aepi','Ah','As','reid_weight','norm','a_epi','a_box','a_homo','a_reid','final_score']}
   score=(v['a_epi']*v['Aepi']+v['a_box']*v['Ab']+v['a_homo']*v['Ah']+v['As']*v['a_reid']*v['reid_weight'])/(1+v['reid_weight'])
   if v['norm']>0:score-=(v['a_box']-v['norm'])*.1
   mx=max(mx,abs(max(0,score)-v['final_score']));count+=1
  assert mx<1e-10;result[run]={'status':'PASS','logged_affinity_pairs':count,'formula_max_abs_error':mx,'all_solvers_optimal':True,'streaming_retry_frame_counts_identical':True,'GT_rows':metrics['gt_rows']}
 # Actual common results -> canonical adapter schema -> public converter.
 actual=read(O/'evaluation/C_timestamp/common_predictions.csv');base=[]
 for r in actual:
  base.append(dict(kind='prediction',image_space='raw',id_scope='sequence',sequence='sequence_1',camera=r['camera_id'],frame=r['source_frame_id'],timestamp_seconds=r['timestamp_seconds'],native_track_id=r['global_id'],score=r['detection_confidence'],**{k:r[k] for k in ['x','y','width','height']}))
 with tempfile.TemporaryDirectory(prefix='scout-common-converter-') as td:
  td=Path(td);p=td/'canonical.csv'
  def write(rs):
   with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
  write(base);cmd=['python3',str(R/'scripts/normalize-scout-common-evaluation.py'),'--input',str(p),'--frame-map',str(O/'frame_mapping.csv'),'--output',str(td/'converted.csv'),'--model','PoseTrack','--config','roundtrip']
  subprocess.run(cmd,check=True,capture_output=True);got=read(td/'converted.csv');assert len(got)==len(actual)
  for a,b in zip(actual,got):
   for k in ['camera_id','source_frame_id','common_time_index','global_id']:assert a[k]==b[k]
   for k in ['timestamp_seconds','x','y','width','height','detection_confidence']:assert float(a[k])==float(b[k])
  tests=[]
  for key,value in [('image_space','undistorted'),('id_scope','camera'),('timestamp_seconds','0')]:
   bad=dict(base[0]);bad[key]=value;write([bad]);out=td/('bad_'+key+'.csv');command=cmd.copy();command[command.index('--output')+1]=str(out);res=subprocess.run(command,capture_output=True);assert res.returncode!=0 and not out.exists();tests.append(key)
 result['common_converter']={'actual_rows_roundtripped':len(actual),'unsafe_inputs_rejected':tests}
 result['scope_note']='Numerical/provenance checks, not independent physical calibration accuracy. No GT used to modify neural inputs or tune parameters.'
 (O/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n');(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
