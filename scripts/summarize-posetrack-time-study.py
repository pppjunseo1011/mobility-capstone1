#!/usr/bin/env python3
"""Create evidence-backed comparison and a concise review bundle source."""
import csv,difflib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-time-study-20261106';S=O/'summary'
def load(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def write(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 S.mkdir(exist_ok=True);names=['original_shared','B_timestamp','C_timestamp','C_height_abstain','B_same_output','C_same_output'];rows=[load(O/'evaluation'/n/'metrics.json')['summary'] for n in names]
 legacy=load(R/'outputs/posetrack-mesh-20261104/C_30s/metrics.json')['summary'];save(S/'original_legacy_metrics.json',legacy)
 write(S/'metrics_comparison.csv',rows);save(S/'metrics_comparison.json',rows);write(S/'ground_affinity_B_vs_C.csv',[r for r in rows if r['config'] in ['B_timestamp','C_timestamp','B_same_output','C_same_output']])
 cameras=[]
 for name in names[:4]:
  for r in csv.DictReader((O/'evaluation'/name/'per_camera_metrics.csv').open()):cameras.append(dict(config=name,**r))
 write(S/'per_camera_summary.csv',cameras)
 summary=dict(final_recommended_configuration='C_timestamp',additional_candidate='C_height_abstain NOT adopted: lower overall HOTA/shared IDF1 and output; kept as controlled negative result',
  scope='Sequence1 cam0..7 source277..576 mapped to 300 timestamps; no ROI',timestamp=load(O/'timestamp_policy.json'),
  metrics=rows,legacy_scope_warning='Original legacy: 52713 GT rows; shared mask: 51500. Direct legacy-vs-new difference mixes coverage and tracking effects.',
  geometry=load(O/'geometry/mesh_summary.json'),original_5s_losses=load(O/'association/original_5s_output_loss_summary.json'),
  recovery=load(O/'association/missing_reactivation_summary.json'),verification=load(O/'verification.json'),
  GT_usage='evaluation and posthoc failure/visual analysis only',score_targets_met=False,
  remaining=['No proven correct recovery example among 14 original events; 13 lack sufficient GT evidence. One historical GT conflict has a 147-frame support gap.',
  'No independent physical calibration or mesh-coverage repair verified. Missing World coordinates remain.',
  'No 60s extension or four-system inference run in this task.',
  'Common aligned protocol is an internal 2D/view-aware comparison, not official SCOUT 3D benchmark.',
  'Raw geometric scores use SCOUT coordinate units; original numeric thresholds preserved, not retuned for higher metrics.'])
 save(S/'experiment_summary.json',summary)
 patch=''.join(difflib.unified_diff((R/'scripts/run-posetrack-mesh.py').read_text().splitlines(True),(R/'scripts/run-posetrack-time-study.py').read_text().splitlines(True),fromfile='previous/scripts/run-posetrack-mesh.py',tofile='current/scripts/run-posetrack-time-study.py'))
 (S/'code_changes.patch').write_text(patch)
 # Actual result names avoid representing the rejected candidate as an improvement.
 T=O/'tracking_results';T.mkdir(exist_ok=True)
 for name,file in [('original_shared','original_30s_shared_scope.csv'),('B_timestamp','ground_off_30s.csv'),('C_timestamp','timestamp_fixed_30s.csv'),('C_height_abstain','candidate_height_abstain_30s.csv')]:shutil.copy2(O/'evaluation'/name/'common_predictions.csv',T/file)
 shutil.copy2(T/'timestamp_fixed_30s.csv',T/'final_recommended_30s.csv')
 print('SUMMARY created; final = C_timestamp')
if __name__=='__main__':main()
