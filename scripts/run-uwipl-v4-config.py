#!/usr/bin/env python3
import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v4-config-20261109';O.mkdir(exist_ok=True)
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=[R/'scripts'/n for n in ['uwipl-v3.py','uwipl-v4.py','uwipl-v4-detect.py','evaluate-uwipl-study.py']]+[R/'outputs/uwipl-v3-20261107/run_stcra_scout.py']+list((R/'repos/uwipl/STCRA').glob('*.py'))
manifest=O/'unchanged_code.json'
if not manifest.exists():save(manifest,{str(p.relative_to(R)):sha(p) for p in protected})
for path,digest in json.loads(manifest.read_text()).items():assert sha(R/path)==digest
save(O/'experiment_design.json',dict(baseline=dict(area=1000,distance=80,score=.95,samples=11,vote=15,detection_size=[800,1440],new_track_thresh=.7,track_high_thresh=.4),scope=['development','holdout'],seconds=[5,30],votes=[30,40,50],combinations=['vote30 + STCRA4/3/2 + postvote15','vote30 + STCRA4/3/2 + postvote30'],model_code_modified=False,neural_inference=False,GT_used_for_prediction=False,tiling='DEFERRED: existing detector has no full+tile merge, camera-specific, or tile-only score options; new input-feature remapping would exceed settings-only scope',reevaluation=['GT height>=50 strict/all predictions retained','GT height>=50 auxiliary small-GT ignore at IoU0.5 after prioritizing eligible GT','IoU0.3 Identity/CLEAR; standard HOTA unchanged; HOTA@0.3 separately']))
def run(scope):
 for action in ['anchor','stcra','reevaluate']:
  status=O/'logs'/f'{scope}-{action}.json'
  if status.exists() and json.loads(status.read_text()).get('returncode')==0:continue
  cmd=['docker','run','--rm','--pull=never','--network=none','--runtime=runc','--cpus','3','--memory','10g','--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/tmp','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-e','NVIDIA_VISIBLE_DEVICES=void','-e','PYTHONDONTWRITEBYTECODE=1']
  for src,dst,ro in [(R,'/project',True),(O,'/out',False),(R/'vendor','/vendor',True),(R/'datasets/scout/extracted/scout','/data',True)]:cmd+=['-v',str(src)+':'+dst+(':ro' if ro else '')]
  cmd+=['scout-mtmc/uwipl-botsort:8e23a69','python','-u','/project/scripts/uwipl-v4-config-study.py',action,'--scope',scope];t=time.time()
  with (O/'logs'/f'{scope}-{action}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  save(status,dict(command=cmd,returncode=r.returncode,start=t,seconds=time.time()-t));print(scope,action,r.returncode,flush=True)
  if r.returncode:raise RuntimeError(str(status))
if __name__=='__main__':
 save(O/'state.json',dict(status='RUNNING',started=time.time()))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[pool.submit(run,s) for s in ['development','holdout']]
   for j in jobs:j.result()
  for path,digest in json.loads(manifest.read_text()).items():assert sha(R/path)==digest
  save(O/'state.json',dict(status='COMPLETE',completed=time.time()))
 except Exception as e:save(O/'state.json',dict(status='FAILED',error=str(e)));raise
