#!/usr/bin/env python3
"""Bounded V4 orchestration on existing images, one worker per GPU/scope."""
import concurrent.futures,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v4-20261108';O.mkdir(parents=True,exist_ok=True)
VARIANTS=[('D1_1088x1920',['--height','1088','--width','1920']),('D1_1280x2304',['--height','1280','--width','2304']),('D2_tiles2x2',['--tiles']),('D3_nms0.8',['--nms','0.8'])]
def source(scope):return R/'outputs/uwipl-study-20261101/inference30' if scope=='development' else R/'outputs/uwipl-v3-20261107/holdout/inference'
def mount(a,b,ro=True):return ['-v',str(a)+':'+b+(':ro' if ro else '')]
def run(scope,stage,variant,argv,image='botsort',gpu=None,extra=None):
 log=O/'logs'/f'{scope}-{variant}-{stage}.log';log.parent.mkdir(exist_ok=True);status=log.with_suffix('.status.json')
 if status.exists() and json.loads(status.read_text()).get('returncode')==0:return
 cmd=['docker','run','--rm','--name',f'scout-v4-{scope}-{variant.replace(".","_")}-{stage}','--network','none','--cpus','3','--memory','12g','--shm-size','1g','--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/tmp','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2']
 if gpu is not None:cmd+=['--gpus',f'device={gpu}','-e','NVIDIA_VISIBLE_DEVICES='+str(gpu)]
 cmd+=mount(R,'/project')+mount(O,'/out',False)+mount(R/'vendor','/vendor')
 if extra:cmd+=extra
 cmd+=[f'scout-mtmc/uwipl-{image}:8e23a69','python',*argv]
 start=time.time();status.write_text(json.dumps(dict(state='RUNNING',command=cmd,start=start)))
 with log.open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 status.write_text(json.dumps(dict(state='PASS' if result.returncode==0 else 'FAILED',command=cmd,start=start,end=time.time(),elapsed_seconds=time.time()-start,returncode=result.returncode),indent=2))
 print(scope,variant,stage,'exit',result.returncode,'seconds',round(time.time()-start),flush=True)
 if result.returncode:raise RuntimeError(str(log))
def worker(scope,gpu):
 run(scope,'diagnostic','baseline',['/project/scripts/uwipl-v4.py','detection','--scope',scope])
 run(scope,'tracking','baseline',['/project/scripts/uwipl-v4.py','tracking','--scope',scope])
 for vote in [20,30]:run(scope,'tracking',f'vote{vote}',['/project/scripts/uwipl-v4.py','tracking','--scope',scope,'--vote',str(vote)])
 run(scope,'postvote','baseline',['/project/scripts/uwipl-v4.py','postvote','--scope',scope])
 for variant,options in VARIANTS:
  root=O/'inference'/scope/variant;det=root/'uwipl/detection';det.mkdir(parents=True,exist_ok=True);(root/'input').mkdir(exist_ok=True)
  run(scope,'detect',variant,['/project/scripts/uwipl-v4-detect.py',*options],gpu=gpu,extra=mount(source(scope)/'input','/sample/input')+mount(det,'/outputs',False)+mount(R/'weights/uwipl','/weights'))
  run(scope,'diagnostic',variant,['/project/scripts/uwipl-v4.py','detection','--scope',scope,'--variant',variant])
 # All detector variants are assessed before downstream feature inference/evaluation.
 for variant,_ in VARIANTS:
  root=O/'inference'/scope/variant;reid=root/'uwipl/reid';reid.mkdir(parents=True,exist_ok=True)
  run(scope,'reid',variant,['/project/scripts/scout-sample-stage.py','--model','uwipl','--stage','reid','--max-people-per-frame','0'],image='reid',gpu=gpu,extra=mount(root,'/sample')+mount(source(scope)/'input','/sample/input')+mount(reid,'/outputs',False)+mount(R/'weights/uwipl','/weights'))
  run(scope,'tracking',variant,['/project/scripts/uwipl-v4.py','tracking','--scope',scope,'--variant',variant])
 return scope
if __name__=='__main__':
 state=dict(started_at=datetime.datetime.now().isoformat(),status='RUNNING',variants=VARIANTS,baseline=dict(area=1000,distance=80,score=.95,samples=11,vote=15,new_track_thresh=.7,track_high_thresh=.4,stcra=False,fps=10),gpu_assignment={'development':0,'holdout':1},selection='Independent one-factor comparisons to V3; no holdout-driven selection',script_hashes={n:hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in ['run-uwipl-v4.py','uwipl-v4.py','uwipl-v4-detect.py','scout-sample-stage.py','uwipl-v3.py','evaluate-uwipl-study.py']})
 (O/'run_state.json').write_text(json.dumps(state,indent=2))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[pool.submit(worker,s,g) for s,g in [('development',0),('holdout',1)]]
   for j in concurrent.futures.as_completed(jobs):print('COMPLETE',j.result(),flush=True)
  state['status']='COMPLETE'
 except Exception as e:state.update(status='FAILED',error=str(e));raise
 finally:
  state['updated_at']=datetime.datetime.now().isoformat();(O/'run_state.json').write_text(json.dumps(state,indent=2))
