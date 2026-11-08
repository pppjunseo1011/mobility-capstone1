#!/usr/bin/env python3
"""CPU phases for the bounded OSC checklist; does not infer or download."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','track','evaluate']);p.add_argument('--output',type=Path,required=True)
p.add_argument('--data',type=Path,default=ROOT/'datasets/scout/extracted/scout')
p.add_argument('--export',type=Path,default=ROOT/'outputs/scout-first8-20s-v03-20261027/sequence1-yachiyo')
p.add_argument('--old-run',type=Path,default=ROOT/'outputs/scout-inference-small-20261031')
p.add_argument('--extra-run',type=Path,default=ROOT/'outputs/osc-extra-inference-20261101')
p.add_argument('--inference-run',type=Path,help='Single complete inference run, replacing old + extra')
p.add_argument('--short-track-th',type=int,default=40)
p.add_argument('--repeats',type=int,default=2)
a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
images={'prepare':'scout-mtmc/adapter:0.5','track':'scout-mtmc/yachiyo-tracking:f881fe0','evaluate':'scout-mtmc/yachiyo-detection:f881fe0'}
script='osc-evaluate.py' if a.phase=='evaluate' else 'osc-pilot.py'
cmd=['docker','run','--rm','--pull=never','--label','project=scout-mtmc','--user','%d:%d'%(os.getuid(),os.getgid()),'--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','2','--memory','6g','--network','none','--runtime=runc','--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=','--env','PYTHONHASHSEED=0','--env','OMP_NUM_THREADS=2','--env','OPENBLAS_NUM_THREADS=2','--env','PYTHONDONTWRITEBYTECODE=1','--env','HOME=/tmp','--entrypoint','python']
mounts=[(ROOT/'scripts'/script,'/runner.py',True),(ROOT/'adapters','/adapter',True)]
if a.phase=='prepare':
 mounts += [(out,'/work',False),(a.data.resolve(),'/data',True),(a.export.resolve(),'/export',True)]
 if a.inference_run:
  mounts += [(a.inference_run.resolve(),'/inference',True)]
 else:
  mounts += [(a.old_run.resolve(),'/old',True),(a.extra_run.resolve(),'/extra',True)]
elif a.phase=='track':
 (out/'tracking').mkdir(exist_ok=True)
 mounts += [(out/'inputs','/work/inputs',True),(out/'tracking','/work/tracking',False)]
else:
 (out/'evaluation').mkdir(exist_ok=True)
 mounts += [(out,'/work',True),(out/'evaluation','/evaluation',False),(ROOT/'scripts/osc-pilot.py','/osc_pilot.py',True)]
for src,dst,ro in mounts:
 cmd += ['--mount','type=bind,src=%s,dst=%s%s'%(src,dst,',readonly' if ro else '')]
cmd += ['--env','PYTHONPATH=/adapter',images[a.phase],'-u','/runner.py']
if a.phase!='evaluate':cmd += [a.phase]
if a.phase=='track':cmd += ['--short-track-th',str(a.short_track_th),'--repeats',str(a.repeats)]
image_id=subprocess.check_output(['docker','image','inspect',images[a.phase],'--format','{{.Id}}'],text=True).strip()
with (out/(a.phase+'.log')).open('w') as log:
 log.write(json.dumps({'command':cmd,'image_id':image_id})+'\n');log.flush()
 process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 code=process.wait()
print(a.phase,'exit',code,'log',out/(a.phase+'.log'),flush=True)
raise SystemExit(code)
