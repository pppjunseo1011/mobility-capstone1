#!/usr/bin/env python3
"""Run read-only diagnostics or one-variable OSC experiments in existing containers."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

R=Path(__file__).resolve().parents[1];A=R/'outputs/osc-analysis-20261101'
B=R/'outputs/osc-pilot-uncapped-20261101'
def run(args):
    print('RUN',*map(str,args),flush=True)
    subprocess.run(list(map(str,args)),cwd=R,check=True)
def phase(which,target,extra=()):run([sys.executable,R/'scripts/run-osc-pilot.py',which,'--output',target,*extra])

p=argparse.ArgumentParser();p.add_argument('action',choices=['forensics','sweep5','run30'])
arg=p.parse_args();A.mkdir(exist_ok=True)
if arg.action=='forensics':
    dest=A/'forensics5';dest.mkdir(exist_ok=True)
    assert not (dest/'summary.json').exists(),'Completed diagnostics already exist'
    if (dest/'run.log').exists():
        (dest/'run.log').rename(dest/('attempt_%d.log'%len(list(dest.glob('attempt_*.log')))))
    image='scout-mtmc/yachiyo-detection:f881fe0'
    cmd=['docker','run','--rm','--pull=never','--network=none','--runtime=runc',
         '--user',f'{os.getuid()}:{os.getgid()}','--cpus','2','--memory','6g','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=',
         '--env','HOME=/tmp','--env','OMP_NUM_THREADS=2','--env','OPENBLAS_NUM_THREADS=2',
         '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/:/runners:/adapter', '--entrypoint','python']
    for src,dst,ro in [(B,'/work',True),(A/'forensics5','/analysis',False),
        (R/'outputs/osc-inference-uncapped-20261101','/inference',True),
        (R/'scripts','/runners',True),(R/'adapters','/adapter',True),
        (R/'repos/yachiyo/tracking/src','/model',True),
        (R/'scripts/osc-pilot.py','/osc_pilot.py',True),(R/'scripts/osc-evaluate.py','/osc_evaluate.py',True)]:
        cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    cmd += [image,'-u','/runners/osc-forensics.py']
    with (dest/'run.log').open('w') as f:
        f.write(json.dumps({'command':cmd,'image_id':subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()})+'\n');f.flush()
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
elif arg.action=='sweep5':
    assert (A/'forensics5/summary.json').exists(),'Analyze baseline first'
    for threshold in (40,30,20,10):
        target=A/('s5_th%d'%threshold);target.mkdir(exist_ok=False)
        for name in ('inputs','gt_selected.json','scout_to_osc_camera_map.json','validation_scope.json'):
            src=B/name
            if src.is_dir():shutil.copytree(src,target/name,copy_function=os.link)
            else:shutil.copyfile(src,target/name)
        phase('track',target,['--short-track-th',str(threshold)])
        phase('evaluate',target)
        if threshold==40:
            assert (target/'tracking/repeat_01/fixed_whole_tracking_results.json').read_bytes()==(B/'tracking/repeat_01/fixed_whole_tracking_results.json').read_bytes()
            old=json.loads((B/'evaluation/evaluation_report.json').read_text())
            assert json.loads((target/'evaluation/evaluation_report.json').read_text())==old
elif arg.action=='run30':
    assert json.loads((A/'inference30/summary.json').read_text())['status']=='PASS'
    target=A/'s30_th40'
    phase('prepare',target,['--inference-run',str(A/'inference30'),'--export',str(A/'export30')])
    phase('track',target,['--short-track-th','40'])
    phase('evaluate',target)
print('COMPLETE',arg.action,flush=True)
