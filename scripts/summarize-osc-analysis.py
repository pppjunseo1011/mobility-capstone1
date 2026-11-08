#!/usr/bin/env python3
"""Collate controlled experiments without mixing frame ranges or evaluation protocols."""
import collections
import csv
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

R=Path(__file__).resolve().parents[1];A=R/'outputs/osc-analysis-20261101'
B=R/'outputs/osc-pilot-uncapped-20261101'
def load(path):return json.loads(path.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

baseline=load(B/'tracking/repeat_01/settings.json')['parameters']
rows=[];camera_rows=[];gates=[]
for seconds,t in [(5,40),(5,30),(5,20),(5,10),(30,40)]:
    folder=A/('s%d_th%d'%(seconds,t))
    if not (folder/'evaluation/evaluation_report.json').exists():continue
    e=load(folder/'evaluation/evaluation_report.json')
    metrics=load(folder/'evaluation/metrics_2d.json')['results']
    params=load(folder/'tracking/repeat_01/settings.json')['parameters']
    expected=dict(baseline,short_track_th=t);assert params==expected
    if seconds==5:
        for cid in range(1,9):
            p='tracking/repeat_01/fixed_camera%03d_tracking_results.json'%cid
            assert (folder/p).read_bytes()==(B/p).read_bytes(),'Local output changed'
    repeat=load(folder/'tracking/repeat_report.json');assert repeat['identical_fixed_whole']
    raw=load(folder/'tracking/repeat_01/whole_tracking_results.json')
    fixed=load(folder/'tracking/repeat_01/fixed_whole_tracking_results.json')
    reps=load(folder/'tracking/repeat_01/representative_nodes_scene1.json')
    scope=load(folder/'validation_scope.json')
    rows.append(dict(duration_nominal_seconds=seconds,source_start=e['source_frame_start'],source_end=e['source_frame_end'],
        short_track_th=t,local_idf1=e['local_idf1'],local_idsw=metrics['local_scpt']['OVERALL']['num_switches'],
        local_mota=metrics['local_scpt']['OVERALL']['mota'],local_rows=e['local_rows'],
        global_idf1_multiview=e['multiview_idf1'],global_idf1_per_camera_2d=metrics['global_mcpt_2d']['OVERALL']['idf1'],
        final_global_rows=e['global_rows'],global_ids=e['global_ids'],
        raw_global_assigned=sum('GlobalOfflineID' in r for v in raw.values() for r in v.values()),
        eligible_tracklets=sum(len(v['all_serials'])>=t and v['representative_node']['score']<=1 for rr in reps.values() for v in rr.values()),
        max_camera_timestamp_skew_seconds=scope['maximum_camera_timestamp_skew_seconds'],
        repeat_equal=True,track_seconds_repeat1=repeat['runs'][0]['total_seconds'],
        hota=None,deta=None,assa=None,official_scout_benchmark=False))
    for cid,rr in reps.items():
        camera='cam_'+str(int(cid)-1)
        camera_rows.append(dict(duration_nominal_seconds=seconds,short_track_th=t,camera=camera,
            local_idf1=metrics['local_scpt'][camera]['idf1'],local_idsw=metrics['local_scpt'][camera]['num_switches'],
            global_rows=len(fixed[cid]),global_ids=len({r['GlobalOfflineID'] for r in fixed[cid].values()})))
        for lid,v in rr.items():
            n=len(v['all_serials']);score=v['representative_node']['score']
            gates.append(dict(duration_nominal_seconds=seconds,short_track_th=t,camera=camera,local_id=int(lid),length=n,pose_condition=score,
                reason='length' if n<t else 'pose' if score>1 else 'eligible'))
write_csv(A/'experiment_comparison.csv',rows);write_csv(A/'per_camera_comparison.csv',camera_rows)
write_csv(A/'tracklet_gate_comparison.csv',gates)
gt_groups=collections.defaultdict(list)
for g in load(B/'gt_selected.json'):gt_groups[(g['camera_id'],g['track_id'])].append(g['source_frame'])
gt_lengths=[dict(camera=c,gt_id=i,observations=len(fs),source_start=min(fs),source_end=max(fs),span=max(fs)-min(fs)+1)
            for (c,i),fs in sorted(gt_groups.items())]
write_csv(A/'forensics5/gt_track_lengths.csv',gt_lengths)
manifest=load(A/'export30/manifest.json');sync=[]
for c in manifest['cameras']:
    fs=c['frames'];ts=[f['timestamp_seconds'] for f in fs]
    gaps=[dict(after_source=a['source_frame'],before_source=b['source_frame'],dt_seconds=b['timestamp_seconds']-a['timestamp_seconds'])
          for a,b in zip(fs,fs[1:]) if b['timestamp_seconds']-a['timestamp_seconds']>.15]
    sync.append(dict(camera=c['id'],first=ts[0],last=ts[-1],elapsed_seconds=ts[-1]-ts[0],gaps_over_150ms=gaps,
        min_step_seconds=min(b-a for a,b in zip(ts,ts[1:])),max_step_seconds=max(b-a for a,b in zip(ts,ts[1:]))))
skews=[dict(source_frame=manifest['frame_ids'][i],skew_seconds=max(c['frames'][i]['timestamp_seconds'] for c in manifest['cameras'])-min(c['frames'][i]['timestamp_seconds'] for c in manifest['cameras'])) for i in range(300)]
write_csv(A/'timestamp_skew.csv',skews)
summary={'experiments':rows,'per_camera':camera_rows,'five_second_local_outputs_byte_identical':True,
         'five_second_gt_camera_id_pairs':len(gt_lengths),
         'only_parameter_changed_in_sweep':'short_track_th', 'extended_nominal_seconds':30,
         'extended_source_range':[277,576], 'time_diagnostics':sync,
         'max_timestamp_skew_seconds':max(r['skew_seconds'] for r in skews),
         'world_time_alignment_validated':False,'hota_deta_assa_computed':False,
         'note':'5s is the prefix of the 30s development clip, not an independent test set. Global IDF1 uses view-local bbox matching and pooled IDs; World fusion is not time-aligned.'}
if (A/'inference30/summary.json').exists() and load(A/'inference30/summary.json').get('status')=='PASS':
    old={r['key']:r for r in load(R/'outputs/osc-inference-uncapped-20261101/yachiyo/detection/detections.json')}
    matched=0
    for r in load(A/'inference30/yachiyo/detection/detections.json'):
        if r['source_frame']<=326:
            assert r['sha256']==old[r['key']]['sha256'] and r['selected']==old[r['key']]['selected']
            matched+=1
    assert matched==400
    summary['prefix_400_images_and_detections_exactly_match']=True
(A/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

P=A/'provenance';P.mkdir(exist_ok=True)
patches=[]
for name in ('osc-forensics.py','run-osc-analysis.py','summarize-osc-analysis.py','osc-pilot.py','osc-evaluate.py',
             'run-osc-pilot.py','infer-scout-sample.py','scout-sample-stage.py','plot-osc-analysis.py',
             'plot-osc-pose-failures.py','create-osc-analysis-report.py','package-osc-analysis.py'):
    shutil.copyfile(R/'scripts'/name,P/name)
    before=B/'provenance'/name
    if before.exists():
        patches.extend(difflib.unified_diff(before.read_text().splitlines(True),(R/'scripts'/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(P/'runner_changes.patch').write_text(''.join(patches))
modelsrc=R/'repos/yachiyo/tracking/src'
for name in ('scpt.py','mcpt.py','tracking.py','pose.py'):
    shutil.copyfile(modelsrc/name,P/('original_'+name))
head=subprocess.check_output(['git','-C',str(R/'repos/yachiyo'),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(R/'repos/yachiyo'),'status','--porcelain'],text=True).strip()
evidence={'model_commit':head,'model_source_modified':False,'weights':load(R/'weights/yachiyo/manifest.json'),
          'script_hashes':{p.name:sha(p) for p in P.glob('*.py')},
          'baseline_fixed_whole_sha256':sha(B/'tracking/repeat_01/fixed_whole_tracking_results.json')}
(P/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(rows,indent=2))
