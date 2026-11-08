#!/usr/bin/env python3
"""Audit the uncapped OSC run and preserve comparison evidence, using stdlib only."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/osc-pilot-20261101'
RUN = ROOT/'outputs/osc-pilot-uncapped-20261101'
INF = ROOT/'outputs/osc-inference-uncapped-20261101'

def load(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

assert load(INF/'summary.json')['status'] == 'PASS'
old = {}
for name in ('scout-inference-small-20261031', 'osc-extra-inference-20261101'):
    for row in load(ROOT/'outputs'/name/'yachiyo/detection/detections.json'):
        assert row['key'] not in old
        old[row['key']] = row
new = load(INF/'yachiyo/detection/detections.json')
assert len(new) == len(old) == 400
assert {r['key'] for r in new} == set(old)
for row in new:
    baseline = old[row['key']]
    assert row['sha256'] == baseline['sha256'], 'Source image differs'
    assert row['detections'] == baseline['detections'], 'Raw detections differ'
    expected = sorted((b for b in row['detections'] if b[4] >= .3), key=lambda b: -b[4])
    assert row['selected'] == expected, 'Confidence-qualified detection omitted'
    assert row['selected'][:10] == baseline['selected'], 'Top ten differs'
    assert row['source_frame'] == baseline['source_frame']

for name in ('scout_to_osc_camera_map.json', 'gt_selected.json',
             'inputs/export/manifest.json', 'inputs/export/profile.json'):
    assert load(RUN/name) == load(BASE/name), name
assert load(RUN/'validation_scope.json')['max_people_per_frame'] is None
for i in (1,2):
    name = 'tracking/repeat_%02d/settings.json'%i
    assert load(RUN/name)['parameters'] == load(BASE/name)['parameters'], 'Tracking settings changed'
repeat = load(RUN/'tracking/repeat_report.json')
assert repeat['identical_fixed_whole']
integrity = load(RUN/'evaluation/csv_integrity.json')
assert integrity['status'] == 'PASS' and integrity['duplicates'] == 0
prep = load(RUN/'prepare_report.json')
selected = sum(len(r['selected']) for r in new)
assert prep['native_prediction_rows'] == selected
assert load(RUN/'inference_reuse.json')['embedding_shape'] == [selected,512]
assert len(list((RUN/'inputs/features').rglob('*.npy'))) == selected

def snapshot(folder):
    e = load(folder/'evaluation/evaluation_report.json')
    m = load(folder/'evaluation/metrics_2d.json')['results']
    a = load(folder/'evaluation/csv_integrity.json')
    tracks = load(folder/'tracking/repeat_01/fixed_whole_tracking_results.json')
    return dict(selected=load(folder/'prepare_report.json')['native_prediction_rows'],
                local_rows=e['local_rows'], global_rows=e['global_rows'], global_ids=e['global_ids'],
                local_idf1=e['local_idf1'], multiview_idf1=e['multiview_idf1'],
                global_2d_idf1=m['global_mcpt_2d']['OVERALL']['idf1'],
                local_recall=m['local_scpt']['OVERALL']['recall'],
                local_mota=m['local_scpt']['OVERALL']['mota'],
                local_id_switches=m['local_scpt']['OVERALL']['num_switches'],
                global_recall=m['global_mcpt_2d']['OVERALL']['recall'],
                local_unassigned=len(a['local_unassigned']),
                global_removed=len(a['global_audit']['native_correction_removed']),
                global_rows_per_camera={'cam_'+str(int(k)-1):len(v) for k,v in tracks.items()},
                local_ids_per_camera=[r['local_ids'] for r in load(folder/'tracking/repeat_01/timing.json')['scpt']])

previous = load(ROOT/'logs/osc-checklist-20261101.json')
unchanged = {}
for rel, checksum in previous['file_hashes'].items():
    if rel.startswith('datasets/'):
        actual = sha(ROOT/rel)
        assert actual == checksum, 'Original data changed: '+rel
        unchanged[rel] = actual
assert load(ROOT/'weights/yachiyo/manifest.json')['files'] == previous['weight_manifest']['files']
model = load(ROOT/'sources.lock.json')['repositories']['yachiyo']
head = subprocess.check_output(['git','-C',str(ROOT/'repos/yachiyo'),'rev-parse','HEAD'],text=True).strip()
assert head == model['commit']
assert not subprocess.check_output(['git','-C',str(ROOT/'repos/yachiyo'),'status','--porcelain'],text=True).strip()
provenance = RUN/'provenance'; provenance.mkdir(exist_ok=True)
for name in ('osc-pilot.py','osc-evaluate.py','run-osc-pilot.py','infer-scout-sample.py',
             'scout-sample-stage.py','audit-osc-uncapped.py'):
    shutil.copyfile(ROOT/'scripts'/name, provenance/name)
files = list(provenance.glob('*.py')) + [RUN/'validation_scope.json',RUN/'scout_to_osc_camera_map.json',
          INF/'selection.json', INF/'input/manifest.json',INF/'summary.json']
for stage in ('detection','pose','reid'):
    files += [INF/'yachiyo'/stage/'report.json', INF/'yachiyo'/stage/'run.log']
for folder in (INF/'yachiyo', RUN/'inputs', RUN/'tracking', RUN/'evaluation'):
    files += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.npy','.csv')]
comparison = {'status':'PASS','only_experimental_change':'max_people_per_frame 10 -> null (unlimited)',
              'raw_detections_exactly_equal':True, 'source_images_identical':True,
              'same_gt_mapping_calibration_and_tracking_parameters':True,
              'all_confidence_qualified_boxes_selected':True,
              'frames':400,'frames_exceeding_ten':sum(len(r['selected'])>10 for r in new),
              'min_selected_per_frame':min(len(r['selected']) for r in new),
              'max_selected_per_frame':max(len(r['selected']) for r in new),
              'baseline':snapshot(BASE),'uncapped':snapshot(RUN),
              'baseline_run':str(BASE.relative_to(ROOT)), 'uncapped_run':str(RUN.relative_to(ROOT))}
(RUN/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
with (RUN/'comparison.csv').open('w') as stream:
    writer=csv.writer(stream);writer.writerow(['metric','top10_baseline','uncapped'])
    for key, value in comparison['baseline'].items():
        if not isinstance(value,(dict,list)):
            writer.writerow([key,value,comparison['uncapped'][key]])
for name in ('evaluation/evaluation_report.json','evaluation/metrics_2d.json',
             'evaluation/metrics_multiview_identity.json','tracking/repeat_report.json',
             'tracking/repeat_01/settings.json','prepare_report.json','validation_scope.json'):
    target=RUN/'baseline_reference'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(BASE/name,target)
evidence = dict(date='2026-11-01',status='PASS',comparison=comparison,model=model,
                weight_manifest=load(ROOT/'weights/yachiyo/manifest.json'),
                file_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},
                unchanged_original_data_hashes=unchanged,
                inference=load(INF/'summary.json'), repeat=repeat,prepare=prep,
                evaluation=load(RUN/'evaluation/evaluation_report.json'))
(RUN/'provenance/evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(comparison,indent=2))
