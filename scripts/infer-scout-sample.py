#!/usr/bin/env python3
"""Run bounded real SCOUT inference in existing Docker images (no downloads)."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--adapter-export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=['posetrack', 'yachiyo', 'uwipl'], default=['posetrack', 'yachiyo', 'uwipl'])
    p.add_argument('--gpu-posetrack', required=True)
    p.add_argument('--gpu-others', required=True)
    p.add_argument('--frames-per-camera', type=int, default=2, choices=range(1, 601))
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--max-people-per-frame', type=int, default=10,
                   help='Maximum downstream detections per frame; 0 means no count limit')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.max_people_per_frame < 0:
        p.error('--max-people-per-frame must be non-negative')
    data, export, output = a.data.resolve(), a.adapter_export.resolve(), a.output.resolve()
    for source in (data, export, ROOT / 'weights'):
        if source == output or source in output.parents or output in source.parents:
            p.error('Input/output paths must be disjoint')
    if output.exists() and not a.resume:
        p.error('Output exists; choose a new output or explicitly --resume')
    output.mkdir(parents=True, exist_ok=True)
    selection = {'max_people_per_frame': a.max_people_per_frame or None,
                 'downstream_objectness_threshold': 0.3}
    selection_file = output / 'selection.json'
    if a.resume:
        previous = json.loads(selection_file.read_text()) if selection_file.exists() else {
            'max_people_per_frame': 10, 'downstream_objectness_threshold': 0.3}
        assert previous == selection, 'Resume selection differs; use a new output'
    selection_file.write_text(json.dumps(selection, indent=2))
    metadata = json.loads((export / 'manifest.json').read_text())
    assert metadata['sequence'] == 'sequence_1' and metadata['labels_used_as_detections'] is False
    cameras = metadata['cameras']
    assert a.start_index >= 0 and all(len(cam['frames']) >= a.start_index + a.frames_per_camera for cam in cameras)
    assert {x['id'] for x in cameras} == {'cam_' + str(i) for i in range(8)}
    inp = output / 'input'
    inp.mkdir(exist_ok=True)
    frames = []
    for cam in cameras:
        for frame in cam['frames'][a.start_index:a.start_index + a.frames_per_camera]:
            path = data / frame['source_path']
            assert data in path.resolve().parents
            rel = cam['id'] + '/frame_%06d.jpg' % frame['index']
            dst = inp / rel
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copyfile(path, dst)
            checksum = sha(path)
            assert sha(dst) == checksum
            frames.append(dict(frame, camera=cam['id'], model_camera_id=cam['model_camera_id'],
                               key=cam['id'] + '_%06d' % frame['index'], file=rel, sha256=checksum))
    manifest = {'sequence': 'sequence_1', 'frames': frames, 'gt_used': False,
                'adapter_manifest_sha256': sha(export / 'manifest.json'),
                'selection': 'first bounded frames of the existing verified adapter export'}
    if (inp / 'manifest.json').exists():
        assert json.loads((inp / 'manifest.json').read_text()) == manifest, 'Resume input differs'
    (inp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    sources = json.loads((ROOT / 'sources.lock.json').read_text())['repositories']
    summary = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'frames': len(frames), 'cameras': len(cameras), 'source_frames': sorted({f['source_frame'] for f in frames}),
               'stages': [], 'gt_used': False, 'multicamera_tracking_run': False, 'evaluation_run': False}
    runner = ROOT / 'scripts/scout-sample-stage.py'
    code = output / 'runner-history'
    code.mkdir(exist_ok=True)
    frozen = code / (sha(runner) + '.py')
    if not frozen.exists():
        shutil.copyfile(runner, frozen)
    for model in a.models:
        weights = ROOT / 'weights' / model
        weights_manifest = json.loads((weights / 'manifest.json').read_text())
        for item in weights_manifest['files']:
            rel = item.get('model_relative_path', item['filename'])
            if not rel.startswith('synthetic/'):
                assert sha(weights / rel) == item['sha256'], 'Weight checksum changed: ' + rel
        for stage in ('detection', 'pose', 'reid'):
            target = output / model / stage
            target.mkdir(parents=True, exist_ok=True)
            report = target / 'report.json'
            if a.resume and report.exists() and json.loads(report.read_text())['status'] == 'PASS':
                summary['stages'].append({'model': model, 'stage': stage, 'status': 'PASS', 'resumed_existing': True})
                continue
            if stage != 'detection' and not (output / model / 'detection/report.json').exists():
                summary['stages'].append({'model': model, 'stage': stage, 'status': 'BLOCKED', 'reason': 'detection failed'})
                continue
            image_stage = 'all' if model == 'posetrack' else ('botsort' if model == 'uwipl' and stage == 'detection' else stage)
            image = 'scout-mtmc/%s-%s:%s' % (model, image_stage, sources[model]['commit'][:7])
            image_id = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
            gpu = a.gpu_posetrack if model == 'posetrack' else a.gpu_others
            free = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()
            if int(free) < 6000:
                raise RuntimeError('Less than 6 GB GPU memory free; not touching other processes')
            cmd = ['docker', 'run', '--rm', '--pull=never', '--label', 'project=scout-mtmc',
                   '--user', '%d:%d' % (os.getuid(), os.getgid()), '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   '--cpus', '2', '--memory', '10g', '--shm-size', '1g', '--network', 'none', '--runtime=nvidia',
                   '--gpus', 'device=' + gpu, '--env', 'NVIDIA_VISIBLE_DEVICES=' + gpu,
                   '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility', '--env', 'HOME=/tmp',
                   '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'OMP_NUM_THREADS=2',
                   '--mount', 'type=bind,src=%s,dst=/sample,readonly' % output,
                   '--mount', 'type=bind,src=%s,dst=/outputs' % target,
                   '--mount', 'type=bind,src=%s,dst=/weights,readonly' % weights,
                   '--mount', 'type=bind,src=%s,dst=/runner.py,readonly' % frozen,
                   image, 'python', '-u', '/runner.py', '--model', model, '--stage', stage,
                   '--max-people-per-frame', str(a.max_people_per_frame)]
            print('RUN', model, stage, 'GPU', gpu, flush=True)
            with (target / 'run.log').open('w') as log:
                log.write(json.dumps({'command': cmd, 'image_id': image_id}) + '\n')
                log.flush()
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            status = 'PASS' if rc == 0 and report.exists() else 'FAILED'
            summary['stages'].append(dict(model=model, stage=stage, status=status, exit_code=rc, image_id=image_id, gpu=gpu))
            print(status, model, stage, str(target / 'run.log'), flush=True)
            if status == 'FAILED':
                print('\n'.join((target / 'run.log').read_text().splitlines()[-16:]), flush=True)
            (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    for f in frames:
        assert sha(data / f['source_path']) == f['sha256'], 'Source image changed'
    summary['source_images_unchanged'] = True
    summary['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary['status'] = 'PASS' if all(s['status'] == 'PASS' for s in summary['stages']) else 'PARTIAL'
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
