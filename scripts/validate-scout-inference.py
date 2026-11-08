#!/usr/bin/env python3
"""Read-only inference artifact checks plus adapter v0.5 PoseTrack input bridge."""
import collections
import json
from pathlib import Path
import numpy as np
from scout_adapter.model_ground import prepare_model_points

sample, export, out = Path('/sample'), Path('/export'), Path('/outputs')
manifest = json.loads((sample / 'input/manifest.json').read_text())
summary = json.loads((sample / 'summary.json').read_text())
assert summary['status'] == 'PASS' and len(summary['stages']) == 9
report = {'status': 'RUNNING', 'models': {}, 'adapter': [], 'gt_used': False, 'multicamera_tracking_run': False}
for model in ('posetrack', 'yachiyo', 'uwipl'):
    root = sample / model
    dets = json.loads((root / 'detection/detections.json').read_text())
    poses = json.loads((root / 'pose/poses.json').read_text())
    assert len(dets) == len(poses) == len(manifest['frames']) == 16
    expected_rows = []
    for source, det, pose in zip(manifest['frames'], dets, poses):
        assert source['key'] == det['key'] == pose['key']
        assert source['source_frame'] == det['source_frame'] == pose['source_frame']
        boxes = np.asarray(det['selected']).reshape(-1, 7)
        kpts = np.asarray(pose['keypoints']).reshape(-1, 17, 3)
        assert len(boxes) <= 10 and len(boxes) == len(kpts)
        np.testing.assert_allclose(boxes[:, :5], pose['bbox_xyxy_score'], rtol=0, atol=0)
        assert np.isfinite(boxes).all() and np.isfinite(kpts).all()
        assert ((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= 1920) & (boxes[:, 3] <= 1080)).all()
        expected_rows.extend([{'key': source['key'], 'box_index': i, 'source_frame': source['source_frame'], 'camera': source['camera']} for i in range(len(boxes))])
    features = []
    for path in sorted((root / 'reid').glob('*.npy')):
        array = np.load(path, allow_pickle=False)
        mapping = json.loads(path.with_name(path.stem + '-rows.json').read_text())
        assert mapping == expected_rows and len(array) == len(expected_rows)
        assert np.isfinite(array).all() and (np.linalg.norm(array, axis=1) > 0).all()
        features.append({'name': path.name, 'shape': list(array.shape)})
    assert len(features) == (5 if model == 'uwipl' else 1)
    report['models'][model] = {'frames': len(dets), 'detections': sum(len(f['detections']) for f in dets),
                                'selected_people': len(expected_rows), 'pose_rows': len(expected_rows), 'features': features,
                                'bbox_pose_reid_row_alignment': 'PASS'}
for camera in range(8):
    name = 'cam_' + str(camera)
    src = sample / 'posetrack/pose'
    bridge = prepare_model_points(export, src / (name + '-pose57.txt'), out / name,
                                  camera=name, format='posetrack-pose57', policy='ankles-only', id_scope='none',
                                  input_kind='model_output', detections=src / (name + '-det7.txt'))
    import csv
    rows = list(csv.DictReader((out / name / 'point_map.csv').open()))
    expected = report['models']['posetrack']['selected_people']
    assert rows and {int(r['source_frame']) for r in rows} <= {277, 278}
    assert all(r['id_scope'] == 'none' and r['native_track_id'] == '' for r in rows)
    counts = dict(collections.Counter(r['status'] for r in rows))
    report['adapter'].append({'camera': name, 'rows': len(rows), 'status_counts': counts,
                              'source_frames': sorted({int(r['source_frame']) for r in rows})})
assert sum(r['rows'] for r in report['adapter']) == report['models']['posetrack']['selected_people']
report['status'] = 'PASS'
report['scope'] = 'Artifact consistency and real PoseTrack output -> adapter input only; no ground projection, tracking or accuracy evaluation'
(out / 'validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
