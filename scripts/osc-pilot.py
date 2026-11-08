#!/usr/bin/env python3
"""Prepare existing OSC predictions or run native tracking, in isolated containers."""
import argparse
import collections
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

WORK = Path('/work')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def iou_matrix(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    overlap = np.maximum(rb-lt, 0).prod(axis=2)
    return overlap / np.maximum((a[:, 2:]-a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:]-b[:, :2]).prod(axis=1)[None, :] - overlap, 1e-12)


def prepare():
    from scipy.optimize import linear_sum_assignment
    from scout_adapter.annotations import iter_ground_truth
    from scout_adapter.geometry import CameraGeometry, GroundMesh, read_ground_ply
    began = time.monotonic()
    inp = WORK / 'inputs'
    assert not inp.exists(), 'Use a fresh pilot output'
    inp.mkdir(parents=True)
    manifest = json.loads(Path('/export/manifest.json').read_text())
    assert manifest['model'] == 'yachiyo'
    sources = (Path('/inference'),) if Path('/inference').exists() else (Path('/old'), Path('/extra'))
    frame_ids = sorted({f['source_frame'] for s in sources for f in json.loads((s/'input/manifest.json').read_text())['frames']})
    start, end, count = frame_ids[0], frame_ids[-1], len(frame_ids)
    assert frame_ids == list(range(start,end+1)), 'Non-contiguous input'
    manifest['frame_ids'] = frame_ids
    for cam in manifest['cameras']:
        cam['frames'] = [f for f in cam['frames'] if f['source_frame'] in frame_ids]
        assert [f['source_frame'] for f in cam['frames']] == frame_ids
        assert [f['index'] for f in cam['frames']] == list(range(count)), 'Use an export starting at input start'
    save(inp / 'export/manifest.json', manifest)
    shutil.copyfile('/export/profile.json', inp / 'export/profile.json')
    mapping = {'sequence': 'sequence_1', 'osc_scene': 'scene_001', 'source_frame_start': start,
               'source_frame_end_inclusive': end, 'osc_frame_start': 1, 'osc_frame_end_inclusive': count,
               'frame_formula': 'osc_frame = scout_source_frame - %d'%(start-1),
               'cameras': {c['id']: {'osc_camera_id': c['model_camera_id'], 'osc_camera_name': 'camera_%04d' % c['model_camera_id']} for c in manifest['cameras']}}
    save(WORK / 'scout_to_osc_camera_map.json', mapping)
    save(inp / 'scout_to_osc_camera_map.json', mapping)
    timestamps = np.asarray([[f['timestamp_seconds'] for f in c['frames']] for c in manifest['cameras']])
    detection_reports = [json.loads((s / 'yachiyo/detection/report.json').read_text()) for s in sources]
    caps = [r['max_people_per_frame'] for r in detection_reports]
    assert all(c == caps[0] for c in caps), 'Mixed selection policies'
    assert all(r['status'] == 'PASS' and r['downstream_objectness_filter'] == .3 for r in detection_reports)
    scope = dict(mapping, purpose='%g-second development pilot; not held-out benchmark'%(count/10),
                 cameras_count=8, frames_per_camera=count, total_images=8*count, nominal_fps=10,
                 maximum_camera_timestamp_skew_seconds=float(np.ptp(timestamps, axis=0).max()),
                 gt_source='SCOUT mot_format/sequence_1/cam_0..cam_7.txt',
                 annotation_quality='first eight sequence_1 views; corrected sparse annotation with interpolation, not manually labeled every frame',
                 detector_objectness_threshold=.1, downstream_objectness_threshold=.3, max_people_per_frame=caps[0],
                 old_frames_reused=0 if len(sources)==1 else 16,
                 new_frames_inferred=8*count if len(sources)==1 else 8*count-16)
    save(WORK / 'validation_scope.json', scope)
    combined, feature_by_row, evidence = {}, {}, []
    for source in sources:
        dets = json.loads((source / 'yachiyo/detection/detections.json').read_text())
        poses = json.loads((source / 'yachiyo/pose/poses.json').read_text())
        features = np.load(source / 'yachiyo/reid/osnet_ms_m_c.npy', allow_pickle=False)
        rows = json.loads((source / 'yachiyo/reid/osnet_ms_m_c-rows.json').read_text())
        assert len(dets) == len(poses), 'Detection/pose frame count mismatch'
        assert features.shape == (len(rows), 512) and np.isfinite(features).all()
        for row, feature in zip(rows, features):
            key = (row['key'], row['box_index'])
            assert key not in feature_by_row
            feature_by_row[key] = feature
        for d, pose in zip(dets, poses):
            assert d['key'] == pose['key'] and d['key'] not in combined
            np.testing.assert_array_equal(np.asarray(d['selected'])[:, :5], pose['bbox_xyxy_score'])
            assert len(pose['keypoints']) == len(d['selected'])
            assert np.isfinite(np.asarray(pose['keypoints'])).all(), 'Non-finite keypoints'
            combined[d['key']] = (d, pose, source / 'input' / d['file'])
        evidence.append({'source': str(source), 'frames': len(dets), 'features_shape': list(features.shape),
                         'detections_sha256': sha(source / 'yachiyo/detection/detections.json'),
                         'poses_sha256': sha(source / 'yachiyo/pose/poses.json'),
                         'features_sha256': sha(source / 'yachiyo/reid/osnet_ms_m_c.npy')})
    assert len(combined) == 8*count
    vertices, faces = read_ground_ply('/data/meshes/mesh_ground_only.ply')
    mesh = GroundMesh(vertices, faces)
    all_tracks, world_records, pose_pair_count = {}, [], 0
    # Prediction-only preparation, before loading any ground-truth annotations.
    for cam in manifest['cameras']:
        name, cid = cam['id'], cam['model_camera_id']
        camera = CameraGeometry(cam['calibration'], 1920, 1080)
        native, pose_json, serial = {}, {}, 0
        feature_dir = inp / ('features/scene_001/camera_%04d' % cid)
        feature_dir.mkdir(parents=True)
        for frame in cam['frames']:
            key = name + '_%06d' % frame['index']
            det, pose, image = combined[key]
            model_frame = frame['index'] + 1
            pose_json[str(model_frame)] = []
            if model_frame in (1, count//2, count):
                dest = inp / 'Images' / name / ('%06d.jpg' % model_frame)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image, dest)
            for bi, (box, kp) in enumerate(zip(det['selected'], pose['keypoints'])):
                x1, y1, x2, y2 = map(int, box[:4])
                feature = feature_by_row[(key, bi)]
                fname = 'feature_%d_%d_%d_%d_%d_%d_%s.npy' % (model_frame, serial, x1, x2, y1, y2, ('%.8f' % box[4]).replace('.', ''))
                np.save(feature_dir / fname, feature)
                pixel = [(x1+x2)/2, y2]
                geom = camera.ground_point(pixel, mesh) if y2 < 1080 and pixel[0] < 1920 else {'status': 'outside_image', 'world': None}
                world = geom['world']
                native[str(serial).zfill(8)] = {'Frame': model_frame, 'NpyPath': str(feature_dir / fname),
                    'Coordinate': dict(x1=x1, y1=y1, x2=x2, y2=y2),
                    'WorldCoordinate': None if world is None else {'x': world[0], 'y': world[1]}, 'OfflineID': -1}
                pose_json[str(model_frame)].append({'bbox': box[:5], 'keypoints': kp})
                world_records.append({'camera': name, 'camera_id': cid, 'frame': frame['source_frame'],
                                      'osc_frame': model_frame, 'serial': str(serial).zfill(8), 'bbox': [x1,y1,x2,y2],
                                      'world': world, 'status': geom['status']})
                serial += 1
                pose_pair_count += 1
        save(inp / ('native/camera%03d.json' % cid), native)
        save(inp / ('Pose/scene_001/camera_%04d/camera_%04d_out_keypoint.json' % (cid,cid)), pose_json)
        all_tracks[name] = native
        print('PREPARED', name, len(native), flush=True)
    save(inp / 'predicted_world.json', world_records)
    save(WORK / 'inference_reuse.json', {'sources': evidence, 'pose_bbox_pairs': pose_pair_count, 'embedding_shape': [pose_pair_count,512], 'finite': True})
    gt = list(iter_ground_truth(Path('/data'), manifest, 'mot'))
    save(WORK / 'gt_selected.json', gt)
    grouped_gt, grouped_pred = collections.defaultdict(list), collections.defaultdict(list)
    for row in gt:
        grouped_gt[(row['camera_id'], row['source_frame'])].append(row)
    for row in world_records:
        grouped_pred[(row['camera'], row['frame'])].append(row)
    comparisons = []
    for key, preds in grouped_pred.items():
        truth = grouped_gt[key]
        gt_boxes = [[b[0],b[1],b[0]+b[2],b[1]+b[3]] for b in (r['bbox_xywh'] for r in truth)]
        overlaps = iou_matrix([r['bbox'] for r in preds], gt_boxes)
        ii,jj = linear_sum_assignment(np.where(overlaps >= .5, 1-overlaps, 1e6))
        for i,j in zip(ii,jj):
            if overlaps[i,j] < .5:
                continue
            pred, truth_row = preds[i], truth[j]
            error = None if pred['world'] is None or truth_row['world_xyz'] is None else float(np.linalg.norm(np.asarray(pred['world'])-truth_row['world_xyz']))
            comparisons.append(dict(camera=key[0], frame=key[1], serial=pred['serial'], gt_id=truth_row['track_id'],
                                    iou=float(overlaps[i,j]), predicted_world=pred['world'], annotation_world=truth_row['world_xyz'], error_dataset_units=error))
    errors = [r['error_dataset_units'] for r in comparisons if r['error_dataset_units'] is not None]
    world_report = {'prediction_status_counts': dict(collections.Counter(r['status'] for r in world_records)),
                    'iou_match_threshold': .5, 'bbox_matched': len(comparisons), 'world_compared': len(errors),
                    'mean_error_dataset_units': float(np.mean(errors)) if errors else None,
                    'median_error_dataset_units': float(np.median(errors)) if errors else None,
                    'p95_error_dataset_units': float(np.percentile(errors,95)) if errors else None,
                    'source': 'Separate SCOUT MOT columns 8..10, not recomputed from prediction',
                    'physical_unit_independently_verified': False, 'calibration_accuracy_accepted': False,
                    'note': 'Prediction localization error against annotation, not independent surveyed camera calibration validation; IoU matched subset only.'}
    save(WORK / 'world_gt_comparison.json', {'summary': world_report, 'pairs': comparisons})
    for cam in manifest['cameras']:
        for index in (1,count//2,count):
            image = cv2.imread(str(inp / 'Images' / cam['id'] / ('%06d.jpg' % index)))
            for row in grouped_gt[(cam['id'], index+start-1)]:
                x,y,w,h = row['bbox_xywh'];cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
                cv2.putText(image,'GT %s'%row['track_id'],(int(x),max(12,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
            dest=WORK/'visualizations/gt';dest.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(dest/(cam['id']+'_%d.jpg'%(index+start-1))),image)
    save(WORK/'prepare_report.json', {'status':'PASS','seconds':time.monotonic()-began,'gt_rows':len(gt),
        'gt_ids':len({g['track_id'] for g in gt}),'native_prediction_rows':pose_pair_count,'world':world_report})


def track(short_track_th=40, repeats=2):
    sys.path.insert(0, '/opt/yachiyo/tracking/src')
    from tracking import Tracker
    import run
    import pose
    inp = WORK/'inputs'; os.chdir(inp)
    params = {'fps':10,'time_period':3,'epsilon_scpt':.10,'epsilon_mcpt':.37,'short_track_th':short_track_th,
              'keypoint_condition_th':1,'replace_similarity_by_wcoordinate':False,
              'distance_type':'min','distance_th':10,'sim_th':.85,'delete_gid_th':0,
              'reassign_global_id':False,'measure_wcoordinate':False,'delete_distant_person':False,
              'interpolate_track':False}
    results=[]
    for repeat in range(1,repeats+1):
        began=time.monotonic();np.random.seed(0)
        target=WORK/'tracking'/('repeat_%02d'%repeat);target.mkdir(parents=True,exist_ok=False)
        save(target/'settings.json',{'parameters':Tracker(params).parameters,
              'variant':'appearance-and-pose pilot; world distance corrections disabled pending validation',
              'changes':['fps 30->10','short_track_th=%d'%short_track_th,
                         'world-dependent distance/deletion/interpolation disabled',
                         'global reassignment disabled for short development clip; unassigned IDs preserved'],
              'seed':0,'repeat':repeat})
        timings=[]
        for cid in range(1,9):
            data=json.loads((inp/('native/camera%03d.json'%cid)).read_text())
            pk=pose.PoseKeypoints(str(inp/('Pose/scene_001/camera_%04d/camera_%04d_out_keypoint.json'%(cid,cid))))
            matched=pk.assign_serial_from_tracking_dict(data)
            assert len(matched)==len(data), 'Native BBox-Keypoint mismatch'
            t=time.monotonic(); tracker=Tracker(params)
            raw=tracker.scpt(copy.deepcopy(data))
            save(target/('camera%03d_tracking_results.json'%cid),raw)
            fixed=tracker.correcting_scpt_result(raw)
            save(target/('fixed_camera%03d_tracking_results.json'%cid),fixed)
            timings.append({'camera':cid,'seconds':time.monotonic()-t,'rows':len(fixed),
                            'local_ids':len({r['OfflineID'] for r in fixed.values() if r['OfflineID']>=0})})
        t=time.monotonic()
        run.run_mcpt(1,str(target),str(target),params)
        run.correct_mcpt_result(1,str(target),str(target),params)
        record={'repeat':repeat,'scpt':timings,'mcpt_seconds':time.monotonic()-t,'total_seconds':time.monotonic()-began,
                'fixed_whole_sha256':sha(target/'fixed_whole_tracking_results.json')}
        save(target/'timing.json',record);results.append(record)
    save(WORK/'tracking/repeat_report.json',{'runs':results,'identical_fixed_whole':all(r['fixed_whole_sha256']==results[0]['fixed_whole_sha256'] for r in results) if len(results)>1 else None})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','track'])
    p.add_argument('--short-track-th',type=int,default=40);p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    prepare() if a.phase=='prepare' else track(a.short_track_th,a.repeats)
