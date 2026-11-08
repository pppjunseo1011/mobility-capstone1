#!/usr/bin/env python3
"""Bounded real checkpoint inference inside existing model containers, without GT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def finite(value):
    assert np.isfinite(value).all(), 'Non-finite model output'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['posetrack', 'yachiyo', 'uwipl'], required=True)
    p.add_argument('--stage', choices=['detection', 'pose', 'reid'], required=True)
    p.add_argument('--max-people-per-frame', type=int, default=10)
    args = p.parse_args()
    assert args.max_people_per_frame >= 0
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    out = Path('/outputs')
    root = Path('/opt') / args.model
    frames = json.loads(Path('/sample/input/manifest.json').read_text())['frames']
    assert len(frames) <= 4800
    started = time.monotonic()
    report = {'model': args.model, 'stage': args.stage, 'status': 'RUNNING',
              'frames': len(frames), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'gt_used': False,
              'multicamera_tracking_run': False, 'checkpoint_loads': []}

    def load_state(net, path):
        state = torch.load(str(path), map_location='cpu')
        state = state.get('state_dict', state.get('model', state))
        state = {k.removeprefix('module.') if hasattr(k, 'removeprefix') else (k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        own = net.state_dict()
        mismatches = [k for k in state if k in own and state[k].shape != own[k].shape]
        allowed = [k for k in mismatches if k.startswith('classifier.')]
        if set(mismatches) != set(allowed):
            raise RuntimeError('Unexpected checkpoint tensor shape mismatch: ' + str(mismatches))
        for k in allowed:
            del state[k]
        result = net.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith('classifier.')]
        assert not missing and not result.unexpected_keys, (missing, result.unexpected_keys)
        report['checkpoint_loads'].append({'file': path.name, 'matched_keys': len(state),
                                           'excluded_classifier_keys': allowed,
                                           'missing_keys': list(result.missing_keys),
                                           'unexpected_keys': list(result.unexpected_keys)})

    if args.stage == 'detection':
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        subdir = root / ('detection' if args.model == 'posetrack' else 'BoT-SORT')
        exp = get_exp(str(subdir / 'yolox/exps/example/mot/yolox_x_mix_det.py'), None)
        net = exp.get_model().eval()
        load_state(net, Path('/weights/bytetrack_x_mot17.pth.tar'))
        net.cuda()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if args.model == 'posetrack':
            preproc = module(subdir / 'get_detection.py', 'native_detection').preproc
        else:
            from yolox.data.data_augment import preproc
        results = []
        for frame in frames:
            img = cv2.imread('/sample/input/' + frame['file'])
            assert img is not None and img.shape == (1080, 1920, 3)
            if args.model == 'posetrack':
                tensor, ratio = preproc([img], exp.test_size, np.array(mean, np.float32), np.array(std, np.float32))
            else:
                tensor, ratio = preproc(img, exp.test_size, mean, std)
                tensor = tensor[None]
            with torch.no_grad():
                det = postprocess(net(torch.from_numpy(tensor).cuda()), exp.num_classes, exp.test_conf, exp.nmsthre)[0]
            boxes = np.empty((0, 7), dtype=np.float32) if det is None else det.cpu().numpy()
            boxes[:, :4] /= ratio
            boxes = boxes[boxes[:, 4] > 0.1]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img.shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img.shape[0])
            boxes[:, :4] = np.trunc(boxes[:, :4])
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            finite(boxes)
            selected = boxes[boxes[:, 4] >= 0.3]
            selected = selected[np.argsort(-selected[:, 4], kind='stable')]
            if args.max_people_per_frame:
                selected = selected[:args.max_people_per_frame]
            results.append(dict(frame, detections=boxes.tolist(), selected=selected.tolist()))
            vis = img.copy()
            for i, box in enumerate(selected):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, '%d %.2f' % (i, box[4]), (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(out / (frame['key'] + '.jpg')), vis)
            print('DETECTION', frame['key'], len(boxes), 'selected', len(selected), flush=True)
        (out / 'detections.json').write_text(json.dumps(results, indent=2))
        report.update(detections=sum(len(x['detections']) for x in results), selected=sum(len(x['selected']) for x in results),
                      input_size=list(exp.test_size), nms=float(exp.nmsthre), native_test_conf=float(exp.test_conf),
                      objectness_filter=0.1, downstream_objectness_filter=0.3,
                      max_people_per_frame=args.max_people_per_frame or None,
                      boxes_clipped_to_image=True, boxes_truncated_to_integer=True)
        assert report['selected'] > 0, 'No people selected; cannot validate downstream inference'
    else:
        detections = json.loads((Path('/sample') / args.model / 'detection/detections.json').read_text())
        assert len(detections) == len(frames)
        if args.stage == 'pose':
            if args.model == 'posetrack':
                from mmpose.apis import init_model, inference_topdown
                config = root / 'mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
                weight = Path('/weights/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth')
                net = init_model(str(config), checkpoint=None, device='cpu')
            else:
                from mmpose.apis import init_pose_model, inference_top_down_pose_model
                from mmpose.datasets import DatasetInfo
                config = root / 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
                weight = Path('/weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
                net = init_pose_model(str(config), checkpoint=None, device='cpu')
            load_state(net, weight)
            net.cuda().eval()
            records, native = [], {}
            for frame in detections:
                img = cv2.imread('/sample/input/' + frame['file'])
                boxes = np.array(frame['selected'], dtype=np.float32).reshape(-1, 7)
                keypoints = []
                if len(boxes):
                    with torch.no_grad():
                        if args.model == 'posetrack':
                            preds = inference_topdown(net, img, boxes[:, :4])
                            keypoints = [np.concatenate((q.pred_instances.keypoints[0], q.pred_instances.keypoint_scores[0, :, None]), axis=1) for q in preds]
                        else:
                            preds, _ = inference_top_down_pose_model(net, img,
                                [{'bbox': b[:5]} for b in boxes], bbox_thr=None, format='xyxy',
                                dataset=net.cfg.data.test.type, dataset_info=DatasetInfo(net.cfg.data.test.dataset_info), return_heatmap=False)
                            keypoints = [q['keypoints'] for q in preds]
                kpts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 17, 3)
                assert kpts.shape[0] == len(boxes)
                finite(kpts)
                records.append({'key': frame['key'], 'camera': frame['camera'], 'source_frame': frame['source_frame'],
                                'keypoints': kpts.tolist(), 'bbox_xyxy_score': boxes[:, :5].tolist()})
                vis = img.copy()
                for person in kpts:
                    for x, y, score in person:
                        if score >= 0.3:
                            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
                cv2.imwrite(str(out / (frame['key'] + '.jpg')), vis)
                if args.model == 'posetrack':
                    group = native.setdefault(frame['camera'], {'det': [], 'pose': []})
                    for box, kp in zip(boxes, kpts):
                        group['det'].append([frame['index'], 1] + box[:5].tolist())
                        group['pose'].append([frame['index']] + box[:5].tolist() + kp.flatten().tolist())
                print('POSE', frame['key'], len(kpts), flush=True)
            (out / 'poses.json').write_text(json.dumps(records))
            for camera, rows in native.items():
                np.savetxt(out / (camera + '-det7.txt'), np.asarray(rows['det']).reshape(-1, 7), delimiter=',')
                np.savetxt(out / (camera + '-pose57.txt'), np.asarray(rows['pose']).reshape(-1, 57))
            report['people'] = sum(len(r['keypoints']) for r in records)
            assert report['people'] > 0
        else:
            if args.model == 'posetrack':
                net = torch.load('/weights/aic24.pkl', map_location='cpu').cuda().eval()
                native = module(root / 'fast-reid/tools/infer.py', 'native_reid')
                infer = native.reid_inferencer(net)
                report['checkpoint_loads'].append({'file': 'aic24.pkl', 'format': 'serialized_module', 'class': type(net).__name__})
                models = [('aic24', infer)]
            else:
                from torchreid.utils import FeatureExtractor
                configs = [('osnet_x1_0', 'osnet_ms_m_c.pth.tar')]
                if args.model == 'uwipl':
                    configs += [('osnet_ibn_x1_0', 'osnet_ibn_ms_m_c.pth.tar'), ('osnet_ain_x1_0', 'osnet_ain_ms_m_c.pth.tar')]
                    configs += [('osnet_x1_0', p.name) for p in sorted(Path('/weights').glob('osnet_x1_0_*.pth'))]
                    assert len(configs) == 5
                models = []
                for arch, name in configs:
                    path = Path('/weights') / name
                    assert path.is_file()
                    infer = FeatureExtractor(model_name=arch, model_path=str(path), device='cuda', verbose=False)
                    load_state(infer.model, path)
                    models.append((name.split('.')[0], infer))
            report['embeddings'] = []
            for name, infer in models:
                features, mapping = [], []
                for frame in detections:
                    img = cv2.imread('/sample/input/' + frame['file'])
                    boxes = np.asarray(frame['selected'], dtype=np.float32).reshape(-1, 7)
                    if not len(boxes):
                        continue
                    with torch.no_grad():
                        if args.model == 'posetrack':
                            emb = infer.process_frame_simplified(img, boxes[:, :4])
                        else:
                            from PIL import Image
                            import torchvision.transforms as T
                            transform = T.Compose([T.Resize([256, 128]), T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
                            rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            crops = torch.stack([transform(rgb.crop(tuple(b[:4]))) for b in boxes])
                            emb = infer(crops).cpu().numpy()
                    finite(emb)
                    assert emb.ndim == 2 and emb.shape[0] == len(boxes)
                    assert (np.linalg.norm(emb, axis=1) > 0).all()
                    features.append(emb)
                    mapping.extend([dict(key=frame['key'], box_index=i, source_frame=frame['source_frame'], camera=frame['camera']) for i in range(len(boxes))])
                    print('REID', name, frame['key'], list(emb.shape), flush=True)
                features = np.concatenate(features)
                np.save(out / (name + '.npy'), features)
                (out / (name + '-rows.json')).write_text(json.dumps(mapping))
                report['embeddings'].append({'checkpoint': name, 'shape': list(features.shape), 'finite': True, 'nonzero': True})
    torch.cuda.synchronize()
    report.update(status='PASS', elapsed_seconds=time.monotonic() - started,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
