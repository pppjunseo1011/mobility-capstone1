"""No weights, videos or model inference: imports, native ops, tiny torch arithmetic."""
import argparse
import importlib
import json
import os
import platform

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()
import numpy as np
import torch
import torchvision
import cv2

torch.set_num_threads(2)
stage = os.environ['SCOUT_STAGE']
mods = {
    'botsort': ['yolox', 'yolox._C', 'yolox.exp', 'yolox.data', 'tracker.bot_sort', 'tracker.matching', 'cython_bbox', 'lap', 'cam_utils', 'file_utils', 'tools.aic_track', 'tools.aic_get_detection_S001', 'tools.aic_hungarian_cluster', 'tools.aic_hungarian_cluster_S001', 'run_stcra'],
    'reid': ['torchreid', 'torchreid.metrics.rank_cylib.rank_cy', 'torchreid.utils.feature_extractor', 'torchreid.aic_extract', 'torchreid.aic_extract_S001'],
    'mmyolo': ['mmcv', 'mmcv._ext', 'mmengine', 'mmdet', 'mmdet.apis', 'mmyolo', 'mmyolo.models', 'mmyolo.datasets', 'tools.aic_get_detection_syn'],
    'pose': ['mmcv', 'mmcv._ext', 'mmdet', 'mmdet.apis', 'mmpose', 'mmpose.apis', 'mmpose.datasets', 'xtcocotools._mask', 'demo.load_tracking_result', 'demo.top_down_video_demo_with_track_file'],
}[stage]
for name in mods:
    importlib.import_module(name)
    print('IMPORT_OK', name, flush=True)
if args.device == 'cuda':
    assert torch.cuda.is_available(), 'CUDA requested but not available'
    assert torch.cuda.device_count() == 1, 'Smoke expects exactly one explicitly assigned GPU'
device = torch.device(args.device)
a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
assert torch.allclose((a @ torch.eye(4, device=device)).cpu(), a.cpu())
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
scores = torch.tensor([0.9, 0.8], device=device)
assert torchvision.ops.nms(boxes, scores, 0.5).numel() == 1
if stage in ['mmyolo', 'pose']:
    from mmcv.ops import nms
    _, keep = nms(boxes, scores, 0.5)
    assert keep.numel() == 1
if stage == 'botsort':
    from cython_bbox import bbox_overlaps
    from tracker.matching import linear_assignment
    b = np.array([[0., 0., 10., 10.]], dtype=np.float64)
    assert bbox_overlaps(b, b)[0, 0] == 1
    matches, _, _ = linear_assignment(np.array([[0.1]], dtype=np.float64), thresh=0.5)
    assert matches.shape == (1, 2)
if stage == 'reid':
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    cmc, mean_ap = evaluate_cy(np.array([[0.1, 0.8]], dtype=np.float32),
                               np.array([1]), np.array([1, 2]),
                               np.array([0]), np.array([1, 1]), 2)
    assert cmc[0] == 1.0 and mean_ap == 1.0
if args.device == 'cuda':
    torch.cuda.synchronize()
print(json.dumps({'status': 'PASS', 'stage': stage, 'device': args.device,
                  'python': platform.python_version(), 'torch': torch.__version__,
                  'torchvision': torchvision.__version__, 'cuda_build': torch.version.cuda,
                  'numpy': np.__version__, 'opencv': cv2.__version__,
                  'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  'model_inference': 'NOT_RUN'}, sort_keys=True))
