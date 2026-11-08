"""Dependency/extension checks only; no checkpoints, datasets or model inference."""
import argparse
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
a = p.parse_args()
import numpy as np
import torch
import torchvision
import cv2
import mmcv
import mmengine
import mmpose
import mmdet
import faiss
import aic_cpp
import lap
from cython_bbox import bbox_overlaps
from cvxopt import glpk, matrix
from mmcv.ops import nms as mmcv_nms
from torchvision.ops import nms, roi_align
from yolox.exp import get_exp
from yolox.utils import postprocess
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.engine import DefaultTrainer
from fastreid.evaluation.rank_cylib import rank_cy, roc_cy
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from Tracker.PoseTracker import PoseTracker
from Solver.bip_solver import GLPKSolver

# Import real entrypoints without executing their __main__ blocks.
for path in ('detection/get_detection.py', 'fast-reid/tools/infer.py',
             'mmpose/demo/save_pose_with_det_multiscene.py', 'track/run_tracking_batch.py'):
    spec = importlib.util.spec_from_file_location('smoke_' + path.replace('/', '_'), '/opt/posetrack/' + path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
register_all_modules(init_default_scope=False)
assert hasattr(faiss, 'StandardGpuResources'), 'GPU-capable FAISS API missing'
idx = faiss.IndexFlatL2(2)
idx.add(np.array([[0., 0.], [1., 1.]], dtype=np.float32))
assert idx.search(np.array([[0., 0.]], dtype=np.float32), 1)[1].item() == 0
boxes_np = np.array([[0., 0., 10., 10.]], dtype=np.float64)
np.testing.assert_allclose(aic_cpp.bbox_overlap_rate(boxes_np, boxes_np), [[1.]])
np.testing.assert_allclose(bbox_overlaps(boxes_np, boxes_np), [[1.]])
assert lap.lapjv(np.array([[1., 5.], [4., 1.]]))[0] == 2.
status, solution = glpk.ilp(matrix([1.]), matrix([-1.], (1, 1)), matrix([-1.]), I={0})
assert status == 'optimal' and abs(solution[0] - 1) < 1e-6
if a.device == 'cuda':
    assert torch.cuda.is_available(), 'CUDA requested but unavailable'
    assert torch.cuda.device_count() == 1, 'Expose only one allocated GPU for this smoke test'
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(16 * 1024 * 1024)
    gpu_index = faiss.index_cpu_to_gpu(resources, 0, idx)
    assert gpu_index.search(np.array([[0., 0.]], dtype=np.float32), 1)[1].item() == 0
device = torch.device(a.device)
x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
assert torch.allclose(x @ torch.eye(4, device=device), x)
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
scores = torch.tensor([0.9, 0.5], device=device)
assert nms(boxes, scores, 0.5).numel() == 1
assert mmcv_nms(boxes, scores, 0.5)[1].numel() == 1
roi = torch.tensor([[0., 0., 0., 3., 3.]], device=device)
assert roi_align(x[None, None], roi, (2, 2)).shape == (1, 1, 2, 2)
if a.device == 'cuda':
    torch.cuda.synchronize()
print(json.dumps({'status': 'PASS', 'scope': 'environment only; no model inference',
    'python': platform.python_version(), 'torch': torch.__version__,
    'torchvision': torchvision.__version__, 'torch_cuda': torch.version.cuda,
    'mmcv': mmcv.__version__, 'mmengine': mmengine.__version__,
    'mmpose': mmpose.__version__, 'mmdet': mmdet.__version__,
    'numpy': np.__version__, 'device': a.device,
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if a.device == 'cuda' else None,
    'extensions': ['aic_cpp', 'cython_bbox', 'lap', 'GLPK', 'faiss-gpu API',
                   'FastReID rank_cy/roc_cy', 'torchvision NMS/ROIAlign', 'mmcv NMS']}, indent=2))
