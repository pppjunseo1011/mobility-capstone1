"""Bounded environment checks; no checkpoint fetch, dataset or model inference."""
import argparse
import importlib
import json
import os
import platform
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()
stage = os.environ["SCOUT_STAGE"]
report = {"stage": stage, "device": args.device, "python": platform.python_version(), "imports": []}


def check(name):
    module = importlib.import_module(name)
    report["imports"].append(name)
    return module


np = check("numpy")
check("scipy")
check("cv2")
if stage == "tracking":
    if args.device == "cuda":
        raise SystemExit("This stage is CPU-only; CUDA checks belong to detection/reid/pose.")
    sys.path.insert(0, "/opt/yachiyo/tracking/src")
    check("scpt")
    check("mcpt")
    check("tracking")
    from sklearn.cluster import AgglomerativeClustering
    labels = AgglomerativeClustering(n_clusters=2).fit_predict(np.array([[0., 0.], [.1, 0.], [5., 5.]]))
    assert labels[0] == labels[1] and labels[0] != labels[2]
else:
    torch = check("torch")
    torchvision = check("torchvision")
    report.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_build=torch.version.cuda)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA unavailable"
        report["gpu"] = torch.cuda.get_device_name(0)
        report["capability"] = list(torch.cuda.get_device_capability(0))
    a = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
    assert torch.isfinite(a @ a.T).all().item()
    boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]], device=device)
    scores = torch.tensor([.9, .8], device=device)
    assert torchvision.ops.nms(boxes, scores, .5).tolist() == [0]
    if stage == "detection":
        for name in ("yolox._C", "yolox.exp", "yolox.utils", "cython_bbox", "lap", "faiss",
                     "tracker.bot_sort", "fast_reid.fastreid.evaluation.rank_cylib.rank_cy",
                     "fast_reid.fastreid.evaluation.rank_cylib.roc_cy", "tools.aic24_get_detection"):
            check(name)
        from cython_bbox import bbox_overlaps
        b = np.array([[0., 0., 4., 4.]], dtype=np.float64)
        assert bbox_overlaps(b, b)[0, 0] == 1.
        import lap
        assert lap.lapjv(np.eye(2, dtype=np.float64))[0] == 0.
    elif stage == "reid":
        check("torchreid")
        check("torchreid.metrics.rank_cylib.rank_cy")
        check("torchreid.utils.feature_extractor")
        check("torchreid.aic24_extract")
    elif stage == "pose":
        for name in ("mmcv._ext", "mmdet", "mmpose", "mmpose.apis", "xtcocotools._mask"):
            check(name)
        from mmcv.ops import nms
        assert nms(boxes, scores, .5)[1].tolist() == [0]
        sys.path.insert(0, "/opt/yachiyo/mmpose/demo")
        check("top_down_video_demo_with_track_file")
    else:
        raise AssertionError("Unknown stage: " + stage)
    if args.device == "cuda":
        torch.cuda.synchronize()
report["status"] = "PASS"
print(json.dumps(report, indent=2))
