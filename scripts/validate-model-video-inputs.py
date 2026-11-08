#!/usr/bin/env python3
"""Decode bounded exported clips in the existing model images, without inference.

Host mode uses only the standard library. Containers read inputs and this script
read-only, use no GPU/network, and write a new, separate report directory.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
IMAGES = {
    "yachiyo": ("scout-mtmc/yachiyo-pose:f881fe0", "sha256:9497ec27ae5322f1712ce896937431bf5b4680dec6ebaf55d080f03e09e2aa7f"),
    "uwipl": ("scout-mtmc/uwipl-pose:8e23a69", "sha256:1693f7cca4b64678473250a3650d88c2678d3636613535592db789e59674599a"),
    "posetrack": ("scout-mtmc/posetrack-all:ba727e6", "sha256:da99db46dd7d4cfbbd936899c7f3941f065df1725132d62417d0d5194d05bf3d"),
}
FRAME_BASES = {"yachiyo": 1, "uwipl": 0, "posetrack": 0}


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def within(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe export-relative path: " + str(relative))
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path escapes the input export: " + str(relative))
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jobs(exports, model, count, sequences=(1,), camera_count=8):
    """Fail before launching Docker if any expected camera/frame map is absent."""
    if (not sequences or any(type(sequence) is not int or sequence not in (1, 2) for sequence in sequences)
            or len(set(sequences)) != len(sequences)):
        raise ValueError("Select distinct SCOUT sequences from 1 and 2")
    if type(camera_count) is not int or not 1 <= camera_count <= 25:
        raise ValueError("Select 1–25 cameras per sequence")
    jobs = []
    for sequence in sequences:
        name = "sequence{}-{}".format(sequence, model)
        folder = within(exports, name)
        profile = load_json(within(folder, "profile.json"))
        manifest = load_json(within(folder, "manifest.json"))
        base = FRAME_BASES[model]
        if (profile.get("name") != model or manifest.get("model") != model or
                manifest.get("sequence") != "sequence_{}".format(sequence) or
                profile.get("frame_base") != base or manifest.get("model_frame_base") != base):
            raise ValueError("Mismatched model/sequence/frame base: " + name)
        cameras = profile["cameras"]
        source_cameras = manifest["cameras"]
        if len(cameras) != camera_count or len(source_cameras) != camera_count:
            raise ValueError("Expected exactly {} cameras per export: {}".format(camera_count, name))
        by_camera = {camera["id"]: camera for camera in source_cameras}
        profile_ids = [camera["source_camera"] for camera in cameras]
        if len(by_camera) != camera_count or len(set(profile_ids)) != camera_count or set(profile_ids) != set(by_camera):
            raise ValueError("Camera lists disagree: " + name)
        frame_ids = manifest["frame_ids"]
        if (len(frame_ids) != count or any(type(frame) is not int or frame < 0 for frame in frame_ids) or
                any(right != left + 1 for left, right in zip(frame_ids, frame_ids[1:]))):
            raise ValueError("Expected the requested consecutive source frames: " + name)
        fps = manifest.get("video_fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("No valid explicit video FPS in export: " + name)
        with within(folder, "frame_map.csv").open(newline="", encoding="utf-8") as stream:
            mapping = list(csv.DictReader(stream))
        if len(mapping) != camera_count * count or any(row["source_camera"] not in by_camera for row in mapping):
            raise ValueError("Unexpected frame_map rows: " + name)
        for camera in cameras:
            source = by_camera[camera["source_camera"]]
            if camera["model_camera_id"] != source["model_camera_id"]:
                raise ValueError("Profile/manifest camera IDs disagree: " + name)
            if (source["width"], source["height"]) != (1920, 1080) or len(source["frames"]) != count:
                raise ValueError("Expected 1920x1080 and the requested frame count: " + name)
            rows = [row for row in mapping if row["source_camera"] == source["id"]]
            rows.sort(key=lambda row: int(row["model_frame"]))
            if len(rows) != count:
                raise ValueError("Missing camera frame_map rows: " + source["id"])
            frame_dir = within(folder, camera["frame_dirs"][0])
            frame_pattern = camera["frame_pattern"]
            for index, (row, original) in enumerate(zip(rows, source["frames"])):
                expected_jpeg = within(frame_dir, frame_pattern.format(frame=base + index))
                mapped_jpeg = within(folder, row["model_frame_path"])
                if (int(row["model_frame"]) != base + index or int(row["source_frame"]) != frame_ids[index] or
                        original["index"] != index or original["source_frame"] != frame_ids[index] or
                        row["sequence"] != manifest["sequence"] or
                        int(row["model_camera_id"]) != camera["model_camera_id"] or
                        row["model_camera_name"] != camera["model_camera_name"] or
                        row["source_path"] != original["source_path"] or
                        float(row["timestamp_seconds"]) != original["timestamp_seconds"] or
                        mapped_jpeg != expected_jpeg or not mapped_jpeg.is_file()):
                    raise ValueError("Inconsistent frame mapping: {} {} frame {}".format(name, source["id"], index))
            video = within(folder, camera["video_path"])
            if not video.is_file():
                raise ValueError("Expected MP4 is absent: " + str(video))
            jobs.append({"export": name, "sequence": sequence, "camera": source["id"],
                         "video": str(video.relative_to(Path(exports).resolve())),
                         "frame_dir": str(frame_dir.relative_to(Path(exports).resolve())),
                         "frame_pattern": frame_pattern,
                         "frame_count": count, "frame_base": base, "source_frames": frame_ids,
                         "fps": fps, "width": 1920, "height": 1080})
    return jobs


def read_exact(stream, count):
    parts, size = [], 0
    while size < count:
        part = stream.read(count - size)
        if not part:
            break
        parts.append(part)
        size += len(part)
    return b"".join(parts)


def check_video(exports, job, cv2, mmcv, np, compare_jpeg=False):
    """Match independent sequential reads, not just reported frame counts."""
    video = within(exports, job["video"])
    before = {"sha256": sha256_file(video), "size": video.stat().st_size,
              "mtime_ns": video.stat().st_mtime_ns}
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    process = None
    reader = None
    hashes = []
    shape = (job["height"], job["width"], 3)
    frame_bytes = math.prod(shape)
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV CAP_FFMPEG cannot open video")
        backend = cap.getBackendName()
        reader = mmcv.VideoReader(str(video), cache_capacity=1)
        if len(reader) != job["frame_count"]:
            raise ValueError("MMCV reported frame count differs from manifest")
        reported_fps = {"opencv": cap.get(cv2.CAP_PROP_FPS), "mmcv": float(reader.fps)}
        if any(not math.isfinite(value) or abs(value - job["fps"]) > 1e-3 for value in reported_fps.values()):
            raise ValueError("Video FPS differs from export manifest: " + repr(reported_fps))
        command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
                   "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0", "-threads", "1",
                   "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for index in range(job["frame_count"]):
            reference = read_exact(process.stdout, frame_bytes)
            ok, cv_frame = cap.read()
            mm_frame = reader.read()
            if len(reference) != frame_bytes or not ok or mm_frame is None:
                raise ValueError("Decoder ended before expected frame {}".format(index))
            for name, frame in (("opencv", cv_frame), ("mmcv", mm_frame)):
                if frame.shape != shape or frame.dtype != np.uint8:
                    raise ValueError("{} frame {} shape/dtype mismatch".format(name, index))
            values = {"ffmpeg_bgr_sha256": hashlib.sha256(reference).hexdigest(),
                      "opencv_bgr_sha256": hashlib.sha256(cv_frame.tobytes(order="C")).hexdigest(),
                      "mmcv_bgr_sha256": hashlib.sha256(mm_frame.tobytes(order="C")).hexdigest()}
            if values["opencv_bgr_sha256"] != values["mmcv_bgr_sha256"]:
                raise ValueError("OpenCV/MMCV pixels or frame order differ at frame {}: {}".format(index, values))
            if values["opencv_bgr_sha256"] != values["ffmpeg_bgr_sha256"]:
                expected = np.frombuffer(reference, np.uint8).reshape(shape)
                maximum = int(np.abs(cv_frame.astype(np.int16) - expected.astype(np.int16)).max())
                raise ValueError("FFmpeg/OpenCV pixels or frame order differ at frame {}; max_abs_delta={}".format(index, maximum))
            if compare_jpeg:
                jpeg = within(within(exports, job["frame_dir"]),
                              job["frame_pattern"].format(frame=index + job["frame_base"]))
                jpeg_frame = cv2.imread(str(jpeg), cv2.IMREAD_COLOR)
                if jpeg_frame is None or jpeg_frame.shape != shape or jpeg_frame.dtype != np.uint8:
                    raise ValueError("Export JPEG decode/shape/dtype mismatch at frame {}: {}".format(index, jpeg))
                values["export_jpeg_bgr_sha256"] = hashlib.sha256(jpeg_frame.tobytes(order="C")).hexdigest()
                values["export_jpeg_path"] = str(jpeg.relative_to(Path(exports).resolve()))
                if values["export_jpeg_bgr_sha256"] != values["opencv_bgr_sha256"]:
                    maximum = int(np.abs(jpeg_frame.astype(np.int16) - cv_frame.astype(np.int16)).max())
                    raise ValueError("Export JPEG/OpenCV MP4 pixels or frame order differ at frame {}; "
                                     "max_abs_delta={}".format(index, maximum))
            hashes.append(dict(index=index, model_frame=index + job["frame_base"],
                               source_frame=job["source_frames"][index], **values))
        if cap.read()[0] or reader.read() is not None:
            raise ValueError("OpenCV/MMCV decoded more frames than the frame map")
        extra, error = process.communicate(timeout=60)
        if process.returncode or extra:
            raise ValueError("FFmpeg returned extra frames or failed: " + error.decode(errors="replace")[:500])
    finally:
        cap.release()
        if reader is not None:
            capture = getattr(reader, "vcap", None)
            if capture is None:
                capture = getattr(reader, "_vcap", None)
            if capture is not None:
                capture.release()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
    after = {"sha256": sha256_file(video), "size": video.stat().st_size,
             "mtime_ns": video.stat().st_mtime_ns}
    if before != after:
        raise ValueError("Input MP4 changed while validating")
    return {"decoded_frames_per_reader": len(hashes), "shape": list(shape), "dtype": "uint8",
            "opencv_backend": backend, "reported_fps": reported_fps,
            "opencv_mmcv_pixels_and_order_exact": True,
            "ffmpeg_opencv_pixels_and_order_exact": True, "frame_map_checked": True,
            "export_jpeg_comparison_requested": compare_jpeg,
            "export_jpeg_opencv_pixels_and_order_exact": True if compare_jpeg else None,
            "input_video_unchanged": True, "input_video": before, "frames": hashes}


def inside(args):
    import cv2
    import mmcv
    import numpy as np
    if shutil.which("ffmpeg") is None:
        raise ValueError("The existing image must provide ffmpeg")
    cv2.setNumThreads(1)
    exports = Path(args.exports_root).resolve()
    jobs = load_jobs(exports, args.model, args.frames, args.sequences, args.camera_count)
    output = Path(args.output)
    report_path = output / (args.model + ".json")
    if report_path.exists():
        raise ValueError("Refusing to overwrite a decoder report")
    report = {"model": args.model, "state": "running", "opencv_version": cv2.__version__,
              "mmcv_version": mmcv.__version__, "numpy_version": np.__version__,
              "cpu_only": True, "model_inference_run": False, "weights_loaded": False,
              "checks": [], "failures": [], "sequences": args.sequences,
              "cameras_per_sequence": args.camera_count, "frames_per_video": args.frames,
              "scope": "MP4 reader compatibility and encoded frame order in existing model images",
              "source_jpeg_to_video_alignment_retested": False,
              "export_jpeg_comparison_requested": args.compare_jpeg,
              "export_jpeg_to_video_alignment_verified": False,
              "note": "Original source JPEGs are not mounted here; their copied-file integrity requires separate validation. "
                      "--compare-jpeg compares each exported JPEG decoded by this image's OpenCV with the mapped MP4 frame. "
                      "Without that flag, only frame-map numbering and sequential MP4 readers are compared; "
                      "it does not repair model-specific frame offsets, synchronize cameras, or run inference."}
    write_json(report_path, report)
    for job in jobs:
        label = "{} sequence{} {}".format(args.model, job["sequence"], job["camera"])
        began = time.monotonic()
        try:
            result = check_video(exports, job, cv2, mmcv, np, compare_jpeg=args.compare_jpeg)
            report["checks"].append({"job": job, "status": "PASS", "seconds": round(time.monotonic() - began, 2), "result": result})
            print("PASS", label, "decoded", job["frame_count"], "frames with three matching readers",
                  "and exact export JPEG pixels" if args.compare_jpeg else "(JPEG comparison not requested)", flush=True)
        except Exception as error:
            report["failures"].append({"job": job, "error": type(error).__name__ + ": " + str(error)})
            print("FAIL", label, str(error), flush=True)
        write_json(report_path, report)
    report["state"] = "passed" if not report["failures"] else "completed_with_failures"
    report["videos_passed"] = len(report["checks"])
    report["export_jpeg_to_video_alignment_verified"] = bool(args.compare_jpeg and not report["failures"])
    write_json(report_path, report)
    return 0 if report["state"] == "passed" else 1


def docker_command(model, exports, output, frames, sequences=(1,), camera_count=8, compare_jpeg=False):
    return ["docker", "run", "--rm", "--pull=never", "--runtime=runc", "--network=none",
            "--user", "{}:{}".format(os.getuid(), os.getgid()), "--cpus=2", "--memory=4g",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
            "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
            "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1",
            "--mount", "type=bind,src={},dst=/exports,readonly".format(exports),
            "--mount", "type=bind,src={},dst=/outputs".format(output),
            "--mount", "type=bind,src={},dst=/checks/validate-model-video-inputs.py,readonly".format(Path(__file__).resolve()),
            "--entrypoint", "python", IMAGES[model][1], "/checks/validate-model-video-inputs.py", "inside",
            "--exports-root", "/exports", "--output", "/outputs", "--model", model, "--frames", str(frames),
            "--sequences", *[str(sequence) for sequence in sequences], "--camera-count", str(camera_count)
            ] + (["--compare-jpeg"] if compare_jpeg else [])


def run(args):
    exports = Path(args.exports_root).expanduser().resolve()
    requested = Path(args.output).expanduser().absolute()
    output = requested.resolve()
    if not exports.is_dir():
        raise ValueError("Export directory is absent: " + str(exports))
    if requested.exists() or requested.is_symlink():
        raise ValueError("Choose a new output directory; existing logs are preserved")
    if exports == output or exports in output.parents or output in exports.parents:
        raise ValueError("Input exports and output logs must not overlap")
    for path in (exports, output, Path(__file__).resolve()):
        if "," in str(path) or "\n" in str(path):
            raise ValueError("Unsupported Docker bind-mount path")
    models = list(IMAGES) if args.model == "all" else [args.model]
    jobs = {model: load_jobs(exports, model, args.frames, args.sequences, args.camera_count) for model in models}
    commands = {model: docker_command(model, exports, output, args.frames, args.sequences, args.camera_count,
                                     compare_jpeg=args.compare_jpeg) for model in models}
    if args.dry_run:
        for model in models:
            print(shlex.join(commands[model]))
        return 0
    for model in models:
        tag, expected = IMAGES[model]
        actual = subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip()
        if actual != expected:
            raise ValueError("Image ID differs from the validated image: " + tag)
    output.mkdir(parents=True, mode=0o700)
    summary = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "state": "running", "exports_root": str(exports), "frames_per_video": args.frames,
               "sequences": args.sequences, "cameras_per_sequence": args.camera_count,
               "width": 1920, "height": 1080, "images": {}, "results": {}, "failures": [],
               "model_inference_run": False, "weights_loaded": False, "cpu_only": True,
               "export_jpeg_comparison_requested": args.compare_jpeg,
               "export_jpeg_to_video_alignment_verified": False,
               "original_models_modified": False, "images_modified": False}
    for model in models:
        summary["images"][model] = {"tag": IMAGES[model][0], "verified_id": IMAGES[model][1],
                                    "video_count": len(jobs[model]), "command": commands[model]}
    write_json(output / "validation.json", summary)
    for model in models:
        log_path = output / (model + ".log")
        print("START", model, "log=" + str(log_path), flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            log.write("$ " + shlex.join(commands[model]) + "\n")
            log.flush()
            status = subprocess.call(commands[model], stdout=log, stderr=subprocess.STDOUT)
            log.write("\nexit_code={}\n".format(status))
        report_path = output / (model + ".json")
        report = load_json(report_path) if report_path.is_file() else {}
        passed = status == 0 and report.get("state") == "passed" and report.get("videos_passed") == len(jobs[model])
        if args.compare_jpeg:
            passed = passed and report.get("export_jpeg_to_video_alignment_verified") is True
        summary["results"][model] = {"status": "PASS" if passed else "FAIL", "exit_code": status,
                                      "report": report_path.name, "log": log_path.name,
                                      "videos_passed": report.get("videos_passed", 0)}
        if not passed:
            summary["failures"].append(model)
        print("PASS" if passed else "FAIL", model, flush=True)
        write_json(output / "validation.json", summary)
    summary["state"] = "passed" if not summary["failures"] else "completed_with_failures"
    summary["export_jpeg_to_video_alignment_verified"] = bool(args.compare_jpeg and not summary["failures"])
    summary["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(output / "validation.json", summary)
    return 0 if summary["state"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "inside"))
    parser.add_argument("--exports-root", default=str(ROOT / "outputs/scout-first8-validation-20261027"))
    parser.add_argument("--output", required=True, help="New, separate log/report directory")
    parser.add_argument("--model", choices=("all", "yachiyo", "uwipl", "posetrack"), default="all")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--sequences", type=int, nargs="+", choices=(1, 2), default=[1],
                        help="SCOUT sequence numbers to validate (default: 1)")
    parser.add_argument("--camera-count", type=int, default=8,
                        help="Exact number of cameras required in each export (default: 8)")
    parser.add_argument("--compare-jpeg", action="store_true",
                        help="Also require exact pixels/order between OpenCV-decoded exported JPEGs and MP4 frames")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 600:
        parser.error("Select 1–600 frames per video")
    if len(set(args.sequences)) != len(args.sequences):
        parser.error("Select each sequence only once")
    if not 1 <= args.camera_count <= 25:
        parser.error("Select 1–25 cameras per sequence")
    if args.action == "inside" and (args.model == "all" or args.dry_run):
        parser.error("inside requires exactly one model and does not support --dry-run")
    return inside(args) if args.action == "inside" else run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print("Video-input validation error: " + str(error), file=sys.stderr)
        sys.exit(1)
