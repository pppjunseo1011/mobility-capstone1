#!/usr/bin/env python3
"""Validate bounded real SCOUT clips in the existing CPU adapter image.

No weights, fake predictions, model inference, downloads, or source modifications.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "scout-mtmc/adapter:0.3"
IMAGE_ID = "sha256:5ebc11a7ae782c8e4b6dbef0ea32f19237f05f16531e49b14c441261eb423900"
SELECTIONS = [(1, ["cam_{}".format(number) for number in range(8)])]


def validate_frame_map_keys(rows, manifest, profile):
    base = {"yachiyo": 1, "uwipl": 0, "posetrack": 0}[profile["name"]]
    if profile["frame_base"] != base:
        raise ValueError("Unexpected model frame base")
    expected = {(camera["id"], frame["source_frame"], frame["index"]+base)
                for camera in manifest["cameras"] for frame in camera["frames"]}
    actual = [(row["source_camera"], int(row["source_frame"]), int(row["model_frame"])) for row in rows]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Frame map has duplicate, missing or unexpected camera/frame mappings")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_exact(stream, count):
    parts, size = [], 0
    while size < count:
        part = stream.read(count-size)
        if not part:
            break
        parts.append(part)
        size += len(part)
    return b"".join(parts)


def decoded_video_check(source_dir, video, start, count, width, height, fps):
    import cv2
    import numpy as np
    video_before = {"sha256": sha256(video), "size": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns}
    frame_bytes = width*height*3
    base = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1"]
    suffix = ["-frames:v", str(count), "-an", "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    commands = [base + ["-i", str(video)] + suffix]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for command in commands]
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        for process in processes:
            process.terminate()
            process.communicate()
        raise ValueError("OpenCV cannot open MP4: " + str(video))
    pixel_max = jpeg_max = 0
    pixel_sum = jpeg_sum = 0
    try:
        for index in range(count):
            jpeg = cv2.imread(str(source_dir / ("image_{}.jpg".format(start+index))), cv2.IMREAD_COLOR)
            if jpeg is None or jpeg.shape != (height, width, 3):
                raise ValueError("OpenCV source JPEG decode failed")
            expected = jpeg.tobytes(order="C")
            actual = read_exact(processes[0].stdout, frame_bytes)
            if len(expected) != frame_bytes or len(actual) != frame_bytes or expected != actual:
                raise ValueError("Video pixels/order differ from OpenCV source JPEG: {} frame {}".format(video, index))
            ok, frame = cap.read()
            if not ok or frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError("OpenCV decoded frame count/shape/dtype mismatch")
            expected_array = np.frombuffer(expected, np.uint8).reshape(height, width, 3)
            delta = np.abs(frame.astype(np.int16)-expected_array.astype(np.int16))
            pixel_max = max(pixel_max, int(delta.max()))
            pixel_sum += int(delta.sum())
            delta = np.abs(frame.astype(np.int16)-jpeg.astype(np.int16))
            jpeg_max = max(jpeg_max, int(delta.max()))
            jpeg_sum += int(delta.sum())
            if jpeg_max:
                raise ValueError("OpenCV MP4 differs from direct OpenCV JPEG pixels")
        if cap.read()[0]:
            raise ValueError("OpenCV decoded extra frames")
        for process in processes:
            extra, error = process.communicate(timeout=60)
            if process.returncode or extra:
                raise ValueError("FFmpeg decode failed or produced extra pixels: " + error.decode(errors="replace")[:500])
    finally:
        cap.release()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=30)
    pixels = frame_bytes*count
    video_after = {"sha256": sha256(video), "size": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns}
    if video_before != video_after:
        raise ValueError("Input MP4 changed during source alignment validation")
    return {"decoded_frames": count, "shape": [height, width, 3], "dtype": "uint8",
            "input_video": video_before, "input_video_unchanged": True,
            "opencv_source_to_video_pixels_and_order_exact": True,
            "opencv_mp4_vs_ffmpeg_max_abs_difference": pixel_max,
            "opencv_mp4_vs_ffmpeg_mean_abs_difference": pixel_sum/pixels,
            "opencv_mp4_vs_opencv_jpeg_max_abs_difference": jpeg_max,
            "opencv_mp4_vs_opencv_jpeg_mean_abs_difference": jpeg_sum/pixels,
            "note": "MP4 pixels match cv2.imread(IMREAD_COLOR) source JPEG pixels exactly."}


def inside(data, output, start, frames, selections=None, gt_formats=("mot",)):
    import cv2
    cv2.setNumThreads(1)
    from scout_adapter import __version__
    from scout_adapter.common import load_json, write_json
    from scout_adapter.dataset import build_manifest
    from scout_adapter.export import export_dataset
    from scout_adapter.results import export_ground_truth
    data, output = Path(data), Path(output)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Expected a fresh, empty output mount")
    report = {"adapter_version": __version__, "opencv_version": cv2.__version__,
              "state": "running", "selection": {"start_frame": start, "frame_count": frames},
              "checks": [], "failures": [], "sequences": {}, "actual_model_inference": False,
              "ground_truth_formats_requested": list(gt_formats), "video_decoder": "opencv",
              "synthetic_predictions_created": False, "full_dataset_pixel_validation": False,
              "ground_truth_used_as_model_input": False}
    def save():
        write_json(output / "validation.json", report)
    def step(label, action):
        print("START", label, flush=True)
        began = time.monotonic()
        try:
            value = action()
            report["checks"].append({"name": label, "status": "PASS", "seconds": round(time.monotonic()-began, 2), "result": value})
            print("PASS", label, flush=True)
            return value
        except Exception as error:
            report["failures"].append({"name": label, "error": type(error).__name__ + ": " + str(error)})
            print("FAIL", label, str(error), flush=True)
            return None
        finally:
            save()
    for sequence, cameras in (selections or SELECTIONS):
        manifest = build_manifest(data, sequence, cameras, start=start, maximum=frames)
        if manifest["frame_ids"] != list(range(start, start+frames)):
            raise ValueError("Sample does not contain the requested consecutive frames")
        original = {}
        for camera in manifest["cameras"]:
            for frame in camera["frames"]:
                path = data / frame["source_path"]
                decoded = cv2.imread(str(path))
                if decoded is None or decoded.shape != (camera["height"], camera["width"], 3):
                    raise ValueError("Real JPEG cannot be decoded: " + str(path))
                original[frame["source_path"]] = {"sha256": sha256(path), "mtime_ns": path.stat().st_mtime_ns}
        report["sequences"][str(sequence)] = {
            "cameras": cameras, "frame_ids": manifest["frame_ids"], "sync": manifest["sync"],
            "unique_jpegs_decoded": len(original), "source_files": original}
        exports = {}
        for model in ("yachiyo", "uwipl", "posetrack"):
            export = output / ("sequence{}-{}".format(sequence, model))
            result = step("sequence{} {} export".format(sequence, model), lambda model=model, export=export: export_dataset(
                data, export, model, sequence=sequence, cameras=cameras, start=start, maximum=frames, videos=True, fps=10))
            if result is None:
                continue
            exports[model] = export
            def mapping_check():
                saved = load_json(export / "manifest.json")
                profile = load_json(export / "profile.json")
                with (export / "frame_map.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                validate_frame_map_keys(rows, saved, profile)
                if len(rows) != frames*len(cameras):
                    raise ValueError("Frame map row count mismatch")
                for row in rows:
                    source_frame = int(row["source_frame"])
                    if int(row["model_frame"]) != source_frame-start+profile["frame_base"]:
                        raise ValueError("Model/source frame mapping mismatch")
                    if int(row["model_camera_id"]) != int(row["source_camera"].split("_")[1])+1:
                        raise ValueError("Camera ID mapping mismatch")
                    source = data / row["source_path"]
                    if sha256(export / row["model_frame_path"]) != original[row["source_path"]]["sha256"]:
                        raise ValueError("Copied JPEG differs from original")
                    camera = next(c for c in manifest["cameras"] if c["id"] == row["source_camera"])
                    expected = next(f for f in camera["frames"] if f["source_frame"] == source_frame)
                    if float(row["timestamp_seconds"]) != expected["timestamp_seconds"]:
                        raise ValueError("Timestamp was changed")
                    camera_profile = next(c for c in profile["cameras"] if c["source_camera"] == camera["id"])
                    for directory in camera_profile["frame_dirs"]:
                        copied = export / directory / camera_profile["frame_pattern"].format(frame=int(row["model_frame"]))
                        if sha256(copied) != sha256(source):
                            raise ValueError("Secondary JPEG copy mismatch")
                if saved["image_space"] != "raw" or result["model_inference_ready"]:
                    raise ValueError("Incorrect raw-image/inference-ready claim")
                return {"verified_rows": len(rows), "all_jpeg_copies_byte_identical": True,
                        "source_ids_and_timestamps_preserved": True}
            step("sequence{} {} mapping".format(sequence, model), mapping_check)
            profile = load_json(export / "profile.json")
            for camera in profile["cameras"]:
                info = next(c for c in manifest["cameras"] if c["id"] == camera["source_camera"])
                source_dir = data / "images" / ("sequence_{}".format(sequence)) / info["id"]
                video = export / camera["video_path"]
                step("sequence{} {} {} video".format(sequence, model, info["id"]),
                     lambda: decoded_video_check(source_dir, video, start, frames, info["width"], info["height"], 10))
        if "posetrack" in exports:
            for format in gt_formats:
                destination = output / ("sequence{}-gt-{}".format(sequence, format))
                step("sequence{} ground-truth {}".format(sequence, format),
                     lambda format=format, destination=destination: export_ground_truth(data, exports["posetrack"], destination, format))
        unchanged = all(sha256(data / name) == value["sha256"] and (data / name).stat().st_mtime_ns == value["mtime_ns"]
                        for name, value in original.items())
        report["sequences"][str(sequence)]["selected_sources_unchanged"] = unchanged
        if not unchanged:
            report["failures"].append({"name": "source preservation", "error": "Selected original JPEG was modified"})
        save()
    report["state"] = "passed" if not report["failures"] else "completed_with_failures"
    report["total_unique_jpegs_decoded"] = sum(x["unique_jpegs_decoded"] for x in report["sequences"].values())
    save()
    print(json.dumps({k: report[k] for k in ("state", "failures", "total_unique_jpegs_decoded", "actual_model_inference")}, ensure_ascii=False), flush=True)
    return 0 if report["state"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "inside"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--sequence", type=int, choices=(1, 2))
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--gt-formats", nargs="+", choices=("mot", "coco"), default=["mot"])
    args = parser.parse_args()
    if not 1 <= args.frames <= 600 or args.start_frame < 0:
        parser.error("Select 1–600 frames and a nonnegative start frame")
    if len(set(args.gt_formats)) != len(args.gt_formats):
        parser.error("Select each ground-truth format only once")
    if (args.sequence is None) != (args.cameras is None):
        parser.error("Provide --sequence and --cameras together")
    selections = [(args.sequence, args.cameras)] if args.sequence is not None else SELECTIONS
    if args.action == "inside":
        return inside(args.data, args.output, args.start_frame, args.frames, selections, args.gt_formats)
    data, output = Path(args.data).expanduser().resolve(), Path(args.output).expanduser().absolute()
    if not data.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("Data must exist and output must be a new directory")
    output = output.resolve()
    if data == output or data in output.parents or output in data.parents:
        raise ValueError("Input/output must not overlap")
    for path in (data, output, Path(__file__).resolve()):
        if "," in str(path) or "\n" in str(path):
            raise ValueError("Unsupported Docker mount path")
    actual = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if actual != IMAGE_ID:
        raise ValueError("Adapter image differs from the validated v0.2 image")
    output.mkdir(parents=True, mode=0o700)
    command = ["docker", "run", "--rm", "--pull=never", "--runtime=runc", "--network=none",
               "--user", "{}:{}".format(os.getuid(), os.getgid()), "--cpus=2", "--memory=4g",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--env", "HOME=/tmp",
               "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
               "--mount", "type=bind,src={},dst=/data/scout,readonly".format(data),
               "--mount", "type=bind,src={},dst=/outputs".format(output),
               "--mount", "type=bind,src={},dst=/checks/validate-scout-sample.py,readonly".format(Path(__file__).resolve()),
               "--entrypoint", "python", IMAGE_ID, "/checks/validate-scout-sample.py", "inside",
               "--data", "/data/scout", "--output", "/outputs", "--start-frame", str(args.start_frame), "--frames", str(args.frames)]
    command.extend(["--sequence", str(selections[0][0]), "--cameras"] + selections[0][1])
    command.extend(["--gt-formats"] + args.gt_formats)
    print(json.dumps({"image": IMAGE, "image_id": actual, "data_readonly": str(data), "output": str(output), "command": command}), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
