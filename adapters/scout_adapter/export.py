"""Materialize a traceable input workspace; never invoke a model."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import tempfile
import time

from . import __version__
from .common import finite, new_output, within, write_csv, write_json
from .dataset import build_manifest
from .profiles import PROFILES

FRAME_FIELDS = ["sequence", "source_camera", "model_camera_id", "model_camera_name", "model_frame",
                "source_frame", "timestamp_seconds", "source_path", "model_frame_path"]


def _opencv_to_ffmpeg(frame_dir, pattern, base, count, width, height, command, timeout):
    """Stream one OpenCV BGR frame at a time, with a deadline on pipe writes."""
    try:
        import cv2
    except ImportError as error:
        raise ValueError("OpenCV video export needs cv2; use the adapter Docker image") from error
    previous_threads = cv2.getNumThreads()
    process = None
    digest = hashlib.sha256()
    deadline = time.monotonic() + timeout
    cv2.setNumThreads(1)
    try:
        # A file keeps stderr from filling a pipe while raw frames are written.
        with tempfile.TemporaryFile() as errors, selectors.DefaultSelector() as selector:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=errors, bufsize=0)
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
            for index in range(count):
                source = frame_dir / pattern.format(frame=base + index)
                frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("OpenCV cannot decode JPEG frame: " + str(source))
                if frame.shape != (height, width, 3) or frame.dtype.name != "uint8":
                    raise ValueError("OpenCV JPEG dimensions/dtype differ from the manifest: " + str(source))
                # At most one decoded frame plus its byte buffer is retained.
                pixels = frame.tobytes(order="C")
                digest.update(pixels)
                pending = memoryview(pixels)
                while pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    if process.poll() is not None:
                        raise ValueError("FFmpeg exited before all OpenCV frames were written")
                    if not selector.select(timeout=min(remaining, 1.0)):
                        continue
                    try:
                        written = os.write(process.stdin.fileno(), pending[:1024**2])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError as error:
                        raise ValueError("FFmpeg stopped accepting OpenCV frame data") from error
                    if written <= 0:
                        raise ValueError("FFmpeg rawvideo pipe accepted no data")
                    pending = pending[written:]
                del frame, pixels, pending
            selector.unregister(process.stdin)
            process.stdin.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            status = process.wait(timeout=remaining)
            if status:
                errors.seek(max(0, errors.tell() - 4096))
                detail = errors.read(4096).decode(errors="replace")
                raise ValueError("FFmpeg failed while encoding OpenCV frames: " + detail)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            with contextlib.suppress(OSError):
                process.stdin.close()
        cv2.setNumThreads(previous_threads)
    return {"opencv_version": cv2.__version__, "source_bgr_sha256": digest.hexdigest(),
            "pixel_reference": "cv2.imread(IMREAD_COLOR)", "input_pixel_format": "bgr24"}


def make_video(frame_dir, pattern, base, count, fps, output, width, height, video_decoder="opencv"):
    """Keep JPEG bytes unchanged and choose the decoder defining MP4 pixels.

    The default matches the model images' OpenCV JPEG reader. ``ffmpeg`` keeps
    the previous image-sequence decoding path available for reproducibility.
    """
    if video_decoder not in ("opencv", "ffmpeg"):
        raise ValueError("video_decoder must be opencv or ffmpeg")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ValueError("Video export needs ffmpeg/ffprobe; use the adapter Docker image")
    frame_dir, output = Path(frame_dir), Path(output)
    match = re.fullmatch(r"\{frame:0([1-9][0-9]*)d\}\.jpg", pattern)
    if not match:
        raise ValueError("Unsupported frame pattern for video export")
    if any(type(value) is not int or value <= 0 for value in (count, width, height)):
        raise ValueError("Video frame count and dimensions must be positive integers")
    if type(base) is not int or base < 0 or finite(fps, "fps") <= 0:
        raise ValueError("Video frame base must be nonnegative and fps must be positive")
    if output.exists() or output.is_symlink():
        raise ValueError("Video output already exists; refusing overwrite: " + str(output))
    source_pattern = str(frame_dir / ("%0" + match.group(1) + "d.jpg"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(120, count * 5)
    decoder_info = {"opencv_version": None, "source_bgr_sha256": None,
                    "pixel_reference": "ffmpeg JPEG decoder", "input_pixel_format": "image2"}
    # Commit only after ffprobe succeeds; an interrupted/failed encode leaves
    # no published MP4, and a pre-existing destination can never be replaced.
    with tempfile.TemporaryDirectory(prefix="." + output.name + "-", dir=str(output.parent)) as temporary:
        encoded = Path(temporary) / "video.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-threads", "1"]
        if video_decoder == "opencv":
            command += ["-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "{}x{}".format(width, height),
                        "-framerate", str(fps), "-i", "pipe:0"]
        else:
            command += ["-framerate", str(fps), "-start_number", str(base), "-i", source_pattern]
        command += ["-frames:v", str(count), "-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
                    "-pix_fmt", "bgr24", "-filter_threads", "1", "-threads", "1", str(encoded)]
        if video_decoder == "opencv":
            decoder_info = _opencv_to_ffmpeg(frame_dir, pattern, base, count, width, height, command, timeout)
        else:
            subprocess.run(command, check=True, timeout=timeout)
        result = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0", "-count_frames",
                                 "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json",
                                 str(encoded)], capture_output=True, text=True, check=True, timeout=timeout)
        stream = json.loads(result.stdout)["streams"][0]
        if (stream["width"], stream["height"], int(stream["nb_read_frames"])) != (width, height, count):
            raise ValueError("Encoded video dimensions/frame count differ from the frame map")
        os.link(encoded, output)
    return {"codec": "libx264rgb", "crf": 0, "fps": fps, "frames": count,
            "width": width, "height": height, "decoded_frame_count_verified": True,
            "video_decoder": video_decoder, **decoder_info}


def export_dataset(root, output, model, sequence=1, cameras=None, start=0, maximum=32,
                   overrides=None, tolerance=0.05, materialize="copy", videos=False, fps=None,
                   video_decoder="opencv"):
    if model not in PROFILES:
        raise ValueError("Unknown model: " + model)
    if materialize not in ("copy", "manifest"):
        raise ValueError("materialize must be copy or manifest")
    if video_decoder not in ("opencv", "ffmpeg"):
        raise ValueError("video_decoder must be opencv or ffmpeg")
    if videos and (materialize != "copy" or fps is None):
        raise ValueError("--videos requires --materialize copy and explicit --fps")
    if fps is not None and finite(fps, "fps") <= 0:
        raise ValueError("fps must be positive")
    manifest = build_manifest(root, sequence, cameras, start, maximum, overrides, tolerance)
    if videos and any(b-a != 1 for a, b in zip(manifest["frame_ids"], manifest["frame_ids"][1:])):
        raise ValueError("Video export requires consecutive source frames; sparse timelines need explicit resampling")
    profile = PROFILES[model].build_profile(manifest)
    manifest.update(model=model, model_frame_base=profile["frame_base"], model_scene=profile["scene_name"],
                    materialize=materialize, video_fps=fps if videos else None,
                    video_decoder=video_decoder if videos else None)
    root = Path(manifest["source_root"])
    by_camera = {row["id"]: row for row in manifest["cameras"]}
    mappings, video_info = [], {}
    with new_output(output, root) as staging:
        for relative in profile["directories"]:
            within(staging, relative).mkdir(parents=True, exist_ok=True)
        for relative, config in profile["config_files"].items():
            write_json(within(staging, relative), config)
        for camera in profile["cameras"]:
            source_camera = by_camera[camera["source_camera"]]
            directories = [within(staging, relative) for relative in camera["frame_dirs"]]
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            for frame in source_camera["frames"]:
                model_frame = frame["index"] + profile["frame_base"]
                filename = camera["frame_pattern"].format(frame=model_frame)
                paths = [directory / filename for directory in directories]
                if materialize == "copy":
                    for destination in paths:
                        shutil.copyfile(within(root, frame["source_path"]), destination)
                mappings.append({"sequence": manifest["sequence"], "source_camera": source_camera["id"],
                                 "model_camera_id": camera["model_camera_id"], "model_camera_name": camera["model_camera_name"],
                                 "model_frame": model_frame, "source_frame": frame["source_frame"],
                                 "timestamp_seconds": frame["timestamp_seconds"], "source_path": frame["source_path"],
                                 "model_frame_path": str(paths[0].relative_to(staging))})
            if videos:
                if not camera.get("video_path"):
                    raise ValueError("Profile has no supported native video path")
                video_info[source_camera["id"]] = make_video(
                    directories[0], camera["frame_pattern"], profile["frame_base"], len(source_camera["frames"]),
                    fps, within(staging, camera["video_path"]), source_camera["width"], source_camera["height"],
                    video_decoder=video_decoder)
            write_json(staging / "metadata" / (source_camera["id"] + ".json"),
                       {"source_calibration": source_camera["calibration"], "geometry": source_camera["geometry"],
                        "timestamp_key": source_camera["timestamp_key"], "native_calibration_generated": False})
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "profile.json", profile)
        write_csv(staging / "frame_map.csv", FRAME_FIELDS, mappings)
        report = {"adapter_version": __version__, "model": model, "sequence": manifest["sequence"],
                  "input_layout_exported": True, "frames_per_camera": len(manifest["frame_ids"]),
                  "camera_count": len(manifest["cameras"]), "materialize": materialize, "videos": video_info,
                  "model_inference_run": False, "model_inference_ready": False,
                  "ground_truth_used_as_input": False, "calibration_generated": False,
                  "sync": manifest["sync"], "blockers": profile["blockers"],
                  "notes": ["Original timestamps are retained; constant-rate videos do not perform synchronization.",
                            "Image size and coordinates are preserved; no undistortion, resizing or world-plane inference."]}
        write_json(staging / "report.json", report)
    return report
