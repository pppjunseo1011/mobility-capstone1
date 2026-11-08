"""YACHIYO layout and strict result readers; no calibration or model execution."""

import csv
import json
import math
import re
from pathlib import Path


NAME = "yachiyo"
FRAME_BASE = 1
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+\Z")


def _integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(field, minimum))
    return value


def _text_integer(value, field, minimum=0):
    if not isinstance(value, str) or not _INTEGER_TEXT.fullmatch(value):
        raise ValueError("{} must contain an integer".format(field))
    return _integer(int(value), field, minimum)


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a finite number".format(field))
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("{} must be a finite number".format(field)) from None
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number".format(field))
    return result


def build_profile(manifest: dict) -> dict:
    """Return export-root-relative layout, config and execution blockers."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    scene_number = _integer(manifest.get("sequence_number"), "sequence_number")
    scene_name = "scene_{:03d}".format(scene_number)
    source_cameras = manifest.get("cameras")
    if not isinstance(source_cameras, list) or not source_cameras:
        raise ValueError("manifest.cameras must be a nonempty list")

    cameras = []
    camera_ids = set()
    source_ids = set()
    dimensions = set()
    missing_calibration = []
    directories = [
        "Original/" + scene_name,
        "Detection/" + scene_name,
        "EmbedFeature/" + scene_name,
        "Pose/" + scene_name,
        "Tracking/" + scene_name,
        "Submission/" + scene_name,
        "tracking/config",
    ]
    for source in source_cameras:
        if not isinstance(source, dict):
            raise ValueError("each manifest camera must be an object")
        number = _integer(source.get("number"), "camera.number")
        source_id = source.get("id")
        if source_id != "cam_{}".format(number):
            raise ValueError("camera.id must agree with camera.number: cam_<number>")
        camera_id = _integer(source.get("model_camera_id"), "model_camera_id", 1)
        if camera_id != number + 1:
            raise ValueError("model_camera_id must equal source camera number + 1")
        if source_id in source_ids or camera_id in camera_ids:
            raise ValueError("duplicate source or model camera ID")
        source_ids.add(source_id)
        camera_ids.add(camera_id)
        width = _integer(source.get("width"), source_id + ".width", 1)
        height = _integer(source.get("height"), source_id + ".height", 1)
        dimensions.add((width, height))
        calibration = source.get("calibration")
        if calibration is None:
            missing_calibration.append(source_id)
        elif not isinstance(calibration, dict):
            raise ValueError(source_id + ".calibration must be an object or null")

        model_name = "camera_{:04d}".format(camera_id)
        camera_root = "Original/{}/{}".format(scene_name, model_name)
        cameras.append({
            "source_camera": source_id,
            "model_camera_id": camera_id,
            "model_camera_name": model_name,
            "frame_dirs": [camera_root + "/Frame"],
            "frame_pattern": "{frame:06d}.jpg",
            # The upstream pose script literally reads video.mp4. This is a
            # required destination, not a claim that any video was produced.
            "video_path": camera_root + "/video.mp4",
            "calibration_path": camera_root + "/calibration.json",
        })
        directories.extend([
            camera_root,
            camera_root + "/Frame",
            "EmbedFeature/{}/{}".format(scene_name, model_name),
            "Pose/{}/{}".format(scene_name, model_name),
        ])

    cameras.sort(key=lambda camera: camera["model_camera_id"])
    blockers = [
        "The native pose loop is zero-based but detection/ReID frames are one-based; "
        "align video index, detection frame and pose JSON keys before inference "
        "(poser/top_down_video_demo_with_track_file.py:149-176).",
        "Native calibration.json is intentionally not generated. Validate raw-image "
        "lens distortion, pixel coordinates, world units and ground-plane geometry "
        "before supplying world-to-image H and camera projection P. "
        "The source calibration dictionary alone does not establish this contract.",
        "The native tracker assumes fps=30 and frame-count thresholds; determine "
        "the selected frames' sampling rate and synchronization before configuring "
        "tracking, particularly for sparse or truncated exports.",
        "Local YOLOX/OSNet/HRNet checkpoints and generated detection, ReID feature "
        "and pose artifacts are still required. This export does not generate or "
        "substitute any of them with ground truth.",
        "The pose stage requires a video.mp4 matching the exported frame order. "
        "A video_path in this profile does not create or verify that video.",
        "Native scripts use a common working directory, write beside inputs, and "
        "activate Conda environments. A stage-specific runner must connect the "
        "export layout, config, read-only inputs and writable intermediate outputs.",
    ]
    if dimensions != {(1920, 1080)}:
        blockers.append(
            "Export dimensions {} differ from the detector's hard-coded 1920x1080 "
            "scale/clipping and the tracker's image_size. Patch those assumptions "
            "using actual image dimensions; this profile does not resize images."
            .format(sorted(dimensions))
        )
    if len(dimensions) > 1:
        blockers.append(
            "Cameras have different image sizes, but the native tracker accepts a "
            "single image_size setting; per-camera handling needs an explicit change."
        )
    if missing_calibration:
        blockers.append("No source calibration for: " + ", ".join(missing_calibration))

    sync = manifest.get("sync")
    if sync is None:
        sync = {}
    if not isinstance(sync, dict):
        raise ValueError("manifest.sync must be an object")
    max_skew = sync.get("max_skew_seconds")
    tolerance = _number(sync.get("tolerance_seconds", 0.05), "sync.tolerance_seconds")
    if tolerance < 0:
        raise ValueError("sync.tolerance_seconds must be >= 0")
    if max_skew is None:
        blockers.append("Cross-camera timestamp synchronization has not been verified.")
    else:
        max_skew = _number(max_skew, "sync.max_skew_seconds")
        if max_skew < 0:
            raise ValueError("sync.max_skew_seconds must be >= 0")
        if max_skew > tolerance:
            blockers.append(
                "Cross-camera timestamp skew {:.9g}s exceeds tolerance {:.9g}s."
                .format(max_skew, tolerance)
            )

    return {
        "name": NAME,
        "scene_name": scene_name,
        "frame_base": FRAME_BASE,
        "inference_ready": False,
        "cameras": cameras,
        "directories": list(dict.fromkeys(directories)),
        "config_files": {
            "tracking/config/scene_2_camera_id_file.json": [{
                "scene_name": scene_name,
                "camera_ids": [camera["model_camera_id"] for camera in cameras],
            }],
        },
        "blockers": blockers,
        "notes": [
            "Export frames are numbered consecutively from 1 as 000001.jpg. "
            "The parent adapter retains the source-frame/timestamp inverse mapping.",
            "The detector enumerates sorted directory entries from 1 rather than "
            "reading a frame ID from the filename; keep Frame directories free of "
            "unrelated files and do not preserve sparse source indices as model IDs.",
            "Detector CSV fields are camera,frame,class=1,x1,y1,x2,y2,score. "
            "The third field is not a person identity. The detector currently "
            "appends CSV and JSON, so fresh output files are required on each run.",
            "ReID expects matching CSV/JSON row order and reads "
            "Frame/{frame:06d}.jpg. Empty/single-row CSV inputs need explicit "
            "handling in the native ReID runner.",
            "Tracking reads feature_{frame}_{serial}_{x1}_{x2}_{y1}_{y2}_{conf}.npy "
            "under EmbedFeature; generated empty directories are not model inputs.",
            "Native H maps the world ground plane to image pixels; image-to-world "
            "uses inv(H) and homogeneous normalization. P is a required JSON key "
            "in the loader even though the inspected tracking path does not use it.",
            "Scene/camera config is written under the export root, not into the "
            "upstream checkout. A runner must use that config explicitly.",
            "JSON GlobalOfflineID may be 0 and is shared across cameras within "
            "one scene. Scene-level namespacing and source-camera/frame restoration "
            "belong to the parent result normalizer.",
            "Native final results have no confidence score or measured world z. "
            "Readers return score=None, world_xyz=None and preserve world_xy when present.",
        ],
        "native_results": {
            "preferred_format": "yachiyo-json",
            "formats": {
                "yachiyo-json": {
                    "path": "Tracking/{}/fixed_whole_tracking_results.json".format(scene_name),
                    "frame_base": 1,
                    "structure": "camera ID -> detection serial -> Frame, Coordinate, GlobalOfflineID, WorldCoordinate",
                    "bbox_format": "xyxy",
                },
                "aic9": {
                    "path": "Submission/{}/track1.txt".format(scene_name),
                    "frame_base": 0,
                    "columns": ["camera", "id", "frame", "x", "y", "w", "h", "world_x", "world_y"],
                    "delimiter": "whitespace (comma-delimited rows also accepted by reader)",
                },
            },
            "reader_frame_base": 1,
            "id_scope": "scene-global, including valid ID 0",
            "confidence_available": False,
            "world_z_available": False,
        },
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError("non-finite JSON numeric constant: " + value)


def _bbox_xywh(x, y, width, height, location):
    values = [_number(value, location + ".bbox") for value in (x, y, width, height)]
    if values[0] < 0 or values[1] < 0 or values[2] <= 0 or values[3] <= 0:
        raise ValueError(location + ": bbox requires x/y >= 0 and width/height > 0")
    return values


def _row(camera, track, frame, bbox, world_xy):
    return {
        "camera_id": camera,
        "track_id": track,
        "frame": frame,
        "bbox_xywh": bbox,
        "score": None,
        "world_xyz": None,
        "world_xy": world_xy,
    }


def _json_rows(path):
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream, object_pairs_hook=_unique_object,
                         parse_constant=_invalid_json_constant)
    if not isinstance(data, dict):
        raise ValueError("YACHIYO result must be a JSON object keyed by camera")
    seen_cameras = set()
    for raw_camera, detections in data.items():
        camera = _text_integer(raw_camera, "camera ID", 1)
        if camera in seen_cameras:
            raise ValueError("duplicate numeric camera ID: {}".format(camera))
        seen_cameras.add(camera)
        if not isinstance(detections, dict):
            raise ValueError("camera {} results must be an object".format(camera))
        for serial, detection in detections.items():
            location = "camera {} serial {}".format(camera, serial)
            if not isinstance(detection, dict):
                raise ValueError(location + ": detection must be an object")
            track = _integer(detection.get("GlobalOfflineID"), location + ".GlobalOfflineID")
            frame = _integer(detection.get("Frame"), location + ".Frame", 1)
            coordinate = detection.get("Coordinate")
            if not isinstance(coordinate, dict):
                raise ValueError(location + ": Coordinate must be an object")
            try:
                x1, y1, x2, y2 = [_number(coordinate[key], location + "." + key)
                                  for key in ("x1", "y1", "x2", "y2")]
            except KeyError as error:
                raise ValueError(location + ": missing Coordinate key " + str(error)) from None
            bbox = _bbox_xywh(x1, y1, x2 - x1, y2 - y1, location)
            world = detection.get("WorldCoordinate")
            world_xy = None
            if world is not None:
                if not isinstance(world, dict) or "x" not in world or "y" not in world:
                    raise ValueError(location + ": WorldCoordinate requires x and y")
                world_xy = [_number(world[key], location + ".world_" + key) for key in ("x", "y")]
            yield _row(camera, track, frame, bbox, world_xy)


def _aic_rows(path):
    with path.open("r", encoding="utf-8") as stream:
        for number, text in enumerate(stream, 1):
            if not text.strip():
                continue
            location = "{}:{}".format(path, number)
            try:
                fields = next(csv.reader([text], strict=True)) if "," in text else text.split()
                fields = [field.strip() for field in fields]
                if len(fields) != 9:
                    raise ValueError("expected exactly 9 fields, got {}".format(len(fields)))
                camera = _text_integer(fields[0], "camera", 1)
                track = _text_integer(fields[1], "track ID")
                frame = _text_integer(fields[2], "frame") + FRAME_BASE
                values = [_number(float(value), "numeric field") for value in fields[3:]]
                bbox = _bbox_xywh(*values[:4], location)
            except (ValueError, OverflowError, csv.Error) as error:
                raise ValueError(location + ": " + str(error)) from None
            yield _row(camera, track, frame, bbox, values[4:6])


def read_results(path: Path, *, format: str):
    """Yield strict native results with one-based model frame numbers.

    Missing/unassigned global IDs and malformed rows raise ValueError. No score,
    identity, calibration or world z is inferred. Readers preserve native XY.
    """
    path = Path(path)
    if format == "yachiyo-json":
        rows = _json_rows(path)
    elif format == "aic9":
        rows = _aic_rows(path)
    else:
        raise ValueError("unsupported YACHIYO result format: " + str(format))
    seen = set()
    for row in rows:
        key = (row["camera_id"], row["track_id"], row["frame"])
        if key in seen:
            raise ValueError("duplicate camera/track/frame result: {}".format(key))
        seen.add(key)
        yield row
