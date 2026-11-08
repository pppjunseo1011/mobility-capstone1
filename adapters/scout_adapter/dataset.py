"""Read SCOUT 1.0 metadata and JPEG frame headers without ML dependencies."""
import math
from pathlib import Path
import re
import statistics
import struct

from .common import finite, integer, load_json, within

TOOLKIT_COMMIT = "1acd79513bfcf16aecef1150283af58c8273542e"


def dataset_root(path):
    root = Path(path).expanduser().resolve()
    candidates = [p for p in (root, root / "scout") if (p / "calibrations").is_dir() and (p / "timestamps").is_dir()]
    if len(candidates) != 1:
        raise ValueError("Expected one SCOUT root containing calibrations/ and timestamps/: " + str(root))
    return candidates[0]


def camera_number(name):
    match = re.fullmatch(r"cam_(0|[1-9][0-9]*)", name)
    if not match:
        raise ValueError("Invalid SCOUT camera name: " + name)
    return int(match.group(1))


def flatten(value):
    if isinstance(value, list):
        return [item for part in value for item in flatten(part)]
    return [value]


def calibration_info(calibration):
    def matrix(key):
        value = calibration.get(key)
        if not isinstance(value, list) or len(value) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in value):
            raise ValueError(key + " must be 3x3")
        return [[finite(cell, key) for cell in row] for row in value]

    k, r = matrix("K"), matrix("R")
    t = [finite(x, "T") for x in flatten(calibration.get("T"))]
    dist = [finite(x, "dist") for x in flatten(calibration.get("dist"))]
    if len(t) != 3 or len(dist) not in (4, 5, 8, 12, 14):
        raise ValueError("Expected T with 3 values and OpenCV distortion with 4/5/8/12/14 values")
    def det(m):
        return sum(m[0][i] * (m[1][(i+1)%3]*m[2][(i+2)%3] - m[1][(i+2)%3]*m[2][(i+1)%3]) for i in range(3))
    if abs(det(k)) < 1e-12 or abs(det(r)-1) > 1e-3:
        raise ValueError("Singular intrinsics or invalid rotation determinant")
    for i in range(3):
        for j in range(3):
            if abs(sum(r[n][i]*r[n][j] for n in range(3)) - (1 if i == j else 0)) > 1e-3:
                raise ValueError("Rotation is not orthonormal")
    extrinsic = [row + [t[i]] for i, row in enumerate(r)]
    projection = [[sum(k[i][n]*extrinsic[n][j] for n in range(3)) for j in range(4)] for i in range(3)]
    return {"projection_pinhole_world_to_image": projection,
            "distortion_nonzero": any(abs(d) > 1e-12 for d in dist),
            "homography": None, "world_unit": "unverified", "ground_plane": "unverified",
            "note": "P=K[R|T] is a pinhole projection; raw pixels require the supplied distortion model."}


def timestamp_key(camera, calibration, available, overrides=None):
    if overrides and camera in overrides:
        candidates = [str(overrides[camera])]
    else:
        view = str(calibration.get("view_id", ""))
        # Sequence 1 uses view IDs such as cvlabrpi10; sequence 2 uses cam0.
        candidates = [camera, camera.replace("cam_", "cam", 1), view]
        match = re.fullmatch(r"(.+)_\d+", view)
        if match:
            candidates.append(match.group(1))
    matches = sorted(set(key for key in candidates if key in available))
    if len(matches) != 1:
        raise ValueError("Cannot uniquely map {} to timestamp keys; use --timestamp-map. Matches: {}".format(camera, matches))
    return matches[0]


def jpeg_dimensions(path):
    """Read JPEG SOF only. Pixel decoding is checked separately in video tests."""
    with Path(path).open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("Not a JPEG: " + str(path))
        while stream.tell() < 1024**2:
            if stream.read(1) != b"\xff":
                raise ValueError("Invalid JPEG marker: " + str(path))
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if not marker or marker[0] in (0xD9, 0xDA):
                break
            if marker[0] in (0x01, *range(0xD0, 0xD8)):
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2:
                break
            if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                header = stream.read(5)
                if len(header) != 5:
                    break
                height, width = struct.unpack(">HH", header[1:])
                if width > 0 and height > 0:
                    return width, height
                break
            stream.seek(length-2, 1)
    raise ValueError("No valid JPEG dimensions: " + str(path))


def metadata(root, sequence, cameras=None, overrides=None, allow_unavailable=False):
    root = dataset_root(root)
    sequence = integer(sequence, "sequence", 1)
    timestamps = load_json(root / "timestamps" / ("timestamps_sequence_{}.json".format(sequence)))
    if not isinstance(timestamps, dict) or not timestamps:
        raise ValueError("Timestamps must be a non-empty frame->camera->seconds object")
    available = set()
    for frame, row in timestamps.items():
        if str(integer(frame, "timestamp frame")) != frame:
            raise ValueError("Timestamp frame keys must be canonical nonnegative integers: " + str(frame))
        if not isinstance(row, dict):
            raise ValueError("Invalid timestamp camera map")
        available.update(row)
    names = sorted((p.stem for p in (root / "calibrations").glob("cam_*.json")), key=camera_number)
    if cameras is not None:
        requested = list(cameras)
        if not requested or len(set(requested)) != len(requested) or set(requested)-set(names):
            raise ValueError("Unknown or duplicated cameras: " + repr(requested))
        names = sorted(requested, key=camera_number)
    rows, aliases = [], set()
    for camera in names:
        path = within(root, "calibrations/" + camera + ".json")
        calibration = load_json(path)
        record = {"id": camera, "number": camera_number(camera), "model_camera_id": camera_number(camera)+1,
                  "calibration": calibration, "geometry": calibration_info(calibration)}
        try:
            key = timestamp_key(camera, calibration, available, overrides)
        except ValueError as error:
            if not allow_unavailable:
                raise
            record.update({"timestamp_key": None, "timestamps_available": False,
                           "timestamp_mapping_error": str(error), "timestamp_count": 0,
                           "missing_timestamps": len(timestamps),
                           "non_increasing_timestamp_steps": 0, "duplicate_timestamp_steps": 0,
                           "backward_timestamp_steps": 0, "estimated_fps": None})
            rows.append(record)
            continue
        if key in aliases:
            raise ValueError("Two cameras map to the same timestamp key: " + key)
        aliases.add(key)
        samples = [(int(frame), finite(row[key], "timestamp")) for frame, row in timestamps.items() if key in row]
        samples.sort()
        observed_deltas = [b[1]-a[1] for a, b in zip(samples, samples[1:])]
        consecutive_deltas = [b[1]-a[1] for a, b in zip(samples, samples[1:]) if b[0] == a[0]+1]
        positive = [d for d in consecutive_deltas if d > 0]
        record.update({"timestamp_key": key, "timestamps_available": True,
                       "timestamp_count": len(samples), "missing_timestamps": len(timestamps)-len(samples),
                       "observed_timestamp_steps": len(observed_deltas),
                       "consecutive_frame_steps": len(consecutive_deltas),
                       "non_increasing_timestamp_steps": sum(d <= 0 for d in observed_deltas),
                       "duplicate_timestamp_steps": sum(d == 0 for d in observed_deltas),
                       "backward_timestamp_steps": sum(d < 0 for d in observed_deltas),
                       "estimated_fps": 1/statistics.median(positive) if positive else None})
        rows.append(record)
    return root, timestamps, rows


def inspect_dataset(root, sequence=1, cameras=None, overrides=None):
    root, timestamps, rows = metadata(root, sequence, cameras, overrides, allow_unavailable=True)
    for row in rows:
        directory = root / "images" / ("sequence_{}".format(sequence)) / row["id"]
        row["images_present"] = directory.is_dir()
        if directory.is_dir():
            ids = scan_frames(directory)
            row["image_count"] = len(ids)
            row["frame_range"] = [min(ids), max(ids)] if ids else None
        row.pop("calibration")
    return {"schema_version": 1, "source_root": str(root), "sequence": "sequence_{}".format(sequence),
            "timestamp_frame_count": len(timestamps), "cameras": rows,
            "toolkit_commit": TOOLKIT_COMMIT, "full_dataset_validated": False}


def scan_frames(directory):
    result = {}
    for path in directory.glob("image_*.jpg"):
        match = re.fullmatch(r"image_(0|[1-9][0-9]*)\.jpg", path.name)
        if not match:
            raise ValueError("Unexpected SCOUT frame name: " + str(path))
        result[int(match.group(1))] = path
    return result


def build_manifest(root, sequence=1, cameras=None, start=0, maximum=32, overrides=None, tolerance=0.05):
    root, timestamps, camera_rows = metadata(root, sequence, cameras, overrides)
    if not camera_rows:
        raise ValueError("No cameras selected")
    start, maximum = integer(start, "start frame"), integer(maximum, "max frames", 1)
    tolerance = finite(tolerance, "timestamp tolerance")
    if tolerance < 0:
        raise ValueError("Timestamp tolerance must be >= 0")
    frame_sets = {row["id"]: scan_frames(root / "images" / ("sequence_{}".format(sequence)) / row["id"]) for row in camera_rows}
    first = camera_rows[0]["id"]
    selected = sorted(frame for frame in frame_sets[first] if frame >= start)[:maximum]
    if not selected:
        raise ValueError("No extracted JPEG frames available; a .tar.gz.part is not a dataset root")
    for row in camera_rows:
        missing = set(selected)-set(frame_sets[row["id"]])
        if missing:
            raise ValueError("{} is missing selected frames {}; no silent dropping/reindexing".format(row["id"], sorted(missing)[:10]))
        frames, dimensions = [], set()
        for index, frame in enumerate(selected):
            source = within(root, frame_sets[row["id"]][frame].relative_to(root))
            dimensions.add(jpeg_dimensions(source))
            timestamp = timestamps.get(str(frame), {}).get(row["timestamp_key"])
            if timestamp is None:
                raise ValueError("Missing timestamp: {}/{}".format(row["id"], frame))
            timestamp = finite(timestamp, "timestamp")
            if frames and timestamp < frames[-1]["timestamp_seconds"]:
                raise ValueError("Selected camera timestamps go backward: " + row["id"])
            duplicate = bool(frames and timestamp == frames[-1]["timestamp_seconds"])
            frames.append({"index": index, "source_frame": frame, "source_path": str(source.relative_to(root)),
                           "timestamp_seconds": timestamp, "timestamp_duplicate": duplicate})
        if len(dimensions) != 1:
            raise ValueError("Camera resolution changes inside selected frames: " + row["id"])
        row["width"], row["height"] = next(iter(dimensions))
        row["frames"] = frames
        row["selected_duplicate_timestamp_count"] = sum(frame["timestamp_duplicate"] for frame in frames)
    skews = [max(row["frames"][i]["timestamp_seconds"] for row in camera_rows) -
             min(row["frames"][i]["timestamp_seconds"] for row in camera_rows) for i in range(len(selected))]
    return {"schema_version": 1, "source_root": str(root), "sequence": "sequence_{}".format(sequence),
            "sequence_number": sequence, "frame_ids": selected, "image_space": "raw",
            "cameras": camera_rows, "toolkit_commit": TOOLKIT_COMMIT,
            "sync": {"policy": "same-source-frame-with-original-timestamps; no resampling",
                     "max_skew_seconds": max(skews), "tolerance_seconds": tolerance,
                     "within_tolerance": max(skews) <= tolerance},
            "labels_used_as_detections": False, "full_dataset_validated": False}
