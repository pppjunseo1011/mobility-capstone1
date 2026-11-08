"""Bridge YACHIYO detector frame 1 -> pose frame 0 -> tracker frame 1.

Only the native eight-column detection CSV and default COCO-17 pose JSON are
supported. No predictions, keypoints, timestamps or identities are invented.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

from ..common import new_output, separated, within, write_csv, write_json


KINDS = ("unverified", "model_output", "synthetic_fixture")
_UINT = re.compile(r"0|[1-9][0-9]*")


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


def _text_integer(value, label, minimum=0):
    if not isinstance(value, str) or not _UINT.fullmatch(value):
        raise ValueError(label + " must be a canonical nonnegative integer")
    return _integer(int(value), label, minimum)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be a finite JSON number")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError(label + " must be finite") from None
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _json_bytes(raw):
    def invalid(value):
        raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=invalid)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _input_kind(value):
    if value not in KINDS:
        raise ValueError("input_kind must be one of " + ", ".join(KINDS))
    return value


def _contract(export, camera):
    export = Path(export).resolve()
    raw_manifest = (export / "manifest.json").read_bytes()
    raw_profile = (export / "profile.json").read_bytes()
    manifest, profile = _json_bytes(raw_manifest), _json_bytes(raw_profile)
    if not isinstance(manifest, dict) or not isinstance(profile, dict):
        raise ValueError("Export manifest/profile must be objects")
    if (manifest.get("schema_version") != 1 or manifest.get("model") != "yachiyo"
            or profile.get("name") != "yachiyo" or manifest.get("image_space") != "raw"
            or type(manifest.get("model_frame_base")) is not int
            or manifest["model_frame_base"] != 1 or type(profile.get("frame_base")) is not int
            or profile["frame_base"] != 1):
        raise ValueError("Expected a raw YACHIYO export with one-based model frames")
    sequence = _integer(manifest.get("sequence_number"), "sequence_number", 1)
    scene = "scene_{:03d}".format(sequence)
    if manifest.get("sequence") != "sequence_{}".format(sequence) or profile.get("scene_name") != scene:
        raise ValueError("Sequence/scene identifiers disagree")
    sources, profiles = manifest.get("cameras"), profile.get("cameras")
    if not isinstance(sources, list) or not isinstance(profiles, list):
        raise ValueError("Export cameras must be lists")
    source_ids = [row.get("id") for row in sources if isinstance(row, dict)]
    profile_ids = [row.get("source_camera") for row in profiles if isinstance(row, dict)]
    if (len(source_ids) != len(sources) or len(profile_ids) != len(profiles)
            or len(set(source_ids)) != len(source_ids) or len(set(profile_ids)) != len(profile_ids)
            or set(source_ids) != set(profile_ids) or camera not in source_ids):
        raise ValueError("Missing, duplicated or inconsistent export cameras")
    source = next(row for row in sources if row["id"] == camera)
    target = next(row for row in profiles if row["source_camera"] == camera)
    number = _integer(source.get("number"), "camera.number")
    model_id = _integer(source.get("model_camera_id"), "model_camera_id", 1)
    model_name = "camera_{:04d}".format(model_id)
    if (camera != "cam_{}".format(number) or model_id != number + 1
            or target.get("model_camera_id") != model_id or target.get("model_camera_name") != model_name):
        raise ValueError("Source/native camera identifiers disagree")
    width = _integer(source.get("width"), "image width", 1)
    height = _integer(source.get("height"), "image height", 1)
    frame_ids, frames = manifest.get("frame_ids"), source.get("frames")
    if not isinstance(frame_ids, list) or not frame_ids or not isinstance(frames, list) or len(frames) != len(frame_ids):
        raise ValueError("Expected a nonempty, consistent exported frame list")
    previous = -1
    for index, (frame_id, frame) in enumerate(zip(frame_ids, frames)):
        _integer(frame_id, "source frame")
        if frame_id <= previous or not isinstance(frame, dict):
            raise ValueError("Source frame IDs must be unique and increasing")
        if type(frame.get("index")) is not int or frame["index"] != index or frame.get("source_frame") != frame_id:
            raise ValueError("Manifest source/model frame mapping is inconsistent")
        expected_path = "images/{}/{}/image_{}.jpg".format(manifest["sequence"], camera, frame_id)
        if frame.get("source_path") != expected_path:
            raise ValueError("Source frame/path mapping is inconsistent")
        _number(frame.get("timestamp_seconds"), "timestamp_seconds")
        previous = frame_id
    return {"export": export, "manifest": manifest, "profile": profile, "source": source,
            "camera": camera, "model_camera_name": model_name, "model_camera_id": model_id,
            "width": width, "height": height, "frames": frames, "frame_count": len(frames),
            "manifest_sha256": _sha(raw_manifest), "profile_sha256": _sha(raw_profile)}


def _separate_inputs(output, contract, *inputs):
    for source in (contract["export"], *inputs):
        separated(source, output)
    if contract["manifest"].get("source_root"):
        separated(contract["manifest"]["source_root"], output)


def _bbox(values, contract, label):
    if len(values) != 4:
        raise ValueError(label + " must have four xyxy coordinates")
    x1, y1, x2, y2 = [_number(value, label) for value in values]
    if not (0 <= x1 < x2 <= contract["width"] and 0 <= y1 < y2 <= contract["height"]):
        raise ValueError(label + " must have positive area within the exported image")
    return (x1, y1, x2, y2)


def _csv_rows(raw, contract, base):
    """Read actual detector rows, not MOT rows or annotations with identity IDs."""
    rows, seen, previous = [], set(), base
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        for line, cells in enumerate(reader, 1):
            label = "CSV row {}".format(line)
            if len(cells) != 8 or any(cell != cell.strip() for cell in cells):
                raise ValueError(label + " requires exactly eight native fields without surrounding whitespace")
            if cells[0] != contract["model_camera_name"] or cells[2] != "1":
                raise ValueError(label + " must use the selected camera_NNNN and class=1 (not a track ID)")
            frame = _text_integer(cells[1], label + " frame", base)
            if not base <= frame < base + contract["frame_count"] or frame < previous:
                raise ValueError(label + " frame is outside the export or out of native order")
            coordinates = [_text_integer(value, label + " coordinate") for value in cells[3:7]]
            bbox = _bbox(coordinates, contract, label + " bbox")
            try:
                score = float(cells[7])
            except (ValueError, OverflowError):
                raise ValueError(label + " score must be finite and in [0,1]") from None
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(label + " score must be finite and in [0,1]")
            key = (frame, bbox)
            if key in seen:
                raise ValueError(label + " duplicates a frame/bbox; tracker pose association would be ambiguous")
            seen.add(key)
            rows.append({"cells": cells, "frame": frame, "bbox": bbox})
            previous = frame
    except csv.Error as error:
        raise ValueError("Malformed native CSV: " + str(error)) from None
    return rows


def _base_report(contract, input_kind, kind):
    return {"schema_version": 1, "kind": kind, "model": "yachiyo", "input_kind": _input_kind(input_kind),
            "sequence": contract["manifest"]["sequence"], "source_camera": contract["camera"],
            "model_camera_name": contract["model_camera_name"], "model_camera_id": contract["model_camera_id"],
            "export_manifest_sha256": contract["manifest_sha256"],
            "export_profile_sha256": contract["profile_sha256"], "exported_frame_count": contract["frame_count"],
            "image_space": "raw", "image_size": [contract["width"], contract["height"]],
            "model_inference_run": False, "model_inference_ready": False,
            "input_provenance_verified": False, "video_alignment_verified": False,
            "ground_truth_used_as_input": False, "synthetic_predictions_created": False,
            "note": "Only frame numbering is transformed. input_kind is a caller declaration, not proof of model execution. "
                    "This bridge does not verify video pixels, synchronize cameras, supply weights/calibration, or run inference."}


def prepare_pose_csv(export, source, output, *, camera, input_kind="unverified"):
    """Create a new directory containing pose_input.csv, bridge.json and frame_map.csv.

    Input: native detector/ReID CSV, frame 1. Output: a separate CSV for the
    original pose CLI only, frame 0. Preserve original detector/ReID files.
    """
    contract = _contract(export, camera)
    source = Path(source).resolve()
    _separate_inputs(output, contract, source)
    raw = source.read_bytes()
    rows = _csv_rows(raw, contract, 1)
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    for row in rows:
        cells = list(row["cells"])
        cells[1] = str(row["frame"] - 1)
        writer.writerow(cells)
    encoded = text.getvalue().encode("utf-8")
    report = _base_report(contract, input_kind, "yachiyo_pose_csv_frame_bridge")
    report.update(source_file=str(source), source_sha256=_sha(raw),
                  source_frame_base=1, output_frame_base=0, frame_offset=-1,
                  detection_count=len(rows), frames_with_detections=len({row["frame"] for row in rows}),
                  output_file="pose_input.csv", output_sha256=_sha(encoded),
                  native_pose_bbox_score=1.0,
                  native_pose_score_note="Original pose loader replaces detector confidence with bbox score 1.0.")
    mappings = [{"source_camera": camera, "source_frame": frame["source_frame"],
                 "model_frame": index + 1, "pose_frame": index,
                 "timestamp_seconds": frame["timestamp_seconds"]}
                for index, frame in enumerate(contract["frames"])]
    with new_output(output) as staging:
        (staging / "pose_input.csv").write_bytes(encoded)
        write_csv(staging / "frame_map.csv", ["source_camera", "source_frame", "model_frame", "pose_frame", "timestamp_seconds"], mappings)
        write_json(staging / "bridge.json", report)
    return report


def restore_pose_json(export, source, output, *, camera, pose_input, input_kind="unverified"):
    """Restore native pose JSON frame 0 to tracker frame 1 in a new directory.

    pose_input must be the directory returned by prepare_pose_csv. Require the
    default native pose contract: exactly one COCO-17 result per supplied bbox,
    bbox score 1.0, and no invented empty frames. Missing results are rejected.
    """
    contract = _contract(export, camera)
    source, pose_input = Path(source).resolve(), Path(pose_input).resolve()
    _separate_inputs(output, contract, source, pose_input)
    metadata_raw = (pose_input / "bridge.json").read_bytes()
    prepared = _json_bytes(metadata_raw)
    if not isinstance(prepared, dict) or prepared.get("kind") != "yachiyo_pose_csv_frame_bridge" or prepared.get("schema_version") != 1:
        raise ValueError("pose_input must be a prepared YACHIYO CSV bridge directory")
    for key, expected in (("export_manifest_sha256", contract["manifest_sha256"]),
                          ("export_profile_sha256", contract["profile_sha256"]),
                          ("source_camera", camera), ("model_camera_name", contract["model_camera_name"]),
                          ("source_frame_base", 1), ("output_frame_base", 0), ("output_file", "pose_input.csv")):
        if prepared.get(key) != expected:
            raise ValueError("pose_input metadata mismatch: " + key)
    prepared_kind = _input_kind(prepared.get("input_kind"))
    _input_kind(input_kind)
    if prepared_kind == "synthetic_fixture" and input_kind != "synthetic_fixture":
        raise ValueError("Synthetic pose inputs must remain explicitly synthetic on restore")
    csv_raw = within(pose_input, prepared["output_file"]).read_bytes()
    if _sha(csv_raw) != prepared.get("output_sha256"):
        raise ValueError("Prepared pose CSV hash mismatch")
    detections = _csv_rows(csv_raw, contract, 0)
    if len(detections) != prepared.get("detection_count"):
        raise ValueError("Prepared pose CSV detection count mismatch")
    expected = {}
    for row in detections:
        expected.setdefault(row["frame"], set()).add(row["bbox"])
    raw = source.read_bytes()
    payload = _json_bytes(raw)
    if not isinstance(payload, dict):
        raise ValueError("Pose JSON must be a frame-keyed object")
    restored, seen_frames, count = {}, set(), 0
    for key, poses in payload.items():
        frame = _text_integer(key, "Pose JSON frame")
        if frame >= contract["frame_count"] or frame not in expected or frame in seen_frames:
            raise ValueError("Unexpected/duplicate pose frame or frame outside export: " + key)
        if not isinstance(poses, list):
            raise ValueError("Each pose frame must contain a list")
        seen_frames.add(frame)
        boxes = set()
        for pose in poses:
            if not isinstance(pose, dict) or set(pose) != {"bbox", "keypoints"}:
                raise ValueError("Expected native COCO pose objects with only bbox and keypoints")
            bbox = pose["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 5 or _number(bbox[4], "pose bbox score") != 1.0:
                raise ValueError("Native pose bbox must be xyxy plus loader-generated score 1.0")
            coordinate = _bbox(bbox[:4], contract, "pose bbox")
            if coordinate in boxes or coordinate not in expected[frame]:
                raise ValueError("Duplicate pose bbox or bbox does not match prepared CSV at frame " + key)
            boxes.add(coordinate)
            keypoints = pose["keypoints"]
            if not isinstance(keypoints, list) or len(keypoints) != 17:
                raise ValueError("Expected COCO-17 keypoints")
            for point in keypoints:
                if not isinstance(point, list) or len(point) != 3:
                    raise ValueError("Each keypoint must contain x, y and score")
                for value in point:
                    _number(value, "keypoint coordinate/score")
            count += 1
        if boxes != expected[frame]:
            raise ValueError("Missing pose results for prepared detections at frame " + key)
        restored[str(frame + 1)] = poses
    if seen_frames != set(expected):
        raise ValueError("Pose JSON is missing frames with prepared detections")
    restored = dict(sorted(restored.items(), key=lambda item: int(item[0])))
    report = _base_report(contract, input_kind, "yachiyo_pose_json_frame_bridge")
    filename = contract["model_camera_name"] + "_out_keypoint.json"
    report.update(source_file=str(source), source_sha256=_sha(raw),
                  prepared_input_kind=prepared_kind, prepared_bridge_sha256=_sha(metadata_raw),
                  prepared_csv_sha256=_sha(csv_raw), source_frame_base=0, output_frame_base=1, frame_offset=1,
                  detection_count=count, frames_with_detections=len(restored), output_file=filename,
                  bbox_and_keypoints_preserved=True, all_prepared_detections_matched=True)
    with new_output(output) as staging:
        write_json(staging / filename, restored)
        report["output_sha256"] = _sha((staging / filename).read_bytes())
        write_json(staging / "bridge.json", report)
    return report
