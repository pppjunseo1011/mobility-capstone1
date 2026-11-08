"""Native artifact -> raw-pixel points -> ground geometry, without inference.

Policies are explicit. PoseTrack ankle-only projection is diagnostic and does
not reproduce its multiview triangulation, age logic or elevated feet plane.
"""
import csv
import json
from pathlib import Path

from .common import finite, integer, load_json, new_output, separated, write_csv, write_json
from .ground import INPUT_FIELDS, project_ground, sha256

CONTRACTS = {
    "yachiyo-sct": ("yachiyo", 1, "yachiyo-bbox-bottom"),
    "uwipl-mot16": ("uwipl", 0, "uwipl-feet"),
    "posetrack-pose57": ("posetrack", 0, "ankles-only"),
}
MAP_FIELDS = ["point_id", "camera", "source_frame", "model_frame", "timestamp_seconds", "native_row",
              "native_serial", "native_track_id", "id_scope", "pixel_x", "pixel_y", "point_method", "status"]


def strict_json(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key: " + key)
            obj[key] = value
        return obj

    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def bounded_file(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > 32*1024*1024:
        raise ValueError("Expected an artifact file no larger than 32 MiB")
    return path


def table(path, columns, comma):
    rows = []
    with bounded_file(path).open(newline="", encoding="utf-8") as stream:
        for text in stream:
            fields = next(csv.reader([text], strict=True)) if comma else text.split()
            if len(fields) != columns or len(rows) >= 4096:
                raise ValueError("Wrong column count/blank row or artifact exceeds 4096 rows")
            # Check all values without converting ID/frame strings through float.
            for field in fields:
                finite(field, "artifact field")
            rows.append(fields)
    return rows


def bbox_xyxy(values):
    x1, y1, x2, y2 = [finite(v, "bbox") for v in values]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox size")
    return x1, y1, x2, y2


def native_points(source, format, detections=None):
    """Yield native rows without reordering or inventing identities."""
    if format == "yachiyo-sct":
        payload = strict_json(bounded_file(source))
        if not isinstance(payload, dict) or len(payload) > 4096:
            raise ValueError("Expected per-camera SCT dictionary with <=4096 rows")
        seen = set()
        for serial, value in payload.items():
            serial_int = integer(serial, "serial")
            if serial_int in seen or not isinstance(value, dict):
                raise ValueError("Duplicate numeric serial or invalid SCT record")
            seen.add(serial_int)
            box = value["Coordinate"]
            coords = [integer(box[k], "native integer bbox", -2147483648) for k in ("x1", "y1", "x2", "y2")]
            x1, y1, x2, y2 = bbox_xyxy(coords)
            track = integer(value["OfflineID"], "OfflineID", -1)
            yield {"frame": integer(value["Frame"], "Frame", 1), "serial": serial,
                   "track": track, "pixel": ((x1+x2)/2, y2), "method": "bbox_bottom_center"}
    elif format == "uwipl-mot16":
        for row in table(source, 16, True):
            frame, track = integer(row[0], "frame"), integer(row[1], "track ID")
            x, y, w, h = [finite(v, "bbox") for v in row[2:6]]
            if w <= 0 or h <= 0:
                raise ValueError("Invalid bbox size")
            score = finite(row[6], "score")
            if score != -1 and not 0 <= score <= 1:
                raise ValueError("Invalid tracking confidence")
            ax, ay, ac, bx, by, bc = [finite(v, "ankle") for v in row[10:16]]
            pixel, method = ((ax+bx)/2, (ay+by)/2), "ankle_mean_gt_0.5"
            if not (ac > .5 and bc > .5):
                pixel, method = (x+.5*w, y+.95*h), "bbox_95pct_fallback"
            yield {"frame": frame, "serial": "", "track": track, "pixel": pixel, "method": method}
    elif format == "posetrack-pose57":
        if detections is None:
            raise ValueError("PoseTrack requires paired --detections to check row alignment")
        poses, dets = table(source, 57, False), table(detections, 7, True)
        if len(poses) != len(dets):
            raise ValueError("Pose/detection row counts differ")
        import numpy as np
        seen = set()
        for pose, det in zip(poses, dets):
            frame = integer(pose[0], "pose frame")
            if frame != integer(det[0], "detection frame") or integer(det[1], "class") != 1:
                raise ValueError("Pose/detection frame or person class mismatch")
            bbox_xyxy(pose[1:5])
            bbox_xyxy(det[2:6])
            # Native pose exporter casts bbox and score to float32 before savetxt.
            left = np.array([finite(v, "pose bbox/score") for v in pose[1:6]], dtype=np.float32)
            right = np.array([finite(v, "det bbox/score") for v in det[2:7]], dtype=np.float32)
            if not np.isfinite(left).all() or not np.array_equal(left, right):
                raise ValueError("Pose/detection bbox or score row alignment mismatch")
            if not 0 <= finite(det[6], "score") <= 1:
                raise ValueError("Invalid detection confidence")
            key = (frame, *right[:4].tolist())
            if key in seen:
                raise ValueError("Ambiguous duplicate detection frame/bbox")
            seen.add(key)
            ax, ay, ac, bx, by, bc = [finite(v, "ankle") for v in pose[51:57]]
            pixel = ((ax+bx)/2, (ay+by)/2) if ac > .7 and bc > .7 else None
            yield {"frame": frame, "serial": "", "track": None, "pixel": pixel,
                   "method": "ankle_mean_gt_0.7_diagnostic"}
    else:
        raise ValueError("Unsupported model artifact format")


def prepare_model_points(export, source, output, *, camera, format, policy, id_scope,
                         input_kind="unverified", detections=None):
    export, source = Path(export).resolve(), bounded_file(source)
    for path in (export, source, detections):
        if path is not None:
            separated(path, output)
    if format not in CONTRACTS or policy != CONTRACTS[format][2]:
        raise ValueError("Select the documented explicit point policy for this format")
    model, base, _ = CONTRACTS[format]
    allowed_scopes = {"yachiyo": ("camera",), "uwipl": ("camera", "sequence"), "posetrack": ("none",)}
    if id_scope not in allowed_scopes[model] or input_kind not in ("unverified", "model_output", "synthetic_fixture"):
        raise ValueError("Invalid ID scope or input kind")
    if model != "posetrack" and detections is not None:
        raise ValueError("Paired detections are only supported for PoseTrack")
    manifest_path, profile_path = export/"manifest.json", export/"profile.json"
    manifest, profile = load_json(manifest_path), load_json(profile_path)
    if manifest.get("schema_version") != 1 or manifest.get("image_space") != "raw":
        raise ValueError("Only schema 1 raw-image exports are supported")
    if manifest.get("model") != model or profile.get("name") != model or profile.get("frame_base") != base or manifest.get("model_frame_base") != base:
        raise ValueError("Wrong model or frame-base contract")
    sequence_number = integer(manifest["sequence_number"], "sequence", 1)
    if manifest["sequence"] != "sequence_{}".format(sequence_number):
        raise ValueError("Inconsistent sequence name")
    if manifest.get("source_root"):
        separated(manifest["source_root"], output)
    names = [c["id"] for c in manifest["cameras"]]
    if len(set(names)) != len(names) or camera not in names:
        raise ValueError("Unknown or duplicate camera")
    item = next(c for c in manifest["cameras"] if c["id"] == camera)
    if camera != "cam_{}".format(integer(item["number"], "camera number")) or item["model_camera_id"] != item["number"]+1:
        raise ValueError("Inconsistent camera ID")
    width, height = integer(item["width"], "width", 1), integer(item["height"], "height", 1)
    frames = item["frames"]
    previous = -1
    for index, f in enumerate(frames):
        source_frame = integer(f["source_frame"], "source frame")
        if f["index"] != index or source_frame <= previous or f["source_path"] != "images/{}/{}/image_{}.jpg".format(manifest["sequence"], camera, source_frame):
            raise ValueError("Inconsistent frame mapping")
        finite(f["timestamp_seconds"], "timestamp")
        previous = source_frame
    if not frames:
        raise ValueError("Empty camera export")
    provenance = {"source_sha256": sha256(source), "manifest_sha256": sha256(manifest_path),
                  "profile_sha256": sha256(profile_path), "detections_sha256": sha256(bounded_file(detections)) if detections else None}
    records, points, identities, counts = [], [], set(), {}
    for row_index, row in enumerate(native_points(source, format, detections)):
        offset = row["frame"]-base
        if not 0 <= offset < len(frames):
            raise ValueError("Native frame outside exported clip")
        f = frames[offset]
        track = row["track"]
        if track is not None and track >= 0:
            key = (row["frame"], track)
            if key in identities:
                raise ValueError("Duplicate identity within camera/frame")
            identities.add(key)
        pixel = row["pixel"]
        status = "ready" if pixel is not None else "low_ankle_confidence"
        if pixel is not None and not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            status = "outside_image"
        point_id = "row_{:06d}".format(row_index)
        record = dict(zip(MAP_FIELDS, [point_id, camera, f["source_frame"], row["frame"], f["timestamp_seconds"],
                                       row_index, row["serial"], track, id_scope,
                                       pixel[0] if pixel is not None else None, pixel[1] if pixel is not None else None,
                                       row["method"], status]))
        records.append(record)
        counts[status] = counts.get(status, 0)+1
        if status == "ready":
            points.append({key: record[key] for key in INPUT_FIELDS})
    report = {"schema_version": 1, "kind": "native_ground_input", "model": model, "format": format,
              "camera": camera, "model_camera_id": item["model_camera_id"], "sequence": manifest["sequence"],
              "policy": policy, "id_scope": id_scope, "input_kind": input_kind, "input_provenance_verified": False,
              "native_row_count": len(records), "projectable_point_count": len(points), "status_counts": counts,
              "provenance": provenance, "point_map_sha256": None,
              "pose_detection_row_alignment_checked": model == "posetrack", "reid_alignment_checked": False,
              "raw_pixel_space_is_caller_contract": True, "model_inference_run": False, "ground_truth_used": False,
              "native_tracking_algorithm_reproduced": False, "point_policy_native_equivalent": model != "posetrack",
              "limitations": ["File schema and pairing do not prove these are actual model predictions or raw-image coordinates.",
                              "BBox/ankle projection is a candidate ground location, not verified person localization.",
                              "PoseTrack ankle-only policy does not reproduce its multiview, age or elevated-feet logic.",
                              "An artifact without camera metadata relies on the explicit --camera declaration."]}
    with new_output(output) as staging:
        write_csv(staging/"points.csv", INPUT_FIELDS, points)
        write_csv(staging/"point_map.csv", MAP_FIELDS, records)
        report["point_map_sha256"] = sha256(staging/"point_map.csv")
        report["points_sha256"] = sha256(staging/"points.csv")
        write_json(staging/"bridge.json", report)
    return report


def project_model_ground(root, export, source, output, *, emit_yachiyo_candidate=False, **options):
    for path in (root, export, source, options.get("detections")):
        if path is not None:
            separated(path, output)
    if emit_yachiyo_candidate and options.get("format") != "yachiyo-sct":
        raise ValueError("Native candidate export is only implemented for YACHIYO SCT")
    with new_output(output) as staging:
        prepared = staging/"prepared"
        bridge = prepare_model_points(export, source, prepared, **options)
        geometry = project_ground(root, export, staging/"projection", prepared/"points.csv", bridge["input_kind"])
        with (staging/"projection/world_points.csv").open() as stream:
            projected = {row["point_id"]: row for row in csv.DictReader(stream)}
        with (prepared/"point_map.csv").open() as stream:
            records = list(csv.DictReader(stream))
        fields = ["world_x", "world_y", "world_z", "distance_world_units", "face_index", "reprojection_error_px"]
        counts = {}
        for row in records:
            result = projected.get(row["point_id"])
            if row["status"] == "ready":
                if result is None or result["camera"] != row["camera"] or result["source_frame"] != row["source_frame"]:
                    raise ValueError("Projection row correspondence failed")
                row["status"] = result["status"]
            row.update({key: result[key] if result else None for key in fields})
            counts[row["status"]] = counts.get(row["status"], 0)+1
        native_candidate = None
        if emit_yachiyo_candidate:
            if any(row["status"] != "hit" for row in records):
                raise ValueError("Cannot emit YACHIYO candidate: at least one row lacks a valid ground point")
            if sha256(source) != bridge["provenance"]["source_sha256"]:
                raise ValueError("Native source changed during conversion")
            payload = strict_json(source)
            for row in records:
                payload[row["native_serial"]]["WorldCoordinate"] = {"x": float(row["world_x"]), "y": float(row["world_y"])}
            native_candidate = "native_candidate/camera{:03d}_tracking_results.json".format(bridge["model_camera_id"])
            write_json(staging/native_candidate, payload)
        write_csv(staging/"ground_candidates.csv", MAP_FIELDS+fields, records)
        report = {"schema_version": 1, "kind": "native_ground_candidates", "bridge": bridge,
                  "geometry_status_counts": geometry["status_counts"], "row_status_counts": counts,
                  "row_count": len(records), "native_candidate": native_candidate,
                  "native_candidate_sha256": sha256(staging/native_candidate) if native_candidate else None,
                  "model_inference_run": False, "ground_truth_used": False, "native_H_generated": False,
                  "native_tracker_connected": False, "model_inference_ready": False,
                  "physical_geometry_validated": False, "world_unit": geometry["world_unit"],
                  "note": "Candidate sidecars only. IDs/frames are retained; no tracking or native H replacement has run. "
                          "Optional YACHIYO JSON changes only WorldCoordinate XY in a separate candidate file."}
        write_json(staging/"model-ground.json", report)
    return report
