"""Bounded point projection and synthetic-grid diagnosis; never tracking results."""
import csv
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from .common import finite, integer, load_json, new_output, separated, within, write_csv, write_json
from .dataset import dataset_root, jpeg_dimensions
from .geometry import CameraGeometry, GroundMesh, read_ground_ply

INPUT_FIELDS = ["camera", "source_frame", "point_id", "pixel_x", "pixel_y"]
OUTPUT_FIELDS = INPUT_FIELDS + ["timestamp_seconds", "status", "world_x", "world_y", "world_z",
                                "distance_world_units", "face_index", "undistort_error_px", "reprojection_error_px"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_ground(root, export, output, source=None, input_kind="unverified", columns=17, rows=11):
    began = time.monotonic()
    root, export = dataset_root(root), Path(export).resolve()
    separated(root, output)
    separated(export, output)
    if source:
        separated(source, output)
    if input_kind not in ("unverified", "synthetic_fixture", "model_output"):
        raise ValueError("Invalid input kind")
    manifest_path = export / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("image_space") != "raw":
        raise ValueError("Only raw SCOUT pixels with schema 1 are supported")
    sequence = integer(manifest["sequence_number"], "sequence", 1)
    if manifest["sequence"] != "sequence_{}".format(sequence):
        raise ValueError("Inconsistent sequence name")
    if not 1 <= len(manifest["cameras"]) <= 25:
        raise ValueError("Invalid camera count")
    times_path = root / "timestamps" / "timestamps_sequence_{}.json".format(sequence)
    times = load_json(times_path)
    cameras, frames, calibration_hashes = {}, {}, {}
    for item in manifest["cameras"]:
        name = item["id"]
        calibration_path = within(root, "calibrations/" + name + ".json")
        calibration = load_json(calibration_path)
        if name in cameras or calibration != item["calibration"]:
            raise ValueError("Duplicate camera or calibration differs from export: " + name)
        camera = CameraGeometry(calibration, item["width"], item["height"])
        frames[name] = {}
        for frame in item["frames"]:
            number = integer(frame["source_frame"], "source frame")
            expected = "images/sequence_{}/{}/image_{}.jpg".format(sequence, name, number)
            if frame["source_path"] != expected or not within(root, expected).is_file():
                raise ValueError("Missing or mismatched source frame")
            timestamp = finite(times.get(str(number), {}).get(item["timestamp_key"]), "timestamp")
            if number in frames[name] or timestamp != frame["timestamp_seconds"]:
                raise ValueError("Duplicate frame or timestamp differs from export")
            frames[name][number] = timestamp
        if not frames[name]:
            raise ValueError("Empty selected camera")
        sample = within(root, item["frames"][0]["source_path"])
        if jpeg_dimensions(sample) != (camera.width, camera.height):
            raise ValueError("Image dimensions differ from export")
        cameras[name] = camera
        calibration_hashes[name] = sha256(calibration_path)

    if source:
        with Path(source).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != INPUT_FIELDS:
                raise ValueError("Expected CSV columns: " + ",".join(INPUT_FIELDS))
            points = []
            for record in reader:
                if len(points) >= 4096 or None in record or any(v is None for v in record.values()):
                    raise ValueError("Malformed point CSV or point limit 4096 exceeded")
                points.append(record)
        kind = input_kind
    else:
        columns, rows = integer(columns, "grid columns", 2), integer(rows, "grid rows", 2)
        if columns*rows*len(cameras) > 4096:
            raise ValueError("Grid exceeds point limit 4096")
        points = []
        for name, camera in cameras.items():
            for y_index, y in enumerate(np.linspace(0, camera.height-1, rows)):
                for x_index, x in enumerate(np.linspace(0, camera.width-1, columns)):
                    points.append(dict(zip(INPUT_FIELDS, [name, min(frames[name]),
                                                         "grid_{}_{}".format(x_index, y_index), float(x), float(y)])))
        kind = "synthetic_grid"
    seen = set()
    for record in points:
        name = record["camera"]
        frame = integer(record["source_frame"], "source frame")
        if name not in cameras or frame not in frames[name] or not record["point_id"]:
            raise ValueError("Unknown camera/frame or empty point ID")
        key = (name, frame, record["point_id"])
        if key in seen:
            raise ValueError("Duplicate point ID within camera/frame")
        seen.add(key)
        record.update(source_frame=frame, pixel_x=finite(record["pixel_x"], "pixel x"),
                      pixel_y=finite(record["pixel_y"], "pixel y"))
        camera = cameras[name]
        if not 0 <= record["pixel_x"] < camera.width or not 0 <= record["pixel_y"] < camera.height:
            raise ValueError("Point outside raw image")
    mesh_path = within(root, "meshes/mesh_ground_only.ply")
    provenance = {"manifest_sha256": sha256(manifest_path), "timestamps_sha256": sha256(times_path),
                  "calibration_sha256": calibration_hashes, "mesh_sha256": sha256(mesh_path),
                  "point_csv_sha256": sha256(source) if source else None}
    with new_output(output) as staging:
        vertices, faces = read_ground_ply(mesh_path)
        mesh = GroundMesh(vertices, faces)
        print("Ground mesh loaded: {} vertices, {} faces, {} BVH nodes".format(len(vertices), len(faces), len(mesh.nodes)), flush=True)
        converted, counts, per_camera, maximum_error = [], {}, {}, None
        for record in points:
            camera = cameras[record["camera"]]
            result = camera.ground_point([record["pixel_x"], record["pixel_y"]], mesh)
            status = result["status"]
            counts[status] = counts.get(status, 0)+1
            camera_counts = per_camera.setdefault(record["camera"], {})
            camera_counts[status] = camera_counts.get(status, 0)+1
            world = result["world"] or [None, None, None]
            converted.append({**record, "timestamp_seconds": frames[record["camera"]][record["source_frame"]],
                              "status": status, "world_x": world[0], "world_y": world[1], "world_z": world[2],
                              "distance_world_units": result["distance"], "face_index": result["face_index"],
                              "undistort_error_px": result["undistort_error_px"],
                              "reprojection_error_px": result["reprojection_error_px"]})
            if status == "hit":
                maximum_error = max(maximum_error or 0, result["reprojection_error_px"])
        report = {"schema_version": 1, "kind": "ground_projection", "sequence": manifest["sequence"],
                  "input_kind": kind, "input_provenance_verified": False, "point_count": len(points),
                  "status_counts": counts, "per_camera": per_camera, "provenance": provenance,
                  "vertices": mesh.vertex_count, "faces": mesh.face_count, "degenerate_faces_ignored": mesh.degenerate_faces,
                  "bvh_nodes": len(mesh.nodes), "maximum_hit_reprojection_error_px": maximum_error,
                  "world_unit": "dataset world units; physical unit not independently validated",
                  "intersection_policy": "nearest positive double-sided ground-mesh intersection; no height/depth heuristic or plane fallback",
                  "inverse_distortion": {"max_iterations": 100, "epsilon": 1e-12, "accepted_roundtrip_px": 1e-4},
                  "model_inference_run": False, "ground_truth_used": False, "native_H_generated": False,
                  "physical_calibration_accuracy_validated": False, "model_geometry_connected": False,
                  "occlusion_checked": False, "image_space": "raw",
                  "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
                  "limitations": ["A ground hit does not prove the input pixel is a person's foot or visible ground.",
                                  "Only ground mesh is intersected; buildings, people and other occluders are not tested.",
                                  "Reprojection checks numerical consistency, not measured world-position accuracy.",
                                  "No-hit/nonconverged rows have blank world coordinates; never replace them with zeros.",
                                  "This is not the toolkit's depth>=1 / height<0.8 heuristic or a replacement native H.",
                                  "Model-output provenance is a caller declaration, not evidence of inference."],
                  "elapsed_seconds": round(time.monotonic()-began, 3)}
        write_csv(staging / "points.csv", INPUT_FIELDS, points)
        write_csv(staging / "world_points.csv", OUTPUT_FIELDS, converted)
        write_json(staging / "geometry.json", report)
    return report
