#!/usr/bin/env python3
"""CPU-only SCOUT calibration/mesh diagnostics; no native H or model inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or not values.size:
        raise ValueError("Non-finite or empty diagnostic array")
    return {"count": int(values.size), "min": float(values.min()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max()), "rms": float(np.sqrt(np.mean(values**2)))}


def ground_vertices(path):
    lines, count, properties, element = [], None, [], None
    with path.open("rb") as stream:
        for unused in range(100):
            line = stream.readline(4096).decode("ascii").strip()
            lines.append(line)
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                count, element = int(fields[2]), "vertex"
            elif fields[:1] == ["element"]:
                element = fields[1]
            elif fields[:1] == ["property"] and element == "vertex":
                properties.append(fields[1:])
            if line == "end_header":
                break
        else:
            raise ValueError("PLY header missing or too long")
        if lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
            raise ValueError("Expected binary little-endian PLY")
        if properties != [["float", "x"], ["float", "y"], ["float", "z"]] or not count:
            raise ValueError("Expected nonempty float xyz vertex element")
        if count > 5000000:
            raise ValueError("Vertex limit exceeded")
        vertices = np.fromfile(stream, dtype=[(axis, "<f4") for axis in "xyz"], count=count)
    if len(vertices) != count:
        raise ValueError("Truncated PLY vertex block")
    xyz = np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite mesh vertices")
    return xyz, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.data.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output must not overlap")
    destination = output / "geometry-audit.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite previous diagnostic")
    began = time.monotonic()
    mesh_path = source / "meshes/mesh_ground_only.ply"
    xyz, header = ground_vertices(mesh_path)
    center = xyz.mean(axis=0)
    centered = xyz-center
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / len(xyz))
    normal = vectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    residual = centered @ normal
    report = {"scope": "sequence_1 cam_0..cam_7 calibration; whole ground mesh vertices",
              "native_calibration_generated": False, "native_H_generated": False,
              "model_inference_run": False, "ground_truth_used_for_fit": False,
              "semantic_geometry_validated": False, "unit": "dataset world units; not independently validated",
              "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
              "mesh": {"path": str(mesh_path), "header": header, "vertices_read": len(xyz),
                       "faces_read": False, "bounds_xyz": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
                       "plane_fit": "PCA least squares over all vertices; vertex-weighted, not surface-area weighted",
                       "plane_center": center.tolist(), "unit_normal": normal.tolist(),
                       "plane_equation_n_dot_x_plus_d": [*normal.tolist(), float(-normal @ center)],
                       "covariance_eigenvalues": eigenvalues.tolist(),
                       "absolute_plane_residual": statistics(np.abs(residual)),
                       "z_distribution": statistics(xyz[:, 2])}, "cameras": {}}
    x, y = np.meshgrid(np.linspace(0, 1919, 21), np.linspace(0, 1079, 13))
    pixels = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    for number in range(8):
        name = "cam_{}".format(number)
        path = source / "calibrations" / (name + ".json")
        raw = path.read_bytes()
        calibration = json.loads(raw)
        k, r = (np.array(calibration[field], dtype=np.float64).reshape(3, 3) for field in ("K", "R"))
        t, dist = (np.array(calibration[field], dtype=np.float64).reshape(-1) for field in ("T", "dist"))
        if t.size != 3 or dist.size not in (4, 5, 8, 12, 14) or not all(np.isfinite(v).all() for v in (k,r,t,dist)):
            raise ValueError("Invalid calibration: " + name)
        if abs(np.linalg.det(k)) < 1e-12:
            raise ValueError("Singular intrinsic matrix")
        orthogonality = float(np.max(np.abs(r.T @ r-np.eye(3))))
        determinant = float(np.linalg.det(r))
        ideal = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(k).T
        raw_pixels = cv2.projectPoints(ideal, np.zeros(3), np.zeros(3), k, dist)[0].reshape(-1, 2)
        recovered = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
        rays = np.column_stack([recovered, np.ones(len(recovered))])
        reprojection = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, dist)[0].reshape(-1, 2)
        world = (ideal*10-t) @ r
        projection = k @ np.column_stack([r, t])
        pinhole = np.column_stack([world, np.ones(len(world))]) @ projection.T
        pinhole = pinhole[:, :2]/pinhole[:, 2:3]
        report["cameras"][name] = {
            "calibration_sha256": hashlib.sha256(raw).hexdigest(), "R_determinant": determinant,
            "R_orthogonality_max_abs_error": orthogonality,
            "rotation_numerically_valid": abs(determinant-1)<1e-6 and orthogonality<1e-6,
            "camera_center_world": (-r.T @ t).tolist(), "distortion_coefficients": dist.tolist(),
            "P_pinhole_reference_only": projection.tolist(),
            "raw_vs_ideal_pixel_displacement": statistics(np.linalg.norm(raw_pixels-pixels, axis=1)),
            "raw_undistort_redistort_error_default_opencv": statistics(np.linalg.norm(reprojection-pixels, axis=1)),
            "P_synthetic_world_roundtrip_pixel_error": statistics(np.linalg.norm(pinhole-pixels, axis=1))}
    report.update(grid={"width":1920,"height":1080,"points":len(pixels),
                        "note":"Full-image ideal grid for distortion displacement; raw grid for default-iteration roundtrip. Algebraic tests, not measured calibration accuracy."},
                  blockers=["World units/axis-to-scene correspondence and a valid ground region still require verification.",
                            "Global mesh-plane fit is diagnostic and cannot establish a single valid ground plane for all views.",
                            "Raw pixels require distortion-aware geometry; model-native P/H wiring and coordinate inversion remain unimplemented."],
                  elapsed_seconds=round(time.monotonic()-began,3))
    output.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output":str(destination),"vertices":len(xyz),"plane_residual":report["mesh"]["absolute_plane_residual"]}))


if __name__ == "__main__":
    main()
