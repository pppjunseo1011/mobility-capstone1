"""Distortion-aware raw pixels -> nearest ground-mesh surface, in dataset units.

Numerical geometry only: neither person localization nor a native model H.
No GT, inferred metric scale, height filter, or plane fallback is used.
"""
from pathlib import Path

import cv2
import numpy as np

from .dataset import calibration_info


def read_ground_ply(path):
    """Read the exact SCOUT float-XYZ / triangle PLY contract, fail closed."""
    with Path(path).open("rb") as stream:
        lines = []
        for _ in range(100):
            raw = stream.readline(4096)
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("Truncated or oversized PLY header")
            line = raw.decode("ascii").strip()
            if not line.startswith("comment "):
                lines.append(line)
            if line == "end_header":
                break
        else:
            raise ValueError("PLY header too long")
        if len(lines) != 9 or lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("Unsupported PLY header")
        if lines[3:6] != ["property float x", "property float y", "property float z"] or lines[7:] != [
                "property list uchar int vertex_indices", "end_header"]:
            raise ValueError("Expected float XYZ and uchar/int triangle faces")
        if not lines[2].startswith("element vertex ") or not lines[6].startswith("element face "):
            raise ValueError("Expected ordered vertex and face elements")
        nv, nf = int(lines[2].split()[2]), int(lines[6].split()[2])
        if not 0 < nv <= 2000000 or not 0 < nf <= 4000000:
            raise ValueError("Mesh exceeds bounded reader limits or is empty")
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        expected = stream.tell() + nv * 12 + nf * face_dtype.itemsize
        if Path(path).stat().st_size != expected:
            raise ValueError("Truncated/non-triangular PLY or unexpected trailing data")
        vertices = np.fromfile(stream, dtype="<f4", count=nv * 3).reshape(nv, 3)
        records = np.fromfile(stream, dtype=face_dtype, count=nf)
    if not np.all(records["count"] == 3):
        raise ValueError("Only triangle faces are supported")
    return vertices.astype(np.float64), records["indices"].copy()


class GroundMesh:
    """Median-split triangle BVH; double-sided, nearest strictly forward hit."""

    def __init__(self, vertices, faces, leaf_size=64):
        vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError("Invalid mesh vertices")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or faces.dtype.kind not in "iu":
            raise ValueError("Invalid triangle indices")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Triangle references absent vertex")
        if not isinstance(leaf_size, int) or not 1 <= leaf_size <= 1024:
            raise ValueError("Invalid BVH leaf size")
        triangles = vertices[faces]
        self.a = triangles[:, 0].copy()
        self.e1 = triangles[:, 1] - self.a
        self.e2 = triangles[:, 2] - self.a
        area = np.linalg.norm(np.cross(self.e1, self.e2), axis=1)
        self.degenerate_faces = int(np.count_nonzero(area == 0))
        valid = np.flatnonzero(area > 0)
        if not len(valid):
            raise ValueError("Mesh has no nondegenerate triangles")
        self.face_count, self.vertex_count = len(faces), len(vertices)
        self.nodes = []
        lower, upper = triangles.min(axis=1), triangles.max(axis=1)
        centers = (lower + upper) / 2
        self.det_scale = np.linalg.norm(self.e1, axis=1) * np.linalg.norm(self.e2, axis=1)

        def build(ids):
            lo, hi = lower[ids].min(axis=0), upper[ids].max(axis=0)
            index = len(self.nodes)
            self.nodes.append(None)
            if len(ids) <= leaf_size:
                node = (lo, hi, None, None, ids)
            else:
                axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
                half = len(ids) // 2
                ids = ids[np.argpartition(centers[ids, axis], half)]
                left, right = build(ids[:half]), build(ids[half:])
                node = (lo, hi, left, right, None)
            self.nodes[index] = node
            return index

        build(valid)

    @staticmethod
    def _box_distance(origin, direction, lo, hi, maximum):
        near, far = 0.0, maximum
        for axis in range(3):
            if direction[axis] == 0:
                if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                    return None
            else:
                a, b = (lo[axis]-origin[axis])/direction[axis], (hi[axis]-origin[axis])/direction[axis]
                near, far = max(near, min(a, b)), min(far, max(a, b))
                if near > far:
                    return None
        return near

    def intersect(self, origin, direction):
        origin, direction = np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
        if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite([origin, direction]).all():
            raise ValueError("Invalid ray")
        length = np.linalg.norm(direction)
        if length == 0 or not np.isfinite(length):
            raise ValueError("Invalid ray direction")
        direction = direction / length
        closest, result, stack = float("inf"), None, [0]
        while stack:
            lo, hi, left, right, ids = self.nodes[stack.pop()]
            if self._box_distance(origin, direction, lo, hi, closest) is None:
                continue
            if ids is None:
                # Near-first traversal improves pruning; correctness does not rely on ordering.
                children = []
                for child in (left, right):
                    lower, upper = self.nodes[child][:2]
                    distance = self._box_distance(origin, direction, lower, upper, closest)
                    if distance is not None:
                        children.append((distance, child))
                stack.extend(child for _, child in sorted(children, reverse=True))
                continue
            e1, e2 = self.e1[ids], self.e2[ids]
            p = np.cross(direction, e2)
            det = np.einsum("ij,ij->i", e1, p)
            valid = np.abs(det) > 1e-12 * self.det_scale[ids]
            inverse = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
            delta = origin - self.a[ids]
            u = np.einsum("ij,ij->i", delta, p) * inverse
            q = np.cross(delta, e1)
            v = (q @ direction) * inverse
            distance = np.einsum("ij,ij->i", e2, q) * inverse
            valid &= (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1+1e-10) & (distance > 1e-8) & (distance < closest)
            if valid.any():
                candidate = np.where(valid, distance, np.inf)
                index = int(np.argmin(candidate))
                closest = float(candidate[index])
                result = {"world": origin + closest*direction, "distance": closest, "face_index": int(ids[index])}
        return result


class CameraGeometry:
    def __init__(self, calibration, width, height):
        calibration_info(calibration)
        self.k, self.r = (np.array(calibration[key], dtype=np.float64) for key in ("K", "R"))
        self.t = np.array(calibration["T"], dtype=np.float64).reshape(3)
        self.dist = np.array(calibration["dist"], dtype=np.float64).reshape(-1)
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid image dimensions")
        if not np.allclose(self.k[2], [0, 0, 1]) or self.k[0, 1] != 0 or self.k[1, 0] != 0:
            raise ValueError("OpenCV projection requires canonical zero-skew intrinsics")
        self.center = -self.r.T @ self.t
        self.rotvec = cv2.Rodrigues(self.r)[0]

    def project(self, world):
        world = np.asarray(world, dtype=np.float64).reshape(3)
        if not np.isfinite(world).all() or (self.r @ world + self.t)[2] <= 0:
            raise ValueError("World point is nonfinite or not in front of camera")
        return cv2.projectPoints(world.reshape(1, 1, 3), self.rotvec, self.t, self.k, self.dist)[0].reshape(2)

    def ray(self, pixel):
        pixel = np.asarray(pixel, dtype=np.float64)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("Invalid raw pixel")
        if not (0 <= pixel[0] < self.width and 0 <= pixel[1] < self.height):
            raise ValueError("Pixel outside raw image")
        # OpenCV's default five inverse iterations were insufficient near SCOUT corners.
        normalized = cv2.undistortPointsIter(pixel.reshape(1, 1, 2), self.k, self.dist, None, None,
                                            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)).reshape(2)
        local = np.array([*normalized, 1.0])
        recovered = cv2.projectPoints(local.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.k, self.dist)[0].reshape(2)
        error = float(np.linalg.norm(recovered-pixel))
        if not np.isfinite(error) or error > 1e-4:
            return None, error
        direction = self.r.T @ local
        return direction / np.linalg.norm(direction), error

    def ground_point(self, pixel, mesh):
        direction, error = self.ray(pixel)
        record = {"status": "undistort_not_converged", "undistort_error_px": error,
                  "world": None, "distance": None, "face_index": None, "reprojection_error_px": None}
        if direction is None:
            if not np.isfinite(error):
                record["undistort_error_px"] = None
            return record
        hit = mesh.intersect(self.center, direction)
        record["status"] = "no_ground_intersection"
        if hit is not None:
            reprojection = float(np.linalg.norm(self.project(hit["world"])-pixel))
            record["reprojection_error_px"] = reprojection
            if not np.isfinite(reprojection) or reprojection > 1e-4:
                record["status"] = "reprojection_failed"
                if not np.isfinite(reprojection):
                    record["reprojection_error_px"] = None
            else:
                record.update(hit, status="hit")
                record["world"] = hit["world"].tolist()
        return record
