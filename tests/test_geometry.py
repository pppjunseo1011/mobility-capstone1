"""Analytic geometry and file-contract tests; no people, GT, or weights."""
import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from scout_adapter.geometry import CameraGeometry, GroundMesh, read_ground_ply
from scout_adapter.ground import INPUT_FIELDS, project_ground


def calibration():
    return {"K": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
            "R": np.eye(3).tolist(), "T": [0., 0., 5.], "dist": [0.]*5, "view_id": "view0"}


def square(z=0, size=10):
    return np.array([[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]], dtype=float), np.array([[0, 1, 2], [0, 2, 3]])


def write_ply(path, vertices, faces):
    header = ("ply\nformat binary_little_endian 1.0\ncomment test fixture\nelement vertex {}\n"
              "property float x\nproperty float y\nproperty float z\nelement face {}\n"
              "property list uchar int vertex_indices\nend_header\n").format(len(vertices), len(faces))
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        np.asarray(vertices, dtype="<f4").tofile(stream)
        for face in faces:
            stream.write(struct.pack("<Biii", 3, *face))


class GeometryTests(unittest.TestCase):
    def test_square_center_shared_edge_nearest(self):
        mesh = GroundMesh(*square(), leaf_size=1)
        hit = mesh.intersect([0, 0, 5], [0, 0, -2])
        np.testing.assert_allclose(hit["world"], [0, 0, 0])
        self.assertEqual(hit["distance"], 5)

    def test_two_sided_and_behind(self):
        mesh = GroundMesh(*square())
        self.assertIsNotNone(mesh.intersect([0, 0, -5], [0, 0, 1]))
        self.assertIsNone(mesh.intersect([0, 0, -5], [0, 0, -1]))

    def test_parallel_outside_and_origin_on_surface(self):
        mesh = GroundMesh(*square())
        for origin, direction in [([0, 0, 5], [1, 0, 0]), ([20, 0, 5], [0, 0, -1]), ([0, 0, 0], [0, 0, 1])]:
            self.assertIsNone(mesh.intersect(origin, direction))

    def test_nearest_of_overlapping_surfaces(self):
        a, f = square(0)
        b, g = square(2)
        mesh = GroundMesh(np.vstack([a, b]), np.vstack([f, g+4]), leaf_size=1)
        np.testing.assert_allclose(mesh.intersect([1, 2, 8], [0, 0, -1])["world"], [1, 2, 2])

    def test_bvh_many_triangles_against_analytic_plane(self):
        vertices, faces = [], []
        for x in range(12):
            for y in range(8):
                offset = len(vertices)
                vertices.extend([[x, y, 2], [x+1, y, 2], [x+1, y+1, 2], [x, y+1, 2]])
                faces.extend([[offset, offset+1, offset+2], [offset, offset+2, offset+3]])
        mesh = GroundMesh(vertices, faces, leaf_size=4)
        rng = np.random.default_rng(19)
        for target in rng.uniform([0.01, .01, 2], [11.99, 7.99, 2], size=(100, 3)):
            origin = np.array([4., 3., 9.])
            np.testing.assert_allclose(mesh.intersect(origin, target-origin)["world"], target, atol=1e-10)

    def test_degenerate_triangles_reported(self):
        v, f = square()
        mesh = GroundMesh(v, np.vstack([f, [0, 0, 0]]))
        self.assertEqual(mesh.degenerate_faces, 1)
        with self.assertRaises(ValueError):
            GroundMesh(v, [[0, 0, 0]])

    def test_invalid_mesh_and_rays(self):
        v, f = square()
        for faces in ([[0, 1, 999]], [[-1, 1, 2]], [[0., 1., 2.]]):
            with self.assertRaises(ValueError):
                GroundMesh(v, faces)
        mesh = GroundMesh(v, f)
        for direction in ([0, 0, 0], [0, 0, float("nan")], [1, 2]):
            with self.assertRaises(ValueError):
                mesh.intersect([0, 0, 1], direction)

    def test_raw_pixel_known_world(self):
        camera = CameraGeometry(calibration(), 100, 100)
        hit = camera.ground_point([70, 30], GroundMesh(*square()))
        self.assertEqual(hit["status"], "hit")
        np.testing.assert_allclose(hit["world"], [1, -1, 0], atol=1e-10)
        self.assertLess(hit["reprojection_error_px"], 1e-8)

    def test_rotated_camera_and_distortion(self):
        c = calibration()
        c["R"] = cv2.Rodrigues(np.array([0.15, -0.2, 0.1]))[0].tolist()
        c["dist"] = [0.15, -0.02, 0.003, -0.004, 0.001]
        camera = CameraGeometry(c, 100, 100)
        for point in ([1, -.5, 0], [-1, .5, 0], [.1, .1, 0]):
            pixel = camera.project(point)
            hit = camera.ground_point(pixel, GroundMesh(*square()))
            np.testing.assert_allclose(hit["world"], point, atol=1e-8)

    def test_missing_ground_not_zero(self):
        camera = CameraGeometry(calibration(), 100, 100)
        hit = camera.ground_point([0, 0], GroundMesh(*square(size=.1)))
        self.assertEqual(hit["status"], "no_ground_intersection")
        self.assertIsNone(hit["world"])

    def test_bad_pixels_and_behind_camera(self):
        camera = CameraGeometry(calibration(), 100, 100)
        for pixel in ([-1, 0], [100, 20], [20, 100], [float("inf"), 0]):
            with self.assertRaises(ValueError):
                camera.ray(pixel)
        with self.assertRaises(ValueError):
            camera.project([0, 0, -10])

    def test_nonconvergence_keeps_no_world(self):
        from unittest.mock import patch
        camera = CameraGeometry(calibration(), 100, 100)
        with patch("scout_adapter.geometry.cv2.undistortPointsIter", return_value=np.array([[[100., 100.]]])):
            result = camera.ground_point([50, 50], GroundMesh(*square()))
        self.assertEqual(result["status"], "undistort_not_converged")
        self.assertIsNone(result["world"])

    def test_ply_roundtrip_and_bad_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"mesh.ply"
            v, f = square()
            write_ply(path, v, f)
            a, b = read_ground_ply(path)
            np.testing.assert_equal(a, v)
            np.testing.assert_equal(b, f)
            raw = path.read_bytes()
            for bad in (raw[:-1], raw+b"x", raw.replace(b"float x", b"float q"), raw.replace(b"little_endian", b"big_endian")):
                path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    read_ground_ply(path)


class GroundCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.export, self.output = self.base/"scout", self.base/"export", self.base/"output"
        for directory in ("calibrations", "timestamps", "meshes", "images/sequence_1/cam_0"):
            (self.root/directory).mkdir(parents=True)
        self.export.mkdir()
        self.calibration = calibration()
        self.write_json(self.root/"calibrations/cam_0.json", self.calibration)
        self.write_json(self.root/"timestamps/timestamps_sequence_1.json", {"277": {"view0": 12.3}})
        Image.new("RGB", (100, 100)).save(self.root/"images/sequence_1/cam_0/image_277.jpg")
        write_ply(self.root/"meshes/mesh_ground_only.ply", *square())
        self.manifest = {"schema_version": 1, "image_space": "raw", "sequence": "sequence_1", "sequence_number": 1,
                         "cameras": [{"id": "cam_0", "calibration": self.calibration, "width": 100, "height": 100,
                                      "timestamp_key": "view0", "frames": [{"source_frame": 277,
                                      "source_path": "images/sequence_1/cam_0/image_277.jpg", "timestamp_seconds": 12.3}]}]}
        self.save_manifest()

    def write_json(self, path, value):
        path.write_text(json.dumps(value))

    def save_manifest(self):
        self.write_json(self.export/"manifest.json", self.manifest)

    def points(self, values):
        path = self.base/"input.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(INPUT_FIELDS)
            writer.writerows(values)
        return path

    def run_projection(self, **kwargs):
        return project_ground(self.root, self.export, self.output, **kwargs)

    def test_probe_labels_and_preserves_sources(self):
        before = {p: p.read_bytes() for directory in (self.root, self.export) for p in directory.rglob("*") if p.is_file()}
        report = self.run_projection(columns=3, rows=2)
        self.assertEqual(report["point_count"], 6)
        self.assertEqual(report["status_counts"], {"hit": 6})
        self.assertEqual(report["input_kind"], "synthetic_grid")
        self.assertFalse(report["physical_calibration_accuracy_validated"])
        for p, content in before.items():
            self.assertEqual(p.read_bytes(), content)

    def test_explicit_points_and_source_frame_mapping(self):
        report = self.run_projection(source=self.points([["cam_0", 277, "p0", 70, 30]]), input_kind="synthetic_fixture")
        self.assertEqual(report["input_kind"], "synthetic_fixture")
        with (self.output/"world_points.csv").open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["source_frame"], "277")
        self.assertEqual(row["timestamp_seconds"], "12.3")
        self.assertAlmostEqual(float(row["world_x"]), 1)

    def test_unknown_frame_camera_and_duplicate_points(self):
        for records in ([["cam_0", 0, "p", 50, 50]], [["cam_1", 277, "p", 50, 50]],
                        [["cam_0", 277, "p", 50, 50]]*2):
            with self.assertRaises(ValueError):
                self.run_projection(source=self.points(records))
            self.assertFalse(self.output.exists())

    def test_missing_ground_csv_blanks(self):
        write_ply(self.root/"meshes/mesh_ground_only.ply", *square(size=.1))
        self.run_projection(source=self.points([["cam_0", 277, "p", 0, 0]]))
        with (self.output/"world_points.csv").open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "no_ground_intersection")
        self.assertEqual(row["world_x"], "")

    def test_empty_points_are_valid_empty_results(self):
        report = self.run_projection(source=self.points([]))
        self.assertEqual(report["point_count"], 0)
        self.assertIsNone(report["maximum_hit_reprojection_error_px"])

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        (self.output/"keep.txt").write_text("keep")
        with self.assertRaises(ValueError):
            self.run_projection()
        self.assertEqual((self.output/"keep.txt").read_text(), "keep")

    def test_mismatched_calibration_rejected(self):
        c = calibration()
        c["T"][0] += 1
        self.write_json(self.root/"calibrations/cam_0.json", c)
        with self.assertRaises(ValueError):
            self.run_projection()

    def test_changed_timestamp_rejected(self):
        self.write_json(self.root/"timestamps/timestamps_sequence_1.json", {"277": {"view0": 99}})
        with self.assertRaises(ValueError):
            self.run_projection()

    def test_grid_limit_and_changed_image_space(self):
        with self.assertRaises(ValueError):
            self.run_projection(columns=100, rows=100)
        self.manifest["image_space"] = "undistorted"
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.run_projection()

    def test_input_output_overlap_rejected(self):
        with self.assertRaises(ValueError):
            project_ground(self.root, self.export, self.root/"output")

    def test_mesh_failure_cleans_staging(self):
        mesh_path = self.root/"meshes/mesh_ground_only.ply"
        mesh_path.write_bytes(mesh_path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            self.run_projection()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob(".output-*")), [])
