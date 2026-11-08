"""Synthetic native artifacts only: IDs and poses are never real predictions."""
import csv
import json
import unittest
import numpy as np

import test_geometry as fixtures
from scout_adapter.model_ground import prepare_model_points, project_model_ground, native_points


class ModelGroundTests(unittest.TestCase):
    write_json = fixtures.GroundCommandTests.write_json
    save_manifest = fixtures.GroundCommandTests.save_manifest

    def setUp(self):
        fixtures.GroundCommandTests.setUp(self)
        self.source = self.base/"synthetic_native.txt"
        self.det = self.base/"synthetic_detections.txt"
        self.options = {"camera": "cam_0", "input_kind": "synthetic_fixture"}

    def model(self, name):
        bases = {"yachiyo": 1, "uwipl": 0, "posetrack": 0}
        self.manifest.update(model=name, model_frame_base=bases[name])
        self.manifest["cameras"][0].update(number=0, model_camera_id=1)
        self.manifest["cameras"][0]["frames"][0]["index"] = 0
        self.save_manifest()
        self.write_json(self.export/"profile.json", {"name": name, "frame_base": bases[name]})
        formats = {"yachiyo": ("yachiyo-sct", "yachiyo-bbox-bottom", "camera"),
                   "uwipl": ("uwipl-mot16", "uwipl-feet", "sequence"),
                   "posetrack": ("posetrack-pose57", "ankles-only", "none")}
        self.options.update(dict(zip(("format", "policy", "id_scope"), formats[name])))

    def yachiyo(self, **changes):
        self.model("yachiyo")
        record = {"Frame": 1, "OfflineID": 0, "Coordinate": {"x1": 40, "y1": 10, "x2": 60, "y2": 30},
                  "WorldCoordinate": None, "NpyPath": "synthetic-not-opened.npy", "Extra": "preserve"}
        record.update(changes)
        self.write_json(self.source, {"00000000": record})
        return record

    def uwipl(self, confidence=.9):
        self.model("uwipl")
        row = [0, 0, 40, 10, 20, 20, .9, -1, -1, -1, 45, 25, confidence, 55, 29, .9]
        self.source.write_text(",".join(map(str, row))+"\n")
        return row

    def posetrack(self, confidence=.9):
        self.model("posetrack")
        det = [0, 1, 40, 10, 60, 30, .91234567]
        kp = [[50, 20, .9] for _ in range(17)]
        kp[15], kp[16] = [45, 25, confidence], [55, 29, .9]
        pose = [0, *np.asarray(det[2:], dtype=np.float32).tolist(), *np.asarray(kp).ravel().tolist()]
        self.det.write_text(",".join(map(str, det))+"\n")
        self.source.write_text(" ".join(map(str, pose))+"\n")
        self.options["detections"] = self.det
        return pose, det

    def prepare(self):
        return prepare_model_points(self.export, self.source, self.output, **self.options)

    def project(self, **extra):
        return project_model_ground(self.root, self.export, self.source, self.output, **self.options, **extra)

    def csv(self, relative):
        with (self.output/relative).open() as stream:
            return list(csv.DictReader(stream))

    def test_yachiyo_frame_and_serial_id_zero(self):
        self.yachiyo()
        report = self.prepare()
        row = self.csv("point_map.csv")[0]
        self.assertEqual((row["model_frame"], row["source_frame"], row["native_serial"], row["native_track_id"]), ("1", "277", "00000000", "0"))
        self.assertEqual((float(row["pixel_x"]), float(row["pixel_y"])), (50, 30))
        self.assertFalse(report["model_inference_run"])
        self.assertEqual(report["input_kind"], "synthetic_fixture")

    def test_yachiyo_unassigned_id_not_promoted(self):
        self.yachiyo(OfflineID=-1)
        self.prepare()
        self.assertEqual(self.csv("point_map.csv")[0]["native_track_id"], "-1")

    def test_yachiyo_candidate_changes_only_world(self):
        original = self.yachiyo()
        before = self.source.read_bytes()
        report = self.project(emit_yachiyo_candidate=True)
        candidate = json.loads((self.output/report["native_candidate"]).read_text())["00000000"]
        world = candidate.pop("WorldCoordinate")
        original.pop("WorldCoordinate")
        self.assertEqual(candidate, original)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertAlmostEqual(world["x"], 0)
        self.assertAlmostEqual(world["y"], -1)
        self.assertFalse(report["native_tracker_connected"])

    def test_candidate_is_explicit_opt_in(self):
        self.yachiyo()
        report = self.project()
        self.assertIsNone(report["native_candidate"])
        self.assertFalse((self.output/"native_candidate").exists())

    def test_yachiyo_failed_point_blocks_candidate_and_cleans_output(self):
        self.yachiyo(Coordinate={"x1": 40, "y1": 10, "x2": 60, "y2": 100})
        with self.assertRaises(ValueError):
            self.project(emit_yachiyo_candidate=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob(".output-*")), [])

    def test_yachiyo_noncanonical_coordinate_rejected(self):
        self.yachiyo(Coordinate={"x1": 40.5, "y1": 10, "x2": 60, "y2": 30})
        with self.assertRaises(ValueError):
            self.prepare()

    def test_yachiyo_duplicate_json_keys_and_nonfinite_rejected(self):
        self.yachiyo()
        self.source.write_text('{"0":{},"0":{}}')
        with self.assertRaises(ValueError):
            self.prepare()
        self.source.write_text('{"0":{"Frame":NaN}}')
        with self.assertRaises(ValueError):
            self.prepare()

    def test_native_frame_outside_clip_rejected(self):
        for frame in (0, 2, 277):
            self.yachiyo(Frame=frame)
            with self.assertRaises(ValueError):
                self.prepare()

    def test_empty_yachiyo_keeps_empty(self):
        self.yachiyo()
        self.source.write_text('{}')
        report = self.project(emit_yachiyo_candidate=True)
        self.assertEqual(report["row_count"], 0)
        self.assertEqual(self.csv("ground_candidates.csv"), [])
        self.assertEqual(json.loads((self.output/report["native_candidate"]).read_text()), {})

    def test_uwipl_ankles_high_confidence(self):
        self.uwipl()
        self.prepare()
        row = self.csv("point_map.csv")[0]
        self.assertEqual((float(row["pixel_x"]), float(row["pixel_y"])), (50, 27))
        self.assertEqual(row["point_method"], "ankle_mean_gt_0.5")
        self.assertEqual(row["id_scope"], "sequence")

    def test_uwipl_threshold_is_strict_and_bbox_95pct(self):
        self.uwipl(.5)
        self.prepare()
        row = self.csv("point_map.csv")[0]
        self.assertEqual(float(row["pixel_y"]), 29)
        self.assertEqual(row["point_method"], "bbox_95pct_fallback")

    def test_uwipl_large_id_preserved_exactly(self):
        row = self.uwipl()
        row[1] = 9007199254740993
        self.source.write_text(",".join(map(str, row))+"\n")
        self.prepare()
        self.assertEqual(self.csv("point_map.csv")[0]["native_track_id"], "9007199254740993")

    def test_uwipl_outside_image_not_clipped(self):
        row = self.uwipl()
        row[10], row[13] = -30, -20
        self.source.write_text(",".join(map(str, row))+"\n")
        report = self.project()
        self.assertEqual(report["row_status_counts"], {"outside_image": 1})
        self.assertEqual(self.csv("ground_candidates.csv")[0]["world_x"], "")

    def test_uwipl_duplicate_identity_rejected(self):
        self.uwipl()
        self.source.write_text(self.source.read_text()*2)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_uwipl_wrong_columns_and_nan(self):
        row = self.uwipl()
        for data in (row[:-1], [*row[:10], "nan", *row[11:]]):
            self.source.write_text(",".join(map(str, data))+"\n")
            with self.assertRaises(ValueError):
                self.prepare()

    def test_posetrack_pair_and_diagnostic_policy(self):
        self.posetrack()
        report = self.prepare()
        row = self.csv("point_map.csv")[0]
        self.assertEqual((float(row["pixel_x"]), float(row["pixel_y"])), (50, 27))
        self.assertEqual(row["native_track_id"], "")
        self.assertEqual(row["id_scope"], "none")
        self.assertTrue(report["pose_detection_row_alignment_checked"])
        self.assertFalse(report["point_policy_native_equivalent"])
        self.assertFalse(report["reid_alignment_checked"])

    def test_posetrack_low_confidence_has_no_bbox_fallback(self):
        self.posetrack(.7)
        report = self.project()
        self.assertEqual(report["row_status_counts"], {"low_ankle_confidence": 1})
        self.assertEqual(self.csv("ground_candidates.csv")[0]["world_x"], "")

    def test_posetrack_missing_detection_pair_rejected(self):
        self.posetrack()
        self.options.pop("detections")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_posetrack_mismatched_box_score_and_frame_rejected(self):
        for field in (0, 2, 6):
            pose, det = self.posetrack()
            det[field] += .1 if field == 6 else 1
            self.det.write_text(",".join(map(str, det))+"\n")
            with self.assertRaises(ValueError):
                self.prepare()

    def test_posetrack_reordered_rows_rejected(self):
        pose, det = self.posetrack()
        p2, d2 = pose.copy(), det.copy()
        p2[1] += 1
        d2[2] += 1
        self.source.write_text("\n".join(" ".join(map(str, p)) for p in (p2, pose))+"\n")
        self.det.write_text("\n".join(",".join(map(str, d)) for d in (det, d2))+"\n")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_posetrack_row_count_mismatch_rejected(self):
        self.posetrack()
        self.det.write_text("")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_posetrack_duplicate_bbox_ambiguous_rejected(self):
        self.posetrack()
        self.source.write_text(self.source.read_text()*2)
        self.det.write_text(self.det.read_text()*2)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_empty_posetrack_rows_preserved(self):
        self.posetrack()
        self.source.write_text("")
        self.det.write_text("")
        self.assertEqual(self.prepare()["native_row_count"], 0)

    def test_wrong_profile_and_policy_rejected(self):
        self.yachiyo()
        self.options["policy"] = "ankles-only"
        with self.assertRaises(ValueError):
            self.prepare()
        self.options["policy"] = "yachiyo-bbox-bottom"
        self.write_json(self.export/"profile.json", {"name": "uwipl", "frame_base": 0})
        with self.assertRaises(ValueError):
            self.prepare()

    def test_wrong_id_scope_rejected(self):
        self.yachiyo()
        self.options["id_scope"] = "sequence"
        with self.assertRaises(ValueError):
            self.prepare()

    def test_wrong_camera_and_frame_mapping_rejected(self):
        self.yachiyo()
        self.options["camera"] = "cam_8"
        with self.assertRaises(ValueError):
            self.prepare()
        self.options["camera"] = "cam_0"
        self.manifest["cameras"][0]["frames"][0]["index"] = 1
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_wrong_image_space_rejected(self):
        self.yachiyo()
        self.manifest["image_space"] = "undistorted"
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_input_output_overlap_and_no_overwrite(self):
        self.yachiyo()
        with self.assertRaises(ValueError):
            prepare_model_points(self.export, self.source, self.export/"output", **self.options)
        self.prepare()
        before = (self.output/"bridge.json").read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual((self.output/"bridge.json").read_bytes(), before)

    def test_uwipl_cannot_emit_yachiyo_candidate(self):
        self.uwipl()
        with self.assertRaises(ValueError):
            self.project(emit_yachiyo_candidate=True)

    def test_no_ground_hit_keeps_row_without_xyz(self):
        self.yachiyo()
        fixtures.write_ply(self.root/"meshes/mesh_ground_only.ply", *fixtures.square(size=.1))
        report = self.project()
        self.assertEqual(report["row_count"], 1)
        self.assertEqual(report["row_status_counts"], {"no_ground_intersection": 1})
        self.assertEqual(self.csv("ground_candidates.csv")[0]["world_z"], "")
