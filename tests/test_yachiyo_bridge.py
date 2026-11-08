"""Synthetic stage-bridge tests only: no actual detections, weights or inference."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scout_adapter.bridges.yachiyo import prepare_pose_csv, restore_pose_json
from scout_adapter.profiles.yachiyo import build_profile


class YachiyoFrameBridgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-yachiyo-bridge-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "original"
        self.original.mkdir()
        self.export = self.root / "export"
        self.export.mkdir()
        self.counter = 0
        self.manifest = {
            "schema_version": 1, "model": "yachiyo", "model_frame_base": 1,
            "source_root": str(self.original), "sequence": "sequence_1", "sequence_number": 1,
            "frame_ids": [81, 82, 83], "image_space": "raw", "fixture_kind": "synthetic_fixture",
            "sync": {"max_skew_seconds": 0.01, "tolerance_seconds": 0.05}, "cameras": [],
        }
        for number in (0, 7):
            camera = "cam_{}".format(number)
            self.manifest["cameras"].append({
                "id": camera, "number": number, "model_camera_id": number + 1,
                "width": 32, "height": 24, "calibration": None,
                "frames": [{"index": index, "source_frame": frame,
                            "source_path": "images/sequence_1/{}/image_{}.jpg".format(camera, frame),
                            "timestamp_seconds": 100 + index / 10 + number / 1000}
                           for index, frame in enumerate((81, 82, 83))],
            })
        self.profile = build_profile(self.manifest)
        self._write_contract()
        self.csv = self.root / "synthetic-detector.txt"
        self.csv.write_text("camera_0001,1,1,2,3,12,20,0.912345\n"
                            "camera_0001,3,1,4,1,16,22,0.8\n")

    def _json(self, path, data):
        path.write_text(json.dumps(data, allow_nan=True) + "\n")
        return path

    def _write_contract(self):
        self._json(self.export / "manifest.json", self.manifest)
        self._json(self.export / "profile.json", self.profile)

    def _destination(self, prefix="output"):
        self.counter += 1
        return self.root / (prefix + "-" + str(self.counter))

    def _prepare(self, **kwargs):
        output = self._destination("prepared")
        report = prepare_pose_csv(self.export, self.csv, output,
                                  camera=kwargs.pop("camera", "cam_0"),
                                  input_kind=kwargs.pop("input_kind", "synthetic_fixture"), **kwargs)
        return output, report

    def _pose(self, bbox):
        # Heatmap confidence is not constrained to [0,1]; unclipped keypoint
        # positions outside the image can also occur in the real pose API.
        return {"bbox": list(bbox) + [1.0], "keypoints": [[-1.5, 25.5, 1.2] for _ in range(17)]}

    def _poses(self):
        return {"0": [self._pose((2, 3, 12, 20))], "2": [self._pose((4, 1, 16, 22))]}

    def _restore(self, prepared, payload=None, **kwargs):
        source = self._json(self._destination("synthetic-pose").with_suffix(".json"),
                            self._poses() if payload is None else payload)
        output = self._destination("restored")
        report = restore_pose_json(self.export, source, output, camera=kwargs.pop("camera", "cam_0"),
                                   pose_input=prepared, input_kind=kwargs.pop("input_kind", "synthetic_fixture"), **kwargs)
        return source, output, report

    def _not_published(self, output):
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob("." + output.name + "-*")), [])

    def _snapshot(self, paths):
        return {str(path): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in paths}

    def test_csv_changes_only_frame_and_maps_nonzero_original_frames(self):
        original = self.csv.read_text().splitlines()
        prepared, report = self._prepare()
        output = (prepared / "pose_input.csv").read_text().splitlines()
        self.assertEqual(len(output), 2)
        for source, target, frame in zip(original, output, ("0", "2")):
            cells = source.split(",")
            cells[1] = frame
            self.assertEqual(target.split(","), cells)
        with (prepared / "frame_map.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["source_frame"]) for row in rows], [81, 82, 83])
        self.assertEqual([int(row["model_frame"]) for row in rows], [1, 2, 3])
        self.assertEqual([int(row["pose_frame"]) for row in rows], [0, 1, 2])
        self.assertEqual(report["detection_count"], 2)
        self.assertFalse(report["model_inference_run"])
        self.assertFalse(report["input_provenance_verified"])

    def test_restore_preserves_bbox_keypoints_and_only_adds_one_to_frame(self):
        prepared, _ = self._prepare()
        source, restored, report = self._restore(prepared)
        result = json.loads((restored / "camera_0001_out_keypoint.json").read_text())
        original = json.loads(source.read_text())
        self.assertEqual(result, {"1": original["0"], "3": original["2"]})
        self.assertNotIn("2", result)
        self.assertEqual(report["input_kind"], "synthetic_fixture")
        self.assertTrue(report["all_prepared_detections_matched"])
        self.assertFalse(report["model_inference_ready"])
        self.assertFalse(report["video_alignment_verified"])

    def test_empty_inputs_remain_empty_without_inventing_detections(self):
        self.csv.write_text("")
        prepared, before = self._prepare()
        _, restored, after = self._restore(prepared, {})
        self.assertEqual((prepared / "pose_input.csv").read_bytes(), b"")
        self.assertEqual(json.loads((restored / "camera_0001_out_keypoint.json").read_text()), {})
        self.assertEqual(before["detection_count"], 0)
        self.assertEqual(after["detection_count"], 0)

    def test_single_row_and_nonselected_camera_number_are_supported(self):
        self.csv.write_text("camera_0008,2,1,0,0,32,24,1\n")
        prepared, report = self._prepare(camera="cam_7")
        self.assertEqual((prepared / "pose_input.csv").read_text(), "camera_0008,1,1,0,0,32,24,1\n")
        _, restored, _ = self._restore(prepared, {"1": [self._pose((0, 0, 32, 24))]}, camera="cam_7")
        self.assertTrue((restored / "camera_0008_out_keypoint.json").is_file())
        self.assertEqual(report["model_camera_id"], 8)

    def test_frame_outside_export_and_wrong_frame_base_are_rejected(self):
        for frame in ("0", "4", "81", "1.0", "01", "-1"):
            with self.subTest(frame=frame):
                self.csv.write_text("camera_0001,{},1,2,3,12,20,0.9\n".format(frame))
                output = self._destination()
                with self.assertRaises(ValueError):
                    prepare_pose_csv(self.export, self.csv, output, camera="cam_0")
                self._not_published(output)

    def test_native_camera_and_class_are_required_not_mot_identity(self):
        for camera, category in (("1", "1"), ("camera_0008", "1"), ("camera_0001", "7"), ("camera_0001", "0")):
            with self.subTest(camera=camera, category=category):
                self.csv.write_text("{},1,{},2,3,12,20,0.9\n".format(camera, category))
                with self.assertRaises(ValueError):
                    self._prepare()

    def test_invalid_columns_and_blank_rows_are_rejected(self):
        for text in ("1,2,3,4,5,6,0.9,-1,-1,-1\n", "\n", "camera_0001,1,1,2,3,12,20\n",
                     "camera_0001, 1,1,2,3,12,20,0.9\n"):
            with self.subTest(text=text):
                self.csv.write_text(text)
                with self.assertRaises(ValueError):
                    self._prepare()

    def test_invalid_boxes_and_detector_confidence_are_rejected(self):
        values = [("2,3,2,20", "0.9"), ("2,3,12,25", "0.9"), ("-1,3,12,20", "0.9"),
                  ("2.5,3,12,20", "0.9"), ("2,3,12,20", "nan"), ("2,3,12,20", "inf"),
                  ("2,3,12,20", "1.1"), ("2,3,12,20", "-0.1")]
        for bbox, score in values:
            with self.subTest(bbox=bbox, score=score):
                self.csv.write_text("camera_0001,1,1,{},{}\n".format(bbox, score))
                with self.assertRaises(ValueError):
                    self._prepare()

    def test_duplicate_frame_bbox_is_rejected_even_with_different_scores(self):
        self.csv.write_text("camera_0001,1,1,2,3,12,20,0.9\ncamera_0001,1,1,2,3,12,20,0.8\n")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self._prepare()

    def test_detection_order_is_not_silently_sorted(self):
        self.csv.write_text("camera_0001,3,1,2,3,12,20,0.9\ncamera_0001,1,1,4,1,16,22,0.8\n")
        with self.assertRaisesRegex(ValueError, "order"):
            self._prepare()

    def test_wrong_model_base_or_image_space_is_rejected(self):
        for key, value in (("model", "posetrack"), ("model_frame_base", 0), ("image_space", "undistorted")):
            old = self.manifest[key]
            self.manifest[key] = value
            self._write_contract()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._prepare()
            self.manifest[key] = old
        self._write_contract()

    def test_manifest_index_path_and_camera_mapping_errors_are_rejected(self):
        mutations = [(self.manifest["cameras"][0]["frames"][0], "index", 1),
                     (self.manifest["cameras"][0]["frames"][0], "source_path", "images/sequence_1/cam_0/image_82.jpg"),
                     (self.manifest["cameras"][0], "model_camera_id", 8)]
        for record, key, value in mutations:
            old = record[key]
            record[key] = value
            self._write_contract()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._prepare()
            record[key] = old
        self._write_contract()

    def test_unknown_camera_and_duplicate_manifest_camera_are_rejected(self):
        with self.assertRaises(ValueError):
            self._prepare(camera="cam_1")
        self.manifest["cameras"].append(self.manifest["cameras"][0])
        self._write_contract()
        with self.assertRaises(ValueError):
            self._prepare()

    def test_restore_rejects_missing_frames_detections_and_duplicates(self):
        prepared, _ = self._prepare()
        fixtures = [{"0": [self._pose((2, 3, 12, 20))]},
                    {"0": [], "2": [self._pose((4, 1, 16, 22))]},
                    {"0": [self._pose((2, 3, 12, 20))] * 2, "2": [self._pose((4, 1, 16, 22))]}]
        for payload in fixtures:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self._restore(prepared, payload)

    def test_restore_rejects_unshifted_frames_unknown_boxes_and_invented_empty_frame(self):
        prepared, _ = self._prepare()
        fixtures = [{"1": [self._pose((2, 3, 12, 20))], "3": [self._pose((4, 1, 16, 22))]},
                    {"0": [self._pose((3, 3, 12, 20))], "2": [self._pose((4, 1, 16, 22))]},
                    dict(self._poses(), **{"1": []})]
        for payload in fixtures:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self._restore(prepared, payload)

    def test_pose_requires_native_score_coco17_and_finite_keypoints(self):
        prepared, _ = self._prepare()
        mutations = [lambda p: p["bbox"].__setitem__(4, 0.9),
                     lambda p: p.__setitem__("keypoints", p["keypoints"][:16]),
                     lambda p: p["keypoints"][0].__setitem__(0, float("nan")),
                     lambda p: p.__setitem__("track_id", 0),
                     lambda p: p["bbox"].__setitem__(2, 33)]
        for mutation in mutations:
            payload = self._poses()
            mutation(payload["0"][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self._restore(prepared, payload)

    def test_duplicate_json_keys_and_numeric_aliases_are_rejected(self):
        prepared, _ = self._prepare()
        for text in ('{"0":[],"0":[]}', '{"00":[]}', '{"0":NaN}', '{"-1":[]}'):
            source = self._destination().with_suffix(".json")
            source.write_text(text)
            output = self._destination()
            with self.subTest(text=text), self.assertRaises(ValueError):
                restore_pose_json(self.export, source, output, camera="cam_0", pose_input=prepared,
                                  input_kind="synthetic_fixture")
            self._not_published(output)

    def test_prepared_csv_tampering_is_rejected_before_restore(self):
        prepared, _ = self._prepare()
        with (prepared / "pose_input.csv").open("a") as stream:
            stream.write("camera_0001,1,1,0,0,2,2,0.8\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            self._restore(prepared)

    def test_changed_manifest_or_profile_cannot_reuse_prepared_bridge(self):
        for name in ("manifest.json", "profile.json"):
            prepared, _ = self._prepare()
            path = self.export / name
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "metadata mismatch"):
                self._restore(prepared)
            path.write_bytes(before)

    def test_synthetic_input_cannot_be_relabeled_as_model_output(self):
        prepared, _ = self._prepare()
        for kind in ("model_output", "unverified"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Synthetic"):
                self._restore(prepared, input_kind=kind)
        with self.assertRaises(ValueError):
            self._prepare(input_kind="ground_truth")

    def test_unverified_is_default_and_never_claims_inference(self):
        prepared, report = self._prepare(input_kind="unverified")
        _, _, restored = self._restore(prepared, input_kind="unverified")
        for value in (report, restored):
            self.assertEqual(value["input_kind"], "unverified")
            self.assertFalse(value["input_provenance_verified"])
            self.assertFalse(value["model_inference_run"])
            self.assertFalse(value["model_inference_ready"])

    def test_source_bytes_and_mtimes_are_preserved(self):
        paths = [self.csv, self.export / "manifest.json", self.export / "profile.json"]
        before = self._snapshot(paths)
        prepared, _ = self._prepare()
        self.assertEqual(before, self._snapshot(paths))
        paths += [prepared / "pose_input.csv", prepared / "bridge.json", prepared / "frame_map.csv"]
        source = self._json(self._destination().with_suffix(".json"), self._poses())
        paths.append(source)
        before = self._snapshot(paths)
        restore_pose_json(self.export, source, self._destination(), camera="cam_0", pose_input=prepared,
                          input_kind="synthetic_fixture")
        self.assertEqual(before, self._snapshot(paths))

    def test_existing_and_overlapping_outputs_are_rejected(self):
        existing = self._destination()
        existing.mkdir()
        (existing / "sentinel").write_text("preserve")
        for output in (existing, self.export / "new", self.original / "new", self.csv):
            with self.subTest(output=output), self.assertRaises(ValueError):
                prepare_pose_csv(self.export, self.csv, output, camera="cam_0")
        self.assertEqual((existing / "sentinel").read_text(), "preserve")
        prepared, _ = self._prepare()
        source = self._json(self._destination().with_suffix(".json"), self._poses())
        for output in (prepared / "new", existing, source):
            with self.subTest(output=output), self.assertRaises(ValueError):
                restore_pose_json(self.export, source, output, camera="cam_0", pose_input=prepared,
                                  input_kind="synthetic_fixture")

    def test_write_failures_leave_no_partial_output(self):
        output = self._destination()
        with mock.patch("scout_adapter.bridges.yachiyo.write_json", side_effect=OSError("synthetic write failure")):
            with self.assertRaises(OSError):
                prepare_pose_csv(self.export, self.csv, output, camera="cam_0")
        self._not_published(output)
        prepared, _ = self._prepare()
        source = self._json(self._destination().with_suffix(".json"), self._poses())
        output = self._destination()
        from scout_adapter.common import write_json
        def fail_metadata(path, value):
            if path.name == "bridge.json":
                raise OSError("synthetic metadata failure")
            return write_json(path, value)
        with mock.patch("scout_adapter.bridges.yachiyo.write_json", side_effect=fail_metadata):
            with self.assertRaises(OSError):
                restore_pose_json(self.export, source, output, camera="cam_0", pose_input=prepared,
                                  input_kind="synthetic_fixture")
        self._not_published(output)


if __name__ == "__main__":
    unittest.main()
