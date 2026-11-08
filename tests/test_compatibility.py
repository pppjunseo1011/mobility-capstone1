"""Synthetic input-contract assessment tests; no downloads or model execution."""

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from scout_adapter.compatibility import GENERATIONS, assess_dataset, timing_summary


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scout-compatibility-test-")
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / "source" / "scout"
        self.root.mkdir(parents=True)
        for number in range(3):
            self._calibration(number)
        self._timestamps({
            "0": {"device0": 0.0, "device1": 0.0, "device2": 0.0},
            "1": {"device0": 0.125, "device1": 0.125, "device2": 0.125},
        })

    def _json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def _calibration(self, number):
        self._json(self.root / "calibrations" / ("cam_{}.json".format(number)), {
            "K": [[200, 0, 16], [0, 200, 12], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "T": [[number], [0], [1]],
            "dist": [0.1 if number == 0 else 0.0, 0, 0, 0, 0],
            "view_id": "device{}_0".format(number),
        })

    def _timestamps(self, values, sequence=1):
        self._json(self.root / "timestamps" / (
            "timestamps_sequence_{}.json".format(sequence)), values)

    def _cameras(self, count=2, unavailable=()):
        return [{"id": "cam_{}".format(number),
                 "timestamp_key": None if number in unavailable else "device{}".format(number),
                 "timestamps_available": number not in unavailable}
                for number in range(count)]

    def _source_snapshot(self):
        return {str(path.relative_to(self.root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file()}

    def _no_partial_output(self, output):
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob("." + output.name + "-*")), [])

    def test_zero_timestamp_is_valid_and_zero_tolerance_can_pass(self):
        report = timing_summary({
            "0": {"device0": 0, "device1": 0},
            "1": {"device0": 0.125, "device1": 0.125},
        }, self._cameras(), 0)
        self.assertEqual(report["frames_with_all_compared_timestamps"], 2)
        self.assertEqual(report["frames_missing_compared_timestamps"], 0)
        self.assertEqual(report["max_skew_seconds"], 0)
        self.assertTrue(report["selected_camera_coverage_complete"])
        self.assertTrue(report["all_selected_same_frame_within_tolerance"])
        self.assertFalse(report["synchronization_performed"])

    def test_skew_statistics_and_strictly_over_tolerance_count(self):
        # Binary fractions avoid making a boundary test depend on float rounding.
        timestamps = {
            "3": {"device0": 3, "device1": 3.375},
            "1": {"device0": 1, "device1": 1.125},
            "0": {"device0": 0, "device1": 0},
            "2": {"device0": 2, "device1": 2.25},
        }
        report = timing_summary(timestamps, self._cameras(), 0.125)
        self.assertEqual(report["max_skew_seconds"], 0.375)
        self.assertEqual(report["median_skew_seconds"], 0.1875)
        self.assertEqual(report["p95_skew_seconds"], 0.375)
        self.assertEqual(report["frames_over_tolerance"], 2)
        self.assertTrue(report["selected_camera_coverage_complete"])
        self.assertFalse(report["all_selected_same_frame_within_tolerance"])

    def test_unavailable_camera_is_retained_and_prevents_overall_pass(self):
        report = timing_summary({"0": {"device0": 0, "device1": 0}},
                                self._cameras(3, unavailable=(2,)), 0.05)
        self.assertEqual(report["compared_cameras"], ["cam_0", "cam_1"])
        self.assertEqual(report["unavailable_cameras"], ["cam_2"])
        self.assertEqual(report["max_skew_seconds"], 0)
        self.assertEqual(report["frames_with_all_compared_timestamps"], 1)
        self.assertFalse(report["selected_camera_coverage_complete"])
        self.assertIsNone(report["all_selected_same_frame_within_tolerance"])

    def test_missing_timestamp_frame_is_not_counted_as_zero_skew(self):
        report = timing_summary({
            "0": {"device0": 0, "device1": 0},
            "1": {"device0": 0.125},
        }, self._cameras(), 0.05)
        self.assertEqual(report["total_timestamp_frames"], 2)
        self.assertEqual(report["frames_with_all_compared_timestamps"], 1)
        self.assertEqual(report["frames_missing_compared_timestamps"], 1)
        self.assertFalse(report["selected_camera_coverage_complete"])
        self.assertIsNone(report["all_selected_same_frame_within_tolerance"])

    def test_single_or_no_available_camera_cannot_pass_synchronization(self):
        cases = [self._cameras(1), self._cameras(2, unavailable=(0, 1))]
        for cameras in cases:
            with self.subTest(cameras=cameras):
                report = timing_summary({"0": {"device0": 0}}, cameras, 0.05)
                self.assertFalse(report["cross_camera_comparison_possible"])
                self.assertFalse(report["selected_camera_coverage_complete"])
                self.assertIsNone(report["max_skew_seconds"])
                self.assertIsNone(report["all_selected_same_frame_within_tolerance"])
                self.assertFalse(report["synchronization_performed"])

    def test_invalid_tolerance_and_noncanonical_frame_keys_leave_no_output(self):
        before = self._source_snapshot()
        for index, tolerance in enumerate((float("nan"), float("inf"), -float("inf"), -0.01, True)):
            with self.subTest(tolerance=tolerance):
                output = self.temp / ("invalid-tolerance-{}".format(index))
                with self.assertRaises(ValueError):
                    assess_dataset(self.root, tolerance=tolerance, output=output)
                self._no_partial_output(output)
                self.assertEqual(self._source_snapshot(), before)
        for index, key in enumerate(("01", "1e0")):
            with self.subTest(frame_key=key):
                self._timestamps({key: {"device0": 0, "device1": 0, "device2": 0}})
                before = self._source_snapshot()
                output = self.temp / ("invalid-frame-key-{}".format(index))
                with self.assertRaises(ValueError):
                    assess_dataset(self.root, output=output)
                self._no_partial_output(output)
                self.assertEqual(self._source_snapshot(), before)

    def test_sequence_two_cam_aliases_report_all_three_unavailable_cameras(self):
        # Reproduce the observed 25 calibrations / 22 timestamp keys shape with
        # synthetic values, without reading the downloaded SCOUT metadata.
        for number in range(3, 25):
            self._calibration(number)
        available = [number for number in range(25) if number not in (1, 3, 4)]
        self._timestamps({"0": {"cam{}".format(number): 0 for number in available}}, sequence=2)
        report = assess_dataset(self.root, sequence=2)
        self.assertEqual(report["data"]["camera_count"], 25)
        self.assertEqual(report["data"]["timestamp_camera_count"], 22)
        self.assertEqual(report["timing"]["unavailable_cameras"], ["cam_1", "cam_3", "cam_4"])
        self.assertEqual(len(report["cameras"]), 25)
        self.assertIsNone(report["timing"]["all_selected_same_frame_within_tolerance"])
        self.assertTrue(any("timestamp" in item for item in report["blockers"]))
        selected = assess_dataset(self.root, sequence=2, cameras=["cam_0", "cam_2"])
        self.assertEqual(selected["timing"]["unavailable_cameras"], [])
        self.assertTrue(selected["timing"]["all_selected_same_frame_within_tolerance"])
        self.assertFalse(selected["model_inference_ready"])

    def test_metadata_only_assessment_writes_reports_without_execution(self):
        output = self.temp / "metadata-only"
        before = self._source_snapshot()
        with mock.patch("subprocess.run", side_effect=AssertionError("must not launch a process")), \
             mock.patch("scout_adapter.dataset.jpeg_dimensions", side_effect=AssertionError("must not decode images")):
            report = assess_dataset(self.root, output=output)
        self.assertEqual({path.name for path in output.iterdir()}, {"compatibility.json", "comparison.md"})
        self.assertEqual(json.loads((output / "compatibility.json").read_text(encoding="utf-8")), report)
        self.assertEqual(report["data"]["cameras_with_images"], 0)
        self.assertEqual(report["data"]["nonzero_distortion_cameras"], 1)
        self.assertEqual(report["model_frame_bases"], {"yachiyo": 1, "uwipl": 0, "posetrack": 0})
        for field in ("model_inference_ready", "model_inference_run", "full_dataset_validated", "ground_truth_used_as_input"):
            self.assertFalse(report[field])
        self.assertFalse(report["data"]["annotation_content_validated"])
        self.assertFalse(report["data"]["mesh_validated"])
        self.assertTrue(all(not row["image_pixels_decoded"] for row in report["cameras"]))
        self.assertTrue(any("JPEG" in item for item in report["blockers"]))
        self.assertEqual(self._source_snapshot(), before)

    def test_json_and_markdown_requirements_have_the_same_six_columns(self):
        output = self.temp / "comparison-table"
        assess_dataset(self.root, output=output)
        report = json.loads((output / "compatibility.json").read_text(encoding="utf-8"))
        markdown = (output / "comparison.md").read_text(encoding="utf-8")
        table_lines = [line for line in markdown.splitlines() if line.startswith("| ")][1:]
        self.assertEqual(len(table_lines), len(report["requirements"]))
        self.assertEqual(len({item["id"] for item in report["requirements"]}), len(table_lines))
        for item, line in zip(report["requirements"], table_lines):
            cells = [cell.strip().replace("\\|", "|")
                     for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
            expected = [str(item[key]).replace("\n", " ") for key in (
                "information", "scout", "yachiyo", "uwipl", "posetrack")]
            expected.append(GENERATIONS[item["generation"]])
            self.assertEqual(cells, expected)
        self.assertIn("model_inference_ready=false", markdown)

    def test_overlap_and_existing_output_preserve_source_and_existing_files(self):
        before = self._source_snapshot()
        for output in (self.root, self.root / "report", self.root.parent):
            with self.subTest(output=output), self.assertRaises(ValueError):
                assess_dataset(self.root, output=output)
            self.assertEqual(self._source_snapshot(), before)
        self.assertFalse((self.root / "report").exists())
        output = self.temp / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("preserve this output", encoding="utf-8")
        with self.assertRaises(ValueError):
            assess_dataset(self.root, output=output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve this output")
        self.assertEqual({path.name for path in output.iterdir()}, {"keep.txt"})
        self.assertEqual(list(output.parent.glob("." + output.name + "-*")), [])
        self.assertEqual(self._source_snapshot(), before)

    def test_report_generation_failure_does_not_commit_partial_json(self):
        before = self._source_snapshot()
        output = self.temp / "incomplete-report"
        with mock.patch("scout_adapter.compatibility.assessment_markdown", side_effect=RuntimeError("synthetic write failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic write failure"):
                assess_dataset(self.root, output=output)
        self._no_partial_output(output)
        self.assertEqual(self._source_snapshot(), before)

    def test_sparse_timestamp_observations_report_backward_and_duplicate_steps(self):
        self._timestamps({
            "10": {"device0": 4.0},
            "0": {"device0": 5.0},
            "2": {"device0": 4.0},
        })
        report = assess_dataset(self.root, cameras=["cam_0"])
        camera = report["cameras"][0]
        self.assertEqual(camera["timestamp_count"], 3)
        self.assertEqual(camera["backward_timestamp_steps"], 1)
        self.assertEqual(camera["duplicate_timestamp_steps"], 1)
        self.assertEqual(camera["non_increasing_timestamp_steps"], 2)
        self.assertIsNone(camera["estimated_fps"])
        self.assertTrue(any("시간 역행" in item for item in report["blockers"]))
        self.assertIsNone(report["timing"]["all_selected_same_frame_within_tolerance"])
        self.assertFalse(report["model_inference_ready"])


if __name__ == "__main__":
    unittest.main()
