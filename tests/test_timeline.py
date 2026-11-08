"""Timing selection regression cases; synthetic metadata only."""
import unittest
from scout_adapter.timeline import analyze_timeline


class TimelineTests(unittest.TestCase):
    cameras = [{"id": "cam_0", "timestamp_key": "a"}, {"id": "cam_1", "timestamp_key": "b"}]

    def run_case(self, values, missing=()):
        times = {str(i): {"a": a, "b": b} for i, (a, b) in enumerate(values)}
        images = {"cam_0": set(range(len(values))), "cam_1": set(range(len(values)))-set(missing)}
        return analyze_timeline(times, self.cameras, images, tolerance=0.04, max_step=0.15)

    def test_duplicate_breaks_window_without_reusing_time(self):
        records, runs = self.run_case([(0, .01), (.1, .11), (.1, .11), (.2, .21)])
        self.assertEqual([(r["start_frame"], r["end_frame"]) for r in runs], [(0, 1), (2, 3)])
        self.assertIn("duplicate_timestamp:cam_0", records[2]["step_from_previous_reasons"])

    def test_bad_skew_is_not_in_any_clean_run(self):
        _, runs = self.run_case([(0, .01), (.1, .2), (.3, .31), (.4, .41)])
        self.assertEqual([(r["start_frame"], r["end_frame"]) for r in runs], [(0, 0), (2, 3)])

    def test_missing_image_is_not_silently_reindexed(self):
        records, runs = self.run_case([(0, .01), (.1, .11), (.2, .21)], missing=[1])
        self.assertEqual([r["start_frame"] for r in runs], [0, 2])
        self.assertIn("missing_image:cam_1", records[1]["frame_reasons"])

    def test_backwards_and_gap_break_windows(self):
        records, runs = self.run_case([(0, .01), (.1, .11), (0, .01), (.5, .51)])
        self.assertEqual([r["frame_count"] for r in runs], [2, 1, 1])
        self.assertIn("backward_timestamp:cam_0", records[2]["step_from_previous_reasons"])
        self.assertIn("timestamp_gap:cam_0", records[3]["step_from_previous_reasons"])

    def test_missing_timestamp_cannot_pass(self):
        records, runs = analyze_timeline({"0": {"a": 0}}, self.cameras, {"cam_0": {0}, "cam_1": {0}})
        self.assertEqual(runs, [])
        self.assertIsNone(records[0]["max_skew_seconds"])

    def test_frame_id_gap_breaks_window(self):
        _, runs = analyze_timeline({"0": {"a": 0, "b": 0}, "2": {"a": .1, "b": .1}}, self.cameras,
                                   {"cam_0": {0, 2}, "cam_1": {0, 2}})
        self.assertEqual([r["frame_count"] for r in runs], [1, 1])

    def test_nonfinite_timestamp_and_single_camera_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case([(float("nan"), 0)])
        with self.assertRaises(ValueError):
            analyze_timeline({"0": {"a": 0}}, self.cameras[:1], {"cam_0": {0}})
