"""Synthetic integration tests; no SCOUT download, weights or model inference."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from PIL import Image

from scout_adapter.__main__ import main
from scout_adapter.common import load_json
from scout_adapter.dataset import build_manifest, inspect_dataset
from scout_adapter.export import export_dataset
from scout_adapter.results import export_ground_truth, normalize_results


class AdapterIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scout-adapter-test-")
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / "source" / "scout"
        self.root.mkdir(parents=True)
        self.counter = 0
        self.frames = (0, 2, 10)
        for number, view in ((0, "c10_0"), (1, "c20_0")):
            self._json(self.root / "calibrations" / ("cam_{}.json".format(number)), {
                "K": [[200, 0, 16], [0, 200, 12], [0, 0, 1]],
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "T": [[number], [0], [1]],
                "dist": [[0.1, 0, 0, 0, 0]],
                "view_id": view,
            })
        self._populate_sequence(1, self.frames)

    def _json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _populate_sequence(self, sequence, frames):
        # Reverse insertion order exposes lexical/JSON-order frame mistakes.
        timestamps = {}
        for frame in reversed(frames):
            timestamps[str(frame)] = {"c10": 100 + frame / 10, "c20": 100.01 + frame / 10}
            for camera in (0, 1):
                path = self.root / "images" / ("sequence_{}".format(sequence)) / (
                    "cam_{}".format(camera)) / ("image_{}.jpg".format(frame))
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), (frame * 17 % 256, camera * 80, 120)).save(
                    path, format="JPEG", quality=95)
        self._json(self.root / "timestamps" / ("timestamps_sequence_{}.json".format(sequence)),
                   timestamps)

    def _new_path(self, prefix):
        self.counter += 1
        return self.temp / ("{}-{}".format(prefix, self.counter))

    def _export(self, model, **kwargs):
        output = self._new_path("export-" + model)
        report = export_dataset(self.root, output, model, maximum=3, **kwargs)
        return output, report

    def _native(self, text, suffix=".txt"):
        path = self._new_path("native").with_suffix(suffix)
        path.write_text(text, encoding="utf-8")
        return path

    def _csv(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def _normalize(self, export, source, format, scope="sequence", camera=None):
        output = self._new_path("normalized")
        summary = normalize_results(export, source, output, format, scope, camera)
        return self._csv(output / "predictions.csv"), summary, output

    def _snapshot(self):
        return {str(path.relative_to(self.root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file()}

    def _no_partial_output(self, output):
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob("." + output.name + "-*")), [])

    def _yachiyo_result(self):
        def entry(frame):
            return {"Frame": frame, "GlobalOfflineID": 0,
                    "Coordinate": {"x1": 2, "y1": 3, "x2": 6, "y2": 8},
                    "WorldCoordinate": {"x": -2.5, "y": 3.5}}
        return {"1": {"00000000": entry(1), "00000001": entry(3)},
                "2": {"00000000": entry(1)}}

    def test_three_profiles_copy_numeric_frames_and_preserve_sources(self):
        original = self._snapshot()
        for model, base in (("yachiyo", 1), ("uwipl", 0), ("posetrack", 0)):
            with self.subTest(model=model):
                output, report = self._export(model)
                manifest = load_json(output / "manifest.json")
                profile = load_json(output / "profile.json")
                self.assertEqual(manifest["frame_ids"], [0, 2, 10])
                self.assertEqual(manifest["model_frame_base"], base)
                self.assertEqual([row["timestamp_key"] for row in manifest["cameras"]], ["c10", "c20"])
                self.assertFalse(report["model_inference_run"])
                self.assertFalse(report["model_inference_ready"])
                self.assertFalse(report["ground_truth_used_as_input"])
                self.assertFalse(report["calibration_generated"])
                self.assertTrue(report["blockers"])
                mappings = self._csv(output / "frame_map.csv")
                self.assertEqual(len(mappings), 6)
                for camera in profile["cameras"]:
                    rows = [row for row in mappings if row["source_camera"] == camera["source_camera"]]
                    self.assertEqual([int(row["model_frame"]) for row in rows], list(range(base, base + 3)))
                    self.assertEqual([int(row["source_frame"]) for row in rows], [0, 2, 10])
                    self.assertEqual(int(rows[0]["model_camera_id"]),
                                     int(camera["source_camera"].split("_")[1]) + 1)
                    for row in rows:
                        original_bytes = (self.root / row["source_path"]).read_bytes()
                        self.assertEqual((output / row["model_frame_path"]).read_bytes(), original_bytes)
                        for directory in camera["frame_dirs"]:
                            filename = camera["frame_pattern"].format(frame=int(row["model_frame"]))
                            self.assertEqual((output / directory / filename).read_bytes(), original_bytes)
                    if camera.get("calibration_path"):
                        self.assertFalse((output / camera["calibration_path"]).exists())
                self.assertEqual(self._snapshot(), original)

    def test_manifest_only_does_not_copy_jpegs(self):
        output, report = self._export("yachiyo", materialize="manifest")
        self.assertEqual(report["materialize"], "manifest")
        self.assertEqual(len(self._csv(output / "frame_map.csv")), 6)
        self.assertEqual(list(output.rglob("*.jpg")), [])
        self.assertFalse(report["model_inference_ready"])

    def test_source_camera_subset_does_not_renumber_identity(self):
        output, _ = self._export("yachiyo", cameras=["cam_1"])
        camera = load_json(output / "profile.json")["cameras"][0]
        self.assertEqual(camera["source_camera"], "cam_1")
        self.assertEqual(camera["model_camera_id"], 2)
        self.assertEqual(camera["model_camera_name"], "camera_0002")

    def test_yachiyo_json_and_aic9_restore_source_frames_identically(self):
        output, _ = self._export("yachiyo")
        json_file = self._native(json.dumps(self._yachiyo_result()), ".json")
        json_rows, _, _ = self._normalize(output, json_file, "yachiyo-json")
        aic_file = self._native(
            "1 0 0 2 3 4 5 -2.5 3.5\n1 0 2 2 3 4 5 -2.5 3.5\n2 0 0 2 3 4 5 -2.5 3.5\n")
        aic_rows, _, _ = self._normalize(output, aic_file, "aic9")
        self.assertEqual(json_rows, aic_rows)
        self.assertEqual([(row["camera"], int(row["frame"])) for row in json_rows],
                         [("cam_0", 0), ("cam_0", 10), ("cam_1", 0)])
        self.assertEqual({row["track_id"] for row in json_rows}, {"yachiyo:sequence_1:0"})
        self.assertEqual({row["score"] for row in json_rows}, {""})
        self.assertEqual({row["world_z"] for row in json_rows}, {""})
        self.assertEqual(float(json_rows[1]["timestamp_seconds"]), 101.0)
        self.assertEqual(float(json_rows[0]["world_x"]), -2.5)

    def test_explicit_camera_scope_separates_same_native_ids(self):
        output, _ = self._export("yachiyo")
        source = self._native(json.dumps(self._yachiyo_result()), ".json")
        rows, summary, _ = self._normalize(output, source, "yachiyo-json", scope="camera")
        self.assertEqual(summary["id_scope"], "camera")
        self.assertEqual({row["track_id"] for row in rows},
                         {"yachiyo:sequence_1:cam_0:0", "yachiyo:sequence_1:cam_1:0"})

    def test_sequence_namespaces_do_not_merge_native_ids(self):
        source = self._native(json.dumps(self._yachiyo_result()), ".json")
        first, _ = self._export("yachiyo")
        first_rows, _, _ = self._normalize(first, source, "yachiyo-json")
        self._populate_sequence(2, self.frames)
        second, _ = self._export("yachiyo", sequence=2)
        second_rows, _, _ = self._normalize(second, source, "yachiyo-json")
        self.assertTrue({row["track_id"] for row in first_rows}.isdisjoint(
            {row["track_id"] for row in second_rows}))
        self.assertEqual({row["track_id"] for row in second_rows}, {"yachiyo:sequence_2:0"})

    def test_posetrack_position_source_is_not_confidence_or_world_z(self):
        output, _ = self._export("posetrack")
        source = self._native("1 7 0 2 3 4 5 1.25 -4.5 3\n2 7 2 2 3 4 5 1.25 -4.5 2\n")
        rows, _, _ = self._normalize(output, source, "posetrack10")
        self.assertEqual([int(row["frame"]) for row in rows], [0, 10])
        self.assertEqual([row["position_source"] for row in rows], ["3", "2"])
        self.assertEqual({row["track_id"] for row in rows}, {"posetrack:sequence_1:7"})
        self.assertEqual({row["score"] for row in rows}, {""})
        self.assertEqual({row["world_z"] for row in rows}, {""})
        self.assertEqual(float(rows[0]["world_x"]), 1.25)
        self.assertEqual(float(rows[0]["world_y"]), -4.5)

    def test_uwipl_mot_requires_source_camera(self):
        output, _ = self._export("uwipl")
        source = self._native("0,7,2,3,4,5,0.7,-1,-1,-1\n2,7,2,3,4,5,0.8,-1,-1,-1\n")
        failed = self._new_path("missing-camera")
        with self.assertRaisesRegex(ValueError, "requires --camera"):
            normalize_results(output, source, failed, "mot10", "camera")
        self._no_partial_output(failed)
        rows, _, _ = self._normalize(output, source, "mot10", scope="camera", camera="cam_1")
        self.assertEqual([int(row["frame"]) for row in rows], [0, 10])
        self.assertEqual({row["camera"] for row in rows}, {"cam_1"})
        self.assertEqual({row["track_id"] for row in rows}, {"uwipl:sequence_1:cam_1:7"})
        self.assertEqual([float(row["score"]) for row in rows], [0.7, 0.8])
        self.assertEqual({row["world_z"] for row in rows}, {""})

    def test_unknown_result_camera_frame_and_duplicate_are_rejected_atomically(self):
        output, _ = self._export("posetrack")
        invalid = {
            "camera": "99 7 0 2 3 4 5 1 2 1\n",
            "frame": "1 7 3 2 3 4 5 1 2 1\n",
            "duplicate": "1 7 0 2 3 4 5 1 2 1\n1 7 0 2 3 4 5 1 2 1\n",
        }
        for label, content in invalid.items():
            with self.subTest(error=label):
                source = self._native(content)
                destination = self._new_path("invalid-" + label)
                with self.assertRaises(ValueError):
                    normalize_results(output, source, destination, "posetrack10", "sequence")
                self._no_partial_output(destination)

    def test_result_camera_override_must_agree_with_native_camera(self):
        output, _ = self._export("posetrack")
        source = self._native("1 7 0 2 3 4 5 1 2 1\n")
        for camera in ("cam_1", "cam_99"):
            with self.subTest(camera=camera):
                destination = self._new_path("bad-camera")
                with self.assertRaises(ValueError):
                    normalize_results(output, source, destination, "posetrack10", "sequence", camera)
                self._no_partial_output(destination)

    def test_input_output_overlap_and_existing_output_preserve_sources(self):
        original = self._snapshot()
        for output in (self.root, self.root / "export", self.root.parent):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    export_dataset(self.root, output, "yachiyo", maximum=3)
                self.assertEqual(self._snapshot(), original)
        existing = self._new_path("existing")
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("keep existing work", encoding="utf-8")
        with self.assertRaises(ValueError):
            export_dataset(self.root, existing, "yachiyo", maximum=3)
        self.assertEqual(sentinel.read_text(), "keep existing work")
        self.assertEqual(list(existing.iterdir()), [sentinel])

    def test_copy_failure_removes_partial_export(self):
        output = self._new_path("failed-copy")
        original = self._snapshot()
        copyfile = shutil.copyfile
        calls = []

        def fail_after_first(source, destination):
            calls.append(str(destination))
            if len(calls) == 2:
                raise OSError("injected copy failure")
            return copyfile(source, destination)

        with mock.patch("scout_adapter.export.shutil.copyfile", side_effect=fail_after_first):
            with self.assertRaisesRegex(OSError, "injected copy failure"):
                export_dataset(self.root, output, "yachiyo", maximum=3)
        self.assertEqual(len(calls), 2)
        self._no_partial_output(output)
        self.assertEqual(self._snapshot(), original)

    def test_normalizer_refuses_existing_output(self):
        export, _ = self._export("posetrack")
        source = self._native("1 7 0 2 3 4 5 1 2 1\n")
        output = self._new_path("existing-normalized")
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaises(ValueError):
            normalize_results(export, source, output, "posetrack10", "sequence")
        self.assertEqual(sentinel.read_text(), "preserve")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_timestamp_aliases_and_repeated_samples_are_preserved(self):
        aliases = (("cam_0", "cam_1"), ("cam0", "cam1"), ("cvlabrpi10", "cvlabrpi20"))
        for keys in aliases:
            with self.subTest(keys=keys):
                for index, key in enumerate(keys):
                    path = self.root / "calibrations" / ("cam_{}.json".format(index))
                    calibration = load_json(path)
                    calibration["view_id"] = key
                    self._json(path, calibration)
                timestamps = {str(frame): {keys[0]: time, keys[1]: time + .01}
                              for frame, time in zip(self.frames, (100, 100, 100.2))}
                self._json(self.root / "timestamps" / "timestamps_sequence_1.json", timestamps)
                manifest = build_manifest(self.root, maximum=3)
                self.assertEqual([row["timestamp_key"] for row in manifest["cameras"]], list(keys))
                for row in manifest["cameras"]:
                    self.assertEqual(row["selected_duplicate_timestamp_count"], 1)
                    self.assertTrue(row["frames"][1]["timestamp_duplicate"])
                    self.assertEqual(row["frames"][0]["timestamp_seconds"],
                                     row["frames"][1]["timestamp_seconds"])

    def test_backward_timestamps_across_sparse_frames_are_rejected(self):
        timestamps = load_json(self.root / "timestamps" / "timestamps_sequence_1.json")
        timestamps["2"]["c10"] = 99
        self._json(self.root / "timestamps" / "timestamps_sequence_1.json", timestamps)
        with self.assertRaisesRegex(ValueError, "backward"):
            build_manifest(self.root, maximum=3)

    def test_ambiguous_timestamp_alias_requires_explicit_mapping(self):
        path = self.root / "timestamps" / "timestamps_sequence_1.json"
        timestamps = load_json(path)
        for row in timestamps.values():
            row["cam_0"] = row["c10"]
        self._json(path, timestamps)
        with self.assertRaisesRegex(ValueError, "uniquely map"):
            build_manifest(self.root, maximum=3)
        manifest = build_manifest(self.root, maximum=3, overrides={"cam_0": "c10"})
        self.assertEqual(manifest["cameras"][0]["timestamp_key"], "c10")

    def test_excessive_skew_is_reported_without_claiming_inference_ready(self):
        path = self.root / "timestamps" / "timestamps_sequence_1.json"
        timestamps = load_json(path)
        for row in timestamps.values():
            row["c20"] = row["c10"] + .2
        self._json(path, timestamps)
        _, report = self._export("yachiyo", tolerance=.05)
        self.assertFalse(report["sync"]["within_tolerance"])
        self.assertGreater(report["sync"]["max_skew_seconds"], .05)
        self.assertFalse(report["model_inference_ready"])
        self.assertTrue(any("skew" in message for message in report["blockers"]))

    def test_missing_camera_frame_is_not_silently_dropped(self):
        (self.root / "images/sequence_1/cam_1/image_2.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing selected frames"):
            build_manifest(self.root, maximum=3)

    def test_invalid_jpeg_header_is_rejected(self):
        (self.root / "images/sequence_1/cam_0/image_2.jpg").write_bytes(b"not a JPEG")
        output = self._new_path("invalid-jpeg")
        with self.assertRaisesRegex(ValueError, "JPEG"):
            export_dataset(self.root, output, "yachiyo", maximum=3)
        self._no_partial_output(output)

    def test_changing_jpeg_dimensions_is_rejected(self):
        path = self.root / "images/sequence_1/cam_0/image_2.jpg"
        Image.new("RGB", (40, 24), (2, 3, 4)).save(path, format="JPEG")
        output = self._new_path("resolution-change")
        with self.assertRaisesRegex(ValueError, "resolution changes"):
            export_dataset(self.root, output, "yachiyo", maximum=3)
        self._no_partial_output(output)

    def test_ground_truth_is_neither_read_nor_exported_as_detections(self):
        labels = self.root / "annotations" / "sequence_1"
        labels.mkdir(parents=True)
        (labels / "gt.txt").write_text("GT SHOULD NOT BECOME DETECTIONS", encoding="utf-8")
        (labels / "coco.json").write_text("deliberately invalid unused labels", encoding="utf-8")
        original = self._snapshot()
        for model in ("yachiyo", "uwipl", "posetrack"):
            with self.subTest(model=model):
                output, report = self._export(model)
                self.assertFalse(report["ground_truth_used_as_input"])
                self.assertFalse(load_json(output / "manifest.json")["labels_used_as_detections"])
                self.assertEqual(list(output.rglob("*.npy")), [])
                self.assertEqual(list(output.rglob("*.txt")), [])
                self.assertFalse((output / "predictions.csv").exists())
                self.assertFalse((output / "ground_truth.csv").exists())
        self.assertEqual(self._snapshot(), original)

    def test_cli_inspect_resolves_wrapper_root_and_exports_new_workspace(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["inspect", "--root", str(self.root.parent), "--sequence", "1"])
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["timestamp_frame_count"], 3)
        self.assertEqual(report["cameras"][0]["frame_range"], [0, 10])
        output = self._new_path("cli-export")
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["export", "--root", str(self.root), "--model", "yachiyo",
                         "--max-frames", "2", "--materialize", "manifest", "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(load_json(output / "manifest.json")["frame_ids"], [0, 2])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg/ffprobe unavailable; run inside the adapter image")
    def test_explicit_posetrack_video_preserves_dimensions_count_and_order(self):
        self._populate_sequence(2, (0, 1, 2))
        original = self._snapshot()
        output, report = self._export("posetrack", sequence=2, videos=True, fps=10)
        profile = load_json(output / "profile.json")
        for camera in profile["cameras"]:
            video = output / camera["video_path"]
            self.assertTrue(video.is_file())
            info = report["videos"][camera["source_camera"]]
            self.assertEqual((info["width"], info["height"], info["frames"]), (32, 24, 3))
            self.assertTrue(info["decoded_frame_count_verified"])
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
                 "-of", "json", str(video)], capture_output=True, text=True, check=True, timeout=30)
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual((stream["width"], stream["height"], int(stream["nb_read_frames"])),
                             (32, 24, 3))
            numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
            self.assertEqual(numerator / denominator, 10)
            decoded = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video), "-threads", "1", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True, timeout=30).stdout
            frame_bytes = 32 * 24 * 3
            self.assertEqual(len(decoded), 3 * frame_bytes)
            red_means = [sum(decoded[i * frame_bytes:(i + 1) * frame_bytes:3]) / (32 * 24)
                         for i in range(3)]
            self.assertLess(red_means[0], red_means[1])
            self.assertLess(red_means[1], red_means[2])
        self.assertFalse(report["model_inference_run"])
        self.assertEqual(self._snapshot(), original)

    def test_sparse_or_implicit_rate_video_is_rejected(self):
        for kwargs in ({"videos": True, "fps": 10}, {"videos": True},
                       {"videos": True, "fps": 10, "materialize": "manifest"}):
            with self.subTest(kwargs=kwargs):
                output = self._new_path("bad-video")
                with self.assertRaises(ValueError):
                    export_dataset(self.root, output, "posetrack", maximum=3, **kwargs)
                self._no_partial_output(output)

    def test_ground_truth_wrapper_root_and_identity_zero(self):
        export, _ = self._export("posetrack")
        for camera in (0, 1):
            path = self.root / "annotations/mot_format/sequence_1" / ("cam_{}.txt".format(camera))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("0,0,2,3,4,5,1,10,20,0.5\n", encoding="utf-8")
        original = self._snapshot()
        output = self._new_path("ground-truth")
        summary = export_ground_truth(self.root.parent, export, output, "mot")
        rows = self._csv(output / "ground_truth.csv")
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual({row["track_id"] for row in rows}, {"ground_truth:sequence_1:0"})
        self.assertEqual({row["kind"] for row in rows}, {"ground_truth"})
        self.assertFalse((output / "predictions.csv").exists())
        self.assertEqual(original, self._snapshot())

    def test_normalizer_cannot_write_into_original_scout_root(self):
        export, _ = self._export("posetrack")
        source = self._native("1 7 0 2 3 4 5 10 20 1\n")
        output = self.root / "new-predictions"
        original = self._snapshot()
        with self.assertRaises(ValueError):
            normalize_results(export, source, output, "posetrack10", "sequence")
        self._no_partial_output(output)
        self.assertEqual(original, self._snapshot())


if __name__ == "__main__":
    unittest.main()
