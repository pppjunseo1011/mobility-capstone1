"""Synthetic annotation tests: no SCOUT downloads, model execution or external libs."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
annotations = importlib.import_module("scout_adapter.annotations")


class GroundTruthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scout-gt-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {"sequence": "sequence_1", "sequence_number": 1,
                         "frame_ids": [0, 81], "image_space": "raw",
                         "cameras": [{"id": "cam_3", "frames": [
                             {"index": 0, "source_frame": 0},
                             {"index": 1, "source_frame": 81}]}]}

    def mot(self, text, camera="cam_3"):
        path = self.root / "annotations/mot_format/sequence_1" / (camera + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def image(self, camera=3, frame=0, sequence=1):
        return {"id": "{}{:02d}{:05d}".format(sequence, camera, frame),
                "file_name": "images/cam_{}/image_{}.jpg".format(camera, frame),
                "width": 1920, "height": 1080, "date_captured": 1000000000000000}

    def annotation(self, camera=3, frame=0, person=0, world=None, sequence=1):
        return {"id": "10300000000000", "image_id": "{}{:02d}{:05d}".format(sequence, camera, frame),
                "category_id": person, "bbox": [849, 363, 6, 12],
                "world": [33.65, -13.62, .174] if world is None else world}

    def coco(self, images=None, labels=None, annotations_first=False):
        path = self.root / "annotations/coco_format" / ("annotations_coco_" + self.manifest["sequence"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        images = [self.image(), self.image(frame=81)] if images is None else images
        labels = [self.annotation()] if labels is None else labels
        if annotations_first:
            document = {"annotations": labels, "info": {"note": "SCOUT 합성"}, "images": images}
        else:
            document = {"images": images, "annotations": labels, "categories": [{"id": 0}]}
        path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return path

    def rows(self, format):
        return list(annotations.iter_ground_truth(self.root, self.manifest, format))

    def test_mot_preserves_source_frame_identity_world_and_unverified_score(self):
        self.mot("0,0,1,2,3,4,1,10,-2,.5\n1,7,1,2,3,4,1,0,0,0\n81,9007199254740993,5,6,7,8,-1,-1,-1,-1\n")
        rows = self.rows("mot")
        self.assertEqual([r["source_frame"] for r in rows], [0, 81])
        self.assertEqual(rows[0], {"camera_id": "cam_3", "source_frame": 0, "track_id": 0,
                                  "bbox_xywh": [1., 2., 3., 4.], "score": 1.,
                                  "world_xyz": [10., -2., .5], "label_source": "scout-mot", "coverage": "unverified"})
        self.assertEqual(rows[1]["track_id"], 9007199254740993)
        self.assertEqual(rows[1]["score"], -1.)
        self.assertEqual(rows[1]["world_xyz"], [-1., -1., -1.])

    def test_mot_only_selected_cameras_and_empty_file(self):
        self.mot("")
        self.mot("0,1,1,2,3,4,1,0,0,0\n", "cam_4")
        self.assertEqual(self.rows("mot"), [])

    def test_missing_annotation_files_fail(self):
        for format in ("mot", "coco"):
            with self.subTest(format=format), self.assertRaisesRegex(ValueError, "Missing SCOUT"):
                self.rows(format)

    def test_mot_all_selected_files_checked_before_rows(self):
        self.mot("0,1,1,2,3,4,1,0,0,0\n")
        self.manifest["cameras"].append({"id": "cam_4", "frames": [{"source_frame": 0}]})
        with self.assertRaisesRegex(ValueError, "cam_4"):
            next(annotations.iter_ground_truth(self.root, self.manifest, "mot"))

    def test_mot_malformed_rows_strict(self):
        cases = ["0,1,1,2,3,4,1,0,0\n", "0,1.5,1,2,3,4,1,0,0,0\n",
                 "-1,1,1,2,3,4,1,0,0,0\n", "0,1,1,2,0,4,1,0,0,0\n",
                 "0,1,nan,2,3,4,1,0,0,0\n", "0,1,1,2,3,4,inf,0,0,0\n", "\n",
                 '"0,1,1,2,3,4,1,0,0,0\n', "0,1,1,2,3,4,1,0,0,0\n0,1,1,2,3,4,1,0,0,0\n"]
        for text in cases:
            with self.subTest(text=text):
                path = self.mot(text)
                with self.assertRaisesRegex(ValueError, str(path)):
                    self.rows("mot")

    def test_coco_category_is_person_id_zero_valid_and_no_timestamp_guess(self):
        self.coco(labels=[self.annotation(person=0), self.annotation(frame=81, person=17)])
        rows = self.rows("coco")
        self.assertEqual([r["track_id"] for r in rows], [0, 17])
        self.assertEqual([r["source_frame"] for r in rows], [0, 81])
        self.assertEqual(rows[0]["world_xyz"], [33.65, -13.62, .174])
        self.assertIsNone(rows[0]["score"])
        self.assertNotIn("timestamp", rows[0])
        self.assertEqual(rows[0]["coverage"], "unverified")
        self.assertEqual(rows[0]["label_source"], "scout-coco")

    def test_coco_sequence_two_prefix_and_mismatches(self):
        self.manifest["sequence"] = "sequence_2"
        self.manifest["sequence_number"] = 2
        images = [self.image(sequence=2), self.image(frame=81, sequence=2)]
        self.assertEqual(images[0]["id"], "20300000")
        self.assertEqual(images[0]["file_name"], "images/cam_3/image_0.jpg")
        self.coco(images=images, labels=[self.annotation(sequence=2)])
        rows = self.rows("coco")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["camera_id"], rows[0]["source_frame"], rows[0]["track_id"]), ("cam_3", 0, 0))
        self.coco(images=[self.image(), self.image(frame=81, sequence=2)], labels=[])
        with self.assertRaisesRegex(ValueError, "image_id must match sequence_2"):
            self.rows("coco")
        self.coco(images=images, labels=[self.annotation(sequence=1)])
        with self.assertRaisesRegex(ValueError, "image_id must match sequence_2"):
            self.rows("coco")
        self.manifest["sequence"] = "sequence_1"
        self.manifest["sequence_number"] = 1
        self.coco(images=[self.image(sequence=2), self.image(frame=81)], labels=[])
        with self.assertRaisesRegex(ValueError, "image_id must match sequence_1"):
            self.rows("coco")

    def test_coco_world_dictionary_and_null(self):
        a = self.annotation(world={"Xw": 1, "Yw": 2, "Zw": 3})
        b = self.annotation(frame=81, person=2)
        b.pop("world")
        self.coco(labels=[a, b])
        rows = self.rows("coco")
        self.assertEqual(rows[0]["world_xyz"], [1., 2., 3.])
        self.assertIsNone(rows[1]["world_xyz"])

    def test_coco_unselected_cameras_and_frames_filtered(self):
        self.coco(images=[self.image(), self.image(frame=81), self.image(frame=1), self.image(camera=4)],
                  labels=[self.annotation(), self.annotation(frame=1), self.annotation(camera=4)])
        self.assertEqual(len(self.rows("coco")), 1)

    def test_coco_indexed_empty_frames_valid_but_missing_image_fails(self):
        self.coco(labels=[])
        self.assertEqual(self.rows("coco"), [])
        self.coco(images=[self.image()], labels=[])
        with self.assertRaisesRegex(ValueError, "coverage unknown"):
            self.rows("coco")

    def test_coco_annotations_before_images_and_tiny_read_chunks(self):
        self.coco(annotations_first=True)
        with mock.patch.object(annotations, "READ_CHUNK_BYTES", 7), mock.patch.object(json, "load", side_effect=AssertionError("must stream")):
            rows = self.rows("coco")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["track_id"], 0)

    def test_coco_invalid_selected_image_relationship_and_duplicate(self):
        bad = self.image()
        bad["id"] = "10400000"
        self.coco(images=[bad, self.image(frame=81)])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.rows("coco")
        self.coco(images=[self.image(), self.image(), self.image(frame=81)])
        with self.assertRaisesRegex(ValueError, "duplicate selected"):
            self.rows("coco")

    def test_coco_malformed_annotations(self):
        changes = [{"category_id": -1}, {"category_id": True}, {"bbox": [1, 2, -3, 4]},
                   {"bbox": [1, 2, 3]}, {"world": [1, 2]}, {"world": {"Xw": 1}},
                   {"world": [1, float("inf"), 3]}, {"score": float("nan")}, {"image_id": "bad"}]
        for change in changes:
            with self.subTest(change=change):
                a = self.annotation()
                a.update(change)
                self.coco(labels=[a])
                with self.assertRaises(ValueError):
                    self.rows("coco")
        self.coco(labels=[self.annotation(), self.annotation()])
        with self.assertRaisesRegex(ValueError, "duplicate person"):
            self.rows("coco")

    def test_coco_truncation_trailing_data_and_missing_arrays(self):
        for raw in ['{"images":[', '{"images":[],"images":[]}', '{"images":{},"annotations":[]}',
                    '{"annotations":[]}', '{"images":[],"annotations":[NaN]}']:
            path = self.coco()
            path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.rows("coco")
        path = self.coco(labels=[])
        path.write_text(path.read_text() + "junk")
        with self.assertRaisesRegex(ValueError, "trailing"):
            self.rows("coco")
        path = self.coco()
        path.write_text(json.dumps({"images": [self.image(), self.image(frame=81)]}))
        with self.assertRaisesRegex(ValueError, "missing the annotations"):
            self.rows("coco")

    def test_object_limit_and_long_stream_of_small_objects(self):
        labels = [self.annotation(frame=1, person=i) for i in range(1000)] + [self.annotation()]
        self.coco(images=[self.image(), self.image(frame=81), self.image(frame=1)], labels=labels)
        with mock.patch.object(annotations, "MAX_OBJECT_BYTES", 1024), mock.patch.object(annotations, "READ_CHUNK_BYTES", 128):
            self.assertEqual(len(self.rows("coco")), 1)
        large = self.annotation()
        large["unused"] = "x" * 1500
        self.coco(labels=[large])
        with mock.patch.object(annotations, "MAX_OBJECT_BYTES", 1024), mock.patch.object(annotations, "READ_CHUNK_BYTES", 128):
            with self.assertRaisesRegex(ValueError, "limit"):
                self.rows("coco")

    def test_bad_manifest_and_unknown_format(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.rows("other")
        self.manifest["sequence"] = "../sequence_1"
        with self.assertRaisesRegex(ValueError, "sequence"):
            self.rows("mot")


if __name__ == "__main__":
    unittest.main()
