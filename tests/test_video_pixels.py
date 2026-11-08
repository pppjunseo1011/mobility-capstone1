"""Lossless OpenCV JPEG-to-video checks on tiny synthetic images only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

from scout_adapter import export as video_export

try:
    import cv2
except ImportError:
    cv2 = None


@unittest.skipUnless(cv2 is not None and shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "Run in the adapter image with OpenCV and FFmpeg")
class VideoPixelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scout-video-pixels-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.frames = self.root / "frames"
        self.frames.mkdir()
        self.width, self.height = 64, 48
        self.pattern = "{frame:06d}.jpg"
        # High-frequency chroma makes a JPEG decoder mismatch visible; solid
        # colors would miss the original FFmpeg/OpenCV color-conversion bug.
        for frame in range(3):
            picture = Image.new("RGB", (self.width, self.height))
            picture.putdata([((x * 71 + y * 13 + frame * 19) % 256,
                              (x * 11 + y * 97 + frame * 67) % 256,
                              (x * 47 + y * 29 + frame * 113) % 256)
                             for y in range(self.height) for x in range(self.width)])
            picture.save(self.frames / self.pattern.format(frame=frame),
                         format="JPEG", quality=73, subsampling=2)
        self.output = self.root / "video.mp4"

    def _encode(self, **kwargs):
        return video_export.make_video(self.frames, self.pattern, 0, 3, 10,
                                       self.output, self.width, self.height, **kwargs)

    def _snapshot(self):
        return {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in self.frames.iterdir()}

    def _assert_clean_failure(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".video.mp4-*")), [])

    def _ffmpeg_decode(self, source, *, jpeg_sequence=False):
        command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1"]
        if jpeg_sequence:
            command += ["-framerate", "10", "-start_number", "0"]
        command += ["-i", str(source), "-frames:v", "3", "-threads", "1",
                    "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        return subprocess.run(command, capture_output=True, check=True, timeout=30).stdout

    def test_default_video_equals_direct_opencv_pixels_and_order(self):
        before = self._snapshot()
        originals = [cv2.imread(str(self.frames / self.pattern.format(frame=index)), cv2.IMREAD_COLOR)
                     for index in range(3)]
        expected = b"".join(frame.tobytes(order="C") for frame in originals)
        self.assertEqual(len({hashlib.sha256(frame.tobytes()).hexdigest() for frame in originals}), 3)
        prior_threads = cv2.getNumThreads()
        info = self._encode()
        self.assertEqual(cv2.getNumThreads(), prior_threads)
        self.assertEqual(info["video_decoder"], "opencv")
        self.assertEqual(info["opencv_version"], cv2.__version__)
        self.assertEqual(info["source_bgr_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(self._ffmpeg_decode(self.output), expected)
        capture = cv2.VideoCapture(str(self.output), cv2.CAP_FFMPEG)
        try:
            self.assertTrue(capture.isOpened())
            for original in originals:
                ok, decoded = capture.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape, original.shape)
                self.assertEqual(decoded.tobytes(), original.tobytes())
            self.assertFalse(capture.read()[0])
        finally:
            capture.release()
        self.assertEqual(self._snapshot(), before)
        self.assertEqual(list(self.root.glob(".video.mp4-*")), [])

    def test_explicit_ffmpeg_decoder_retains_previous_pixel_reference(self):
        before = self._snapshot()
        info = self._encode(video_decoder="ffmpeg")
        self.assertEqual(info["video_decoder"], "ffmpeg")
        self.assertIsNone(info["opencv_version"])
        self.assertEqual(self._ffmpeg_decode(self.output),
                         self._ffmpeg_decode(self.frames / "%06d.jpg", jpeg_sequence=True))
        self.assertEqual(self._snapshot(), before)

    def test_corrupt_second_jpeg_aborts_without_partial_video(self):
        (self.frames / self.pattern.format(frame=1)).write_bytes(b"not a JPEG")
        before = self._snapshot()
        with self.assertRaisesRegex(ValueError, "cannot decode JPEG"):
            self._encode()
        self._assert_clean_failure()
        self.assertEqual(self._snapshot(), before)

    def test_wrong_frame_dimensions_abort_without_partial_video(self):
        Image.new("RGB", (32, 24), (30, 60, 90)).save(self.frames / self.pattern.format(frame=1))
        with self.assertRaisesRegex(ValueError, "dimensions/dtype"):
            self._encode()
        self._assert_clean_failure()

    def test_existing_video_is_never_overwritten(self):
        self.output.write_bytes(b"existing user output")
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self._encode()
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".video.mp4-*")), [])

    def test_encoder_early_exit_is_reaped_and_partial_file_removed(self):
        actual_popen = subprocess.Popen
        children = []

        def exited_encoder(command, **kwargs):
            child = actual_popen([sys.executable, "-c", "import sys; sys.exit(9)"], **kwargs)
            children.append(child)
            return child

        with mock.patch.object(video_export.subprocess, "Popen", side_effect=exited_encoder):
            with self.assertRaisesRegex(ValueError, "FFmpeg"):
                self._encode()
        self.assertTrue(children)
        self.assertTrue(all(child.poll() is not None for child in children))
        self._assert_clean_failure()

    def test_probe_failure_does_not_publish_encoded_video(self):
        error = subprocess.CalledProcessError(1, ["ffprobe"])
        with mock.patch.object(video_export.subprocess, "run", side_effect=error):
            with self.assertRaises(subprocess.CalledProcessError):
                self._encode()
        self._assert_clean_failure()

    def test_stalled_encoder_pipe_has_a_deadline_and_child_is_reaped(self):
        actual_popen = subprocess.Popen
        children = []

        def tracked_child(command, **kwargs):
            child = actual_popen(command, **kwargs)
            children.append(child)
            return child

        prior_threads = cv2.getNumThreads()
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        with mock.patch.object(video_export.subprocess, "Popen", side_effect=tracked_child):
            with self.assertRaises(subprocess.TimeoutExpired):
                video_export._opencv_to_ffmpeg(self.frames, self.pattern, 0, 3,
                                                self.width, self.height, command, timeout=0.1)
        self.assertTrue(children)
        self.assertTrue(all(child.poll() is not None for child in children))
        self.assertEqual(cv2.getNumThreads(), prior_threads)

    def test_export_records_decoder_and_preserves_copied_jpeg_bytes(self):
        source = self.root / "dataset"
        calibration = source / "calibrations" / "cam_0.json"
        calibration.parent.mkdir(parents=True)
        calibration.write_text(json.dumps({
            "K": [[200, 0, 32], [0, 200, 24], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T": [[0], [0], [1]],
            "dist": [[0.1, 0, 0, 0, 0]], "view_id": "c10_0"}), encoding="utf-8")
        timestamps = source / "timestamps" / "timestamps_sequence_1.json"
        timestamps.parent.mkdir()
        timestamps.write_text(json.dumps({str(index): {"c10": 100 + index / 10}
                                          for index in range(3)}), encoding="utf-8")
        images = source / "images" / "sequence_1" / "cam_0"
        images.mkdir(parents=True)
        for index in range(3):
            shutil.copyfile(self.frames / self.pattern.format(frame=index), images / ("image_{}.jpg".format(index)))
        destination = self.root / "export"
        report = video_export.export_dataset(source, destination, "posetrack", cameras=["cam_0"],
                                             maximum=3, videos=True, fps=10)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        profile = json.loads((destination / "profile.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["video_decoder"], "opencv")
        self.assertEqual(report["videos"]["cam_0"]["video_decoder"], "opencv")
        self.assertFalse(report["model_inference_run"])
        camera = profile["cameras"][0]
        for index in range(3):
            copied = destination / camera["frame_dirs"][0] / camera["frame_pattern"].format(frame=index)
            self.assertEqual(copied.read_bytes(), (images / ("image_{}.jpg".format(index))).read_bytes())

    def test_unknown_decoder_is_rejected_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "video_decoder"):
            self._encode(video_decoder="automatic")
        self._assert_clean_failure()


if __name__ == "__main__":
    unittest.main()
