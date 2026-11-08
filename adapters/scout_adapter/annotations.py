"""Read official SCOUT GT without using labels as detector/model inputs.

COCO is streamed one top-level array item at a time. Memory scales with the
selected image/identity index, not with the full annotation file size.
"""

import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

from .common import finite, within


MAX_OBJECT_BYTES = 16 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
_CAMERA = re.compile(r"cam_(0|[1-9][0-9]*)\Z")
_SEQUENCE = re.compile(r"sequence_([1-9][0-9]*)\Z")
_IMAGE = re.compile(r"images/(?:(sequence_[1-9][0-9]*)/)?(cam_(?:0|[1-9][0-9]*))/image_(0|[1-9][0-9]*)\.jpg\Z")


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or value is None:
        raise ValueError(label + " must be an integer")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value() or number < minimum:
            raise ValueError()
        return int(number)
    except (InvalidOperation, ValueError, TypeError, OverflowError) as error:
        raise ValueError(label + " must be an integer >= " + str(minimum)) from error


def _bbox(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(label + " must contain x,y,width,height")
    box = [finite(item, label) for item in value]
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(label + " must have positive width and height")
    return box


def _world(value, label):
    if value is None:
        return None
    if isinstance(value, dict):
        if not all(key in value for key in ("Xw", "Yw", "Zw")):
            raise ValueError(label + " requires Xw,Yw,Zw")
        value = [value[key] for key in ("Xw", "Yw", "Zw")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + " must be an Xw/Yw/Zw object or a three-element array")
    return [finite(item, label) for item in value]


def _selection(manifest):
    sequence = manifest.get("sequence")
    if not isinstance(sequence, str) or not _SEQUENCE.fullmatch(sequence):
        raise ValueError("manifest.sequence must be sequence_N")
    selected = {}
    for camera in manifest.get("cameras", []):
        camera_id = camera.get("id")
        if not isinstance(camera_id, str) or not _CAMERA.fullmatch(camera_id):
            raise ValueError("manifest camera id must be cam_N")
        if camera_id in selected:
            raise ValueError("duplicate manifest camera: " + camera_id)
        if "frames" in camera:
            frames = [_integer(row["source_frame"], "source_frame") for row in camera["frames"]]
        else:
            frames = [_integer(frame, "source_frame") for frame in manifest.get("frame_ids", [])]
        if not frames or len(frames) != len(set(frames)):
            raise ValueError("selected source frames must be nonempty and unique for " + camera_id)
        selected[camera_id] = set(frames)
    if not selected:
        raise ValueError("manifest must select at least one camera")
    return sequence, selected


def _row(camera, frame, track, bbox, score, world, source):
    return {"camera_id": camera, "source_frame": frame, "track_id": track,
            "bbox_xywh": bbox, "score": score, "world_xyz": world,
            "label_source": source, "coverage": "unverified"}


def _mot(root, sequence, selected):
    paths = {camera: within(root, "annotations/mot_format/{}/{}.txt".format(sequence, camera))
             for camera in selected}
    for path in paths.values():
        if not path.is_file():
            raise ValueError("Missing SCOUT MOT annotation file: " + str(path))
    for camera, path in paths.items():
        seen = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                for fields in reader:
                    try:
                        if len(fields) != 10 or any(not value.strip() for value in fields):
                            raise ValueError("SCOUT MOT requires exactly 10 nonempty columns")
                        frame = _integer(fields[0], "frame")
                        track = _integer(fields[1], "track_id")
                        bbox = _bbox(fields[2:6], "bbox")
                        # Label score semantics have not been established. Do not
                        # convert to a detector confidence or discard a sentinel.
                        score = finite(fields[6], "label score")
                        world = _world(fields[7:10], "world_xyz")
                        if frame not in selected[camera]:
                            continue
                        identity = (frame, track)
                        if identity in seen:
                            raise ValueError("duplicate person ID in the same camera/frame")
                        seen.add(identity)
                        yield _row(camera, frame, track, bbox, score, world, "scout-mot")
                    except ValueError as error:
                        raise ValueError("{}:{}: {}".format(path, reader.line_num, error)) from error
            except csv.Error as error:
                raise ValueError("{}:{}: malformed CSV: {}".format(path, reader.line_num, error)) from error


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


class _JSONStream:
    """Small incremental JSON cursor; every decoded value has a byte limit."""
    def __init__(self, handle):
        self.handle = handle
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)

    def _fill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        chunk = self.handle.read(READ_CHUNK_BYTES)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position] in " \t\r\n":
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return None
            self._fill()

    def expect(self, character):
        if self.peek() != character:
            raise ValueError("expected JSON {!r}".format(character))
        self.position += 1

    def value(self):
        if self.peek() is None:
            raise ValueError("unexpected end of JSON")
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError("malformed or truncated JSON: " + error.msg) from error
                if len(self.buffer[self.position:].encode("utf-8")) >= MAX_OBJECT_BYTES:
                    raise ValueError("JSON object/value exceeds 16 MiB limit") from error
                self._fill()
                continue
            if len(self.buffer[self.position:end].encode("utf-8")) > MAX_OBJECT_BYTES:
                raise ValueError("JSON object/value exceeds 16 MiB limit")
            # A number may end exactly at the read boundary, or an exponent may
            # be split across chunks. Wait until its delimiter is available.
            if end == len(self.buffer) and not self.eof:
                self._fill()
                continue
            if end < len(self.buffer) and self.buffer[end] not in " \t\r\n,:]}":
                tail = self.buffer[end:]
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and not self.eof and all(c in "0123456789.eE+-" for c in tail)):
                    if len(self.buffer[self.position:].encode("utf-8")) >= MAX_OBJECT_BYTES:
                        raise ValueError("JSON object/value exceeds 16 MiB limit")
                    self._fill()
                    continue
                raise ValueError("invalid character after JSON value")
            self.position = end
            return value


def _array_items(path, wanted, stop_after=False):
    """Yield one chosen top-level array, parsing other arrays without retaining them."""
    found = False
    with path.open("r", encoding="utf-8-sig") as handle:
        stream = _JSONStream(handle)
        stream.expect("{")
        keys = set()
        if stream.peek() == "}":
            stream.expect("}")
        else:
            while True:
                key = stream.value()
                if not isinstance(key, str) or key in keys:
                    raise ValueError("top-level JSON keys must be unique strings")
                keys.add(key)
                if len(keys) > 1024:
                    raise ValueError("too many top-level JSON keys")
                stream.expect(":")
                if key == wanted:
                    found = True
                    if stream.peek() != "[":
                        raise ValueError("COCO {} must be an array".format(wanted))
                if stream.peek() == "[":
                    stream.expect("[")
                    if stream.peek() != "]":
                        while True:
                            item = stream.value()
                            if key == wanted:
                                yield item
                            if stream.peek() == "]":
                                break
                            stream.expect(",")
                    stream.expect("]")
                    if key == wanted and stop_after:
                        return
                else:
                    stream.value()
                if stream.peek() == "}":
                    stream.expect("}")
                    break
                stream.expect(",")
        if stream.peek() is not None:
            raise ValueError("trailing data after COCO JSON")
    if not found:
        raise ValueError("COCO is missing the {} array".format(wanted))


def _image_id(value, sequence):
    identifier = _integer(value, "image_id")
    text = str(identifier)
    match = _SEQUENCE.fullmatch(sequence)
    if match is None:
        raise ValueError("sequence must be sequence_N")
    prefix = match.group(1)
    if not re.fullmatch(re.escape(prefix) + r"[0-9]{7}", text):
        raise ValueError("SCOUT image_id must match {}: sequence prefix {} + camera(2 digits) + frame(5 digits)".format(sequence, prefix))
    suffix = text[len(prefix):]
    return text, "cam_{}".format(int(suffix[:2])), int(suffix[2:])


def _coco(root, sequence, selected):
    path = within(root, "annotations/coco_format/annotations_coco_{}.json".format(sequence))
    if not path.is_file():
        raise ValueError("Missing SCOUT COCO annotation file: " + str(path))
    expected = {(camera, frame) for camera, frames in selected.items() for frame in frames}
    image_index = {}
    found_pairs = set()
    try:
        for image in _array_items(path, "images", stop_after=True):
            if not isinstance(image, dict):
                raise ValueError("COCO images entries must be objects")
            identifier, id_camera, id_frame = _image_id(image.get("id"), sequence)
            name = image.get("file_name")
            match = _IMAGE.fullmatch(name) if isinstance(name, str) else None
            if match is None:
                raise ValueError("unrecognized SCOUT COCO file_name: {!r}".format(name))
            filename_sequence, camera, frame_text = match.groups()
            frame = int(frame_text)
            if filename_sequence is not None and filename_sequence != sequence:
                raise ValueError("COCO file_name sequence differs from selected sequence")
            if (camera, frame) != (id_camera, id_frame):
                raise ValueError("COCO image_id does not match file_name")
            for dimension in ("width", "height"):
                _integer(image.get(dimension), "image " + dimension, minimum=1)
            if (camera, frame) in expected:
                if identifier in image_index or (camera, frame) in found_pairs:
                    raise ValueError("duplicate selected COCO image")
                image_index[identifier] = (camera, frame)
                found_pairs.add((camera, frame))
        missing = expected - found_pairs
        if missing:
            example = sorted(missing)[:8]
            raise ValueError("Selected images are absent from COCO images; annotation coverage unknown: {} ({} missing)".format(example, len(missing)))
        seen = set()
        for annotation in _array_items(path, "annotations"):
            if not isinstance(annotation, dict):
                raise ValueError("COCO annotations entries must be objects")
            identifier, camera, frame = _image_id(annotation.get("image_id"), sequence)
            # SCOUT's category_id denotes the person identity, not a class ID.
            track = _integer(annotation.get("category_id"), "category_id/person ID")
            bbox = _bbox(annotation.get("bbox"), "bbox")
            world = _world(annotation.get("world"), "world")
            score = None if annotation.get("score") is None else finite(annotation["score"], "label score")
            if identifier not in image_index:
                continue
            identity = (camera, frame, track)
            if identity in seen:
                raise ValueError("duplicate person ID in the same selected COCO camera/frame")
            seen.add(identity)
            yield _row(camera, frame, track, bbox, score, world, "scout-coco")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("{}: {}".format(path, error)) from error


def iter_ground_truth(root: Path, manifest: dict, format: str):
    """Yield only selected camera/source-frame GT; never produce detector inputs.

    MOT lacks an explicit labeled-frame index, so absent frame rows do not prove
    an empty annotated frame. COCO requires every selected image in `images`;
    an indexed image with no annotations is a valid empty result. In both cases
    coverage and label-score semantics remain unverified.
    """
    sequence, selected = _selection(manifest)
    if format in ("mot", "scout-mot"):
        yield from _mot(Path(root), sequence, selected)
    elif format in ("coco", "scout-coco"):
        yield from _coco(Path(root), sequence, selected)
    else:
        raise ValueError("Unsupported SCOUT GT format: {!r}".format(format))
