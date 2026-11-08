"""Choose a traceable contiguous clip; never resample or infer missing times."""
import hashlib
from pathlib import Path

from .common import finite, integer, new_output, write_csv, write_json
from .dataset import build_manifest, metadata, scan_frames

FIRST_EIGHT = ["cam_{}".format(i) for i in range(8)]


def analyze_timeline(timestamps, cameras, image_frames, tolerance=0.05, max_step=0.15):
    tolerance, max_step = finite(tolerance, "tolerance"), finite(max_step, "max step")
    if tolerance < 0 or max_step <= 0 or len(cameras) < 2:
        raise ValueError("Need >=2 cameras, nonnegative tolerance and positive max step")
    names = [row["id"] for row in cameras]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate cameras")
    records, runs = [], []
    beginning, previous = None, None
    for raw in sorted(timestamps, key=int):
        frame = integer(raw, "frame")
        if str(frame) != raw:
            raise ValueError("Noncanonical timestamp frame")
        values, reasons, step_reasons = {}, [], []
        for camera in cameras:
            name, key = camera["id"], camera["timestamp_key"]
            value = timestamps[raw].get(key)
            if value is None:
                reasons.append("missing_timestamp:"+name)
            else:
                values[name] = finite(value, "timestamp")
            if frame not in image_frames[name]:
                reasons.append("missing_image:"+name)
        skew = max(values.values())-min(values.values()) if len(values) == len(names) else None
        if skew is not None and skew > tolerance:
            reasons.append("camera_skew")
        if previous is not None:
            if frame != previous["source_frame"]+1:
                step_reasons.append("frame_gap")
            for name, value in values.items():
                if name not in previous["timestamps"]:
                    step_reasons.append("missing_previous_timestamp:"+name)
                    continue
                delta = value-previous["timestamps"][name]
                if delta == 0:
                    step_reasons.append("duplicate_timestamp:"+name)
                elif delta < 0:
                    step_reasons.append("backward_timestamp:"+name)
                elif delta > max_step:
                    step_reasons.append("timestamp_gap:"+name)
        if reasons or step_reasons:
            if beginning is not None:
                runs.append({"start_frame": beginning, "end_frame": previous["source_frame"],
                             "frame_count": previous["source_frame"]-beginning+1})
                beginning = None
        if not reasons and beginning is None:
            beginning = frame
        record = {"source_frame": frame, "timestamps": values, "max_skew_seconds": skew,
                  "frame_reasons": reasons, "step_from_previous_reasons": step_reasons}
        records.append(record)
        previous = record
    if beginning is not None:
        runs.append({"start_frame": beginning, "end_frame": previous["source_frame"],
                     "frame_count": previous["source_frame"]-beginning+1})
    return records, runs


def select_window(root, output, sequence=1, cameras=None, frames=200, start=0,
                  tolerance=0.05, max_step=0.15, overrides=None):
    frames, start = integer(frames, "frames", 2), integer(start, "start")
    cameras = FIRST_EIGHT if cameras is None else cameras
    root, timestamps, rows = metadata(root, sequence, cameras, overrides)
    image_frames = {c["id"]: set(scan_frames(root / "images" / ("sequence_{}".format(sequence)) / c["id"])) for c in rows}
    records, runs = analyze_timeline(timestamps, rows, image_frames, tolerance, max_step)
    choices = [(max(start, run["start_frame"]), run) for run in runs
               if run["end_frame"]-max(start, run["start_frame"])+1 >= frames]
    selected = choices[0][0] if choices else None
    manifest = build_manifest(root, sequence, cameras, selected, frames, overrides, tolerance) if selected is not None else None
    if manifest is not None and manifest["frame_ids"] != list(range(selected, selected+frames)):
        raise ValueError("Selected images differ from the planned contiguous window")
    summary = {"sequence": "sequence_{}".format(sequence), "cameras": [c["id"] for c in rows],
               "policy": "earliest contiguous same-source-frame window satisfying explicit timing checks",
               "requested_frames": frames, "start_search_frame": start,
               "tolerance_seconds": tolerance, "maximum_adjacent_step_seconds": max_step,
               "timestamp_frame_count": len(records),
               "frames_over_skew_tolerance": sum("camera_skew" in r["frame_reasons"] for r in records),
               "duplicate_camera_steps": sum(sum(x.startswith("duplicate_timestamp:") for x in r["step_from_previous_reasons"]) for r in records),
               "clean_runs": runs, "selected": None, "resampling_performed": False,
               "timestamp_interpolation_performed": False, "model_inference_run": False,
               "benchmark_split_defined": False, "ground_truth_used_to_select_window": False}
    if manifest is not None:
        summary["selected"] = {"start_frame": selected, "end_frame": selected+frames-1, "frame_count": frames,
                               "max_skew_seconds": manifest["sync"]["max_skew_seconds"],
                               "timestamp_spans_seconds": {c["id"]: c["frames"][-1]["timestamp_seconds"]-c["frames"][0]["timestamp_seconds"] for c in manifest["cameras"]}}
    metadata_paths = [root / "timestamps" / ("timestamps_sequence_{}.json".format(sequence))] + [root / "calibrations" / (c["id"]+".json") for c in rows]
    summary["input_metadata_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata_paths}
    with new_output(output, root) as staging:
        write_json(staging / "selection.json", summary)
        if manifest is not None:
            write_json(staging / "manifest.json", manifest)
        write_csv(staging / "timeline.csv", ["source_frame", "max_skew_seconds", "frame_reasons", "step_from_previous_reasons"],
                  ({"source_frame": r["source_frame"], "max_skew_seconds": r["max_skew_seconds"],
                    "frame_reasons": ";".join(r["frame_reasons"]),
                    "step_from_previous_reasons": ";".join(r["step_from_previous_reasons"])} for r in records))
    return summary
