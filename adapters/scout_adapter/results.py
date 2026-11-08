"""Normalize native outputs to original SCOUT frame/camera coordinates."""
from pathlib import Path

from .common import finite, integer, load_json, new_output, separated, write_csv, write_json
from .profiles import PROFILES

FIELDS = ["kind", "model", "sequence", "camera", "frame", "timestamp_seconds", "track_id", "native_track_id",
          "id_scope", "x", "y", "width", "height", "score", "world_x", "world_y", "world_z",
          "position_source", "image_space", "label_source", "label_quality"]


def canonical_row(manifest, camera, frame, native_id, bbox, score, world, world_xy=None,
                  model=None, scope="sequence", kind="prediction", position=None, label_source="", quality=""):
    native_id = integer(native_id, "track id")
    if len(bbox) != 4:
        raise ValueError("bbox must have x,y,width,height")
    bbox = [finite(x, "bbox") for x in bbox]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("bbox width/height must be positive")
    values = [None, None, None]
    if world is not None:
        if len(world) != 3:
            raise ValueError("world_xyz must have 3 coordinates")
        values = [finite(x, "world coordinate") for x in world]
    elif world_xy is not None:
        if len(world_xy) != 2:
            raise ValueError("world_xy must have 2 coordinates")
        values[:2] = [finite(x, "world coordinate") for x in world_xy]
    if score is not None:
        score = finite(score, "score")
    namespace = [model or "ground_truth", manifest["sequence"]]
    if scope == "camera":
        namespace.append(camera["id"])
    track_id = ":".join(namespace + [str(native_id)])
    return dict(zip(FIELDS, [kind, model or "ground_truth", manifest["sequence"], camera["id"],
                            frame["source_frame"], frame["timestamp_seconds"], track_id, native_id, scope,
                            *bbox, score, *values, position, manifest["image_space"], label_source, quality]))


def normalize_results(export, source, output, format, scope, camera=None):
    if scope not in ("camera", "sequence"):
        raise ValueError("Explicit --id-scope camera or sequence is required")
    export, source = Path(export).resolve(), Path(source).resolve()
    manifest, profile = load_json(export / "manifest.json"), load_json(export / "profile.json")
    separated(export, output)
    separated(source, output)
    if manifest.get("source_root"):
        separated(manifest["source_root"], output)
    if manifest.get("schema_version") != 1 or profile["name"] != manifest["model"]:
        raise ValueError("Unsupported or mismatched export manifest")
    model = manifest["model"]
    by_id = {int(item["model_camera_id"]): item for item in manifest["cameras"]}
    by_name = {item["id"]: item for item in manifest["cameras"]}
    if camera is not None and camera not in by_name:
        raise ValueError("--camera must name a source camera in this export")
    rows, seen = [], set()
    for native in PROFILES[model].read_results(source, format=format):
        if native["camera_id"] is None:
            if camera is None:
                raise ValueError("This per-camera result format requires --camera cam_N")
            selected = by_name[camera]
        else:
            if native["camera_id"] not in by_id:
                raise ValueError("Result camera is absent from the frame map: " + str(native["camera_id"]))
            selected = by_id[native["camera_id"]]
            if camera is not None and selected["id"] != camera:
                raise ValueError("--camera conflicts with camera IDs inside native results")
        index = native["frame"] - profile["frame_base"]
        if not 0 <= index < len(selected["frames"]):
            raise ValueError("Result frame is outside the exported clip: " + str(native["frame"]))
        frame = selected["frames"][index]
        key = (selected["id"], frame["source_frame"], native["track_id"])
        if key in seen:
            raise ValueError("Duplicate track within camera/frame: " + str(key))
        seen.add(key)
        rows.append(canonical_row(manifest, selected, frame, native["track_id"], native["bbox_xywh"],
                                  native.get("score"), native.get("world_xyz"), native.get("world_xy"),
                                  model=model, scope=scope, position=native.get("position_source")))
    rows.sort(key=lambda r: (r["camera"], r["frame"], r["native_track_id"]))
    summary = {"schema_version": 1, "kind": "predictions", "model": model, "source_format": format,
               "sequence": manifest["sequence"], "row_count": len(rows), "id_scope": scope,
               "source_file": str(source), "export_manifest": str(export / "manifest.json"),
               "evaluation_performed": False,
               "note": "IDs retain source sequence scope; camera-local IDs are not cross-camera identities. Empty score/z means unavailable."}
    with new_output(output) as staging:
        write_csv(staging / "predictions.csv", FIELDS, rows)
        write_json(staging / "metadata.json", summary)
    return summary


def export_ground_truth(root, export, output, format):
    from .annotations import iter_ground_truth
    from .dataset import dataset_root
    root = dataset_root(root)
    manifest = load_json(Path(export) / "manifest.json")
    separated(root, output)
    separated(export, output)
    cameras = {row["id"]: row for row in manifest["cameras"]}
    frames = {camera: {f["source_frame"]: f for f in info["frames"]} for camera, info in cameras.items()}
    rows, seen = [], set()
    for item in iter_ground_truth(Path(root), manifest, format):
        key = (item["camera_id"], item["source_frame"], item["track_id"])
        if key in seen:
            raise ValueError("Duplicate GT identity within camera/frame: " + str(key))
        seen.add(key)
        camera, frame = cameras[item["camera_id"]], frames[item["camera_id"]][item["source_frame"]]
        rows.append(canonical_row(manifest, camera, frame, item["track_id"], item["bbox_xywh"],
                                  item.get("score"), item.get("world_xyz"), kind="ground_truth",
                                  label_source=item["label_source"], quality=item.get("coverage", "unverified")))
    rows.sort(key=lambda row: (row["camera"], row["frame"], row["native_track_id"]))
    summary = {"schema_version": 1, "kind": "ground_truth", "row_count": len(rows), "format": format,
               "label_quality": "unverified; manual annotations versus pseudo-labels require separate audit",
               "not_for_model_input": True, "evaluation_performed": False}
    with new_output(output) as staging:
        write_csv(staging / "ground_truth.csv", FIELDS, rows)
        write_json(staging / "metadata.json", summary)
    return summary
