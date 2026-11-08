#!/usr/bin/env python3
"""Actual SCOUT geometry + explicitly synthetic native artifacts; no inference."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "scout-mtmc/adapter:0.5"
IMAGE_ID = "sha256:bfd6a2d84792f82e1a23cfc0758306150b4dd3bea9a6bf096e3c188706bce061"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def inside(args):
    import numpy as np
    from scout_adapter.geometry import GroundMesh, CameraGeometry, read_ground_ply
    from scout_adapter.common import write_json

    data, exports, output = Path(args.data), Path(args.exports), Path(args.output)
    output.mkdir()
    fixture_root = output/"synthetic_fixtures"
    fixture_root.mkdir()
    exports_by_model = {model: exports/"sequence1-{}".format(model) for model in ("yachiyo", "uwipl", "posetrack")}
    manifest = json.loads((exports_by_model["posetrack"]/"manifest.json").read_text())
    expected_names = ["cam_{}".format(n) for n in range(8)]
    if manifest["sequence"] != "sequence_1" or [c["id"] for c in manifest["cameras"]] != expected_names:
        raise ValueError("Validator is restricted to sequence_1 cam_0..cam_7")
    protected = [data/"meshes/mesh_ground_only.ply", data/"timestamps/timestamps_sequence_1.json"]
    protected += [data/"calibrations"/(name+".json") for name in expected_names]
    protected += [p/file for p in exports_by_model.values() for file in ("manifest.json", "profile.json")]
    protected += [data/f["source_path"] for c in manifest["cameras"] for f in c["frames"][:2]]
    before = {str(p): {"sha256": digest(p), "mtime_ns": p.stat().st_mtime_ns} for p in protected}
    mesh = GroundMesh(*read_ground_ply(data/"meshes/mesh_ground_only.ply"))
    # Read only the original UWIPL point-selection utility; no models/weights.
    native_path = Path(args.uwipl_utility)
    spec = importlib.util.spec_from_file_location("native_uwipl_file_utils", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    cases = []
    for camera in manifest["cameras"]:
        name = camera["id"]
        if len(camera["frames"]) < 2:
            raise ValueError("Need two source frames")
        geometry = CameraGeometry(camera["calibration"], camera["width"], camera["height"])
        chosen = None
        # Pure geometric fixture selection, not image recognition or person GT.
        for v in np.linspace(camera["height"]*.55, camera["height"]*.9, 8).astype(int):
            for u in np.linspace(camera["width"]*.25, camera["width"]*.75, 9).astype(int):
                result = geometry.ground_point([int(u), int(v)], mesh)
                if result["status"] == "hit":
                    chosen = (int(u), int(v), result["world"])
                    break
            if chosen:
                break
        if chosen is None:
            raise ValueError("No interior fixture ground point for " + name)
        u, v, expected_world = chosen
        fixtures = fixture_root/name
        fixtures.mkdir()
        yachiyo = {str(index).zfill(8): {"Frame": index+1, "OfflineID": 0,
                    "Coordinate": {"x1": u-10, "y1": v-40, "x2": u+10, "y2": v},
                    "WorldCoordinate": None, "NpyPath": "synthetic_fixture_not_loaded.npy"} for index in range(2)}
        y_path = fixtures/"synthetic_yachiyo_sct.json"
        write_json(y_path, yachiyo)
        uw_rows = [[index, 0, u-10, v-38, 20, 40, .9, -1, -1, -1,
                    u-3, v+(6 if index else 0), .5 if index else .9,
                    u+3, v+(6 if index else 0), .9] for index in range(2)]
        u_path = fixtures/"synthetic_uwipl_mot16.txt"
        u_path.write_text("".join(",".join(map(str, row))+"\n" for row in uw_rows))
        pose_rows, det_rows = [], []
        for index in range(2):
            det = [index, 1, u-10, v-40, u+10, v, .91234567]
            det_rows.append(det)
            keypoints = [[u, v-20, .9] for _ in range(17)]
            keypoints[15] = [u-3, v, .7 if index else .9]
            keypoints[16] = [u+3, v, .9]
            pose_rows.append([index, *np.asarray(det[2:], dtype=np.float32).tolist(),
                              *np.asarray(keypoints).ravel().tolist()])
        p_path, d_path = fixtures/"synthetic_posetrack_pose57.txt", fixtures/"synthetic_posetrack_detection7.txt"
        p_path.write_text("".join(" ".join(map(str, row))+"\n" for row in pose_rows))
        d_path.write_text("".join(",".join(map(str, row))+"\n" for row in det_rows))
        write_json(fixtures/"provenance.json", {"input_kind": "synthetic_fixture", "model_inference_run": False,
                  "ground_truth_used": False, "source_frames": [f["source_frame"] for f in camera["frames"][:2]],
                  "selected_grid_pixel": [u,v], "note": "Invented boxes/keypoints/ID 0 only for format tests. Not actual people or model predictions."})
        native_tracks = native.load_tracking(str(u_path), enable_feet=True)
        for frame in (0, 1):
            np.testing.assert_allclose(native_tracks[0][frame][0][4:6], [u, v], atol=0, rtol=0)
        for model, path, format_name, policy, scope in (
                ("yachiyo", y_path, "yachiyo-sct", "yachiyo-bbox-bottom", "camera"),
                ("uwipl", u_path, "uwipl-mot16", "uwipl-feet", "camera"),
                ("posetrack", p_path, "posetrack-pose57", "ankles-only", "none")):
            destination = output/(model+"-"+name)
            command = [sys.executable, "-m", "scout_adapter", "model-ground", "--root", str(data),
                       "--export", str(exports_by_model[model]), "--input", str(path), "--output", str(destination),
                       "--camera", name, "--format", format_name, "--policy", policy, "--id-scope", scope,
                       "--input-kind", "synthetic_fixture"]
            if model == "posetrack":
                command += ["--detections", str(d_path)]
            if model == "yachiyo":
                command += ["--emit-yachiyo-candidate"]
            with (output/(model+"-"+name+".log")).open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((destination/"model-ground.json").read_text())
            expected_counts = {"hit": 1, "low_ankle_confidence": 1} if model == "posetrack" else {"hit": 2}
            assert report["row_status_counts"] == expected_counts
            assert report["bridge"]["input_kind"] == "synthetic_fixture" and not report["model_inference_run"]
            rows = read_csv(destination/"ground_candidates.csv")
            assert len(rows) == 2
            assert [int(r["source_frame"]) for r in rows] == [f["source_frame"] for f in camera["frames"][:2]]
            for row in rows:
                if row["status"] == "hit":
                    np.testing.assert_allclose([float(row[k]) for k in ("world_x", "world_y", "world_z")], expected_world, atol=1e-8, rtol=0)
                else:
                    assert row["world_x"] == row["world_y"] == row["world_z"] == ""
            if model == "yachiyo":
                candidate = json.loads((destination/report["native_candidate"]).read_text())
                for key, row in candidate.items():
                    world = row.pop("WorldCoordinate")
                    original = {k: val for k, val in yachiyo[key].items() if k != "WorldCoordinate"}
                    assert row == original
                    np.testing.assert_allclose([world["x"], world["y"]], expected_world[:2], atol=1e-8, rtol=0)
            cases.append({"model":model, "camera":name, "status":"PASS", "row_status_counts":expected_counts,
                          "report":str(destination.relative_to(output)/"model-ground.json")})
            print(model, name, "PASS", expected_counts, flush=True)
    for path, previous in before.items():
        assert digest(path) == previous["sha256"] and Path(path).stat().st_mtime_ns == previous["mtime_ns"]
    result = {"kind":"real_geometry_synthetic_artifact_validation", "image":IMAGE, "image_id":IMAGE_ID,
              "case_count":len(cases), "passed":len(cases), "cases":cases, "native_uwipl_point_policy_matches":8,
              "uwipl_utility_sha256":digest(native_path), "source_files_unchanged":before,
              "fixture_kind":"synthetic_fixture", "ground_truth_used":False, "model_inference_run":False,
              "actual_model_artifacts_validated":False, "native_tracker_executed":False,
              "note":"Actual SCOUT cameras, mesh and source-frame maps; synthetic boxes/keypoints/IDs only. Not a tracking or localization accuracy test."}
    write_json(output/"validation.json", result)
    print(json.dumps({"cases_passed":len(cases), "synthetic_artifacts":True, "source_files_preserved":len(before)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "inside"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--exports", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uwipl-utility", default=str(ROOT/"repos/uwipl/STCRA/file_utils.py"))
    args = parser.parse_args()
    if args.action == "inside":
        inside(args)
        return 0
    sources = [Path(args.data).resolve(), Path(args.exports).resolve(), Path(args.uwipl_utility).resolve(), Path(__file__).resolve()]
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a new directory")
    output = output.resolve()
    for path in sources+[output]:
        if "," in str(path) or "\n" in str(path):
            raise ValueError("Invalid mount path")
    if not all(p.exists() for p in sources):
        raise ValueError("Missing source")
    if any(p == output or p in output.parents or output in p.parents for p in sources):
        raise ValueError("Inputs and output must not overlap")
    actual = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if actual != IMAGE_ID:
        raise ValueError("Image differs from validated v0.5 build")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-ground-validation-", dir=output.parent))
    try:
        command = ["docker", "run", "--rm", "--pull=never", "--runtime=runc", "--network=none",
                   "--user", "{}:{}".format(os.getuid(), os.getgid()), "--cpus=2", "--memory=4g",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "-e", "HOME=/tmp",
                   "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES="]
        for path, target in zip(sources, ("/data/scout", "/exports", "/native/uwipl.py", "/checks/validate.py")):
            command += ["--mount", "type=bind,src={},dst={},readonly".format(path, target)]
        command += ["--mount", "type=bind,src={},dst=/outputs".format(staging), "--entrypoint", "python", IMAGE_ID,
                    "/checks/validate.py", "inside", "--data", "/data/scout", "--exports", "/exports",
                    "--output", "/outputs/result", "--uwipl-utility", "/native/uwipl.py"]
        print(json.dumps({"image":IMAGE, "image_id":actual, "command":command}), flush=True)
        subprocess.run(command, check=True)
        if output.exists() or output.is_symlink():
            raise ValueError("Output appeared during validation")
        (staging/"result").rename(output)
    finally:
        shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    sys.exit(main())
