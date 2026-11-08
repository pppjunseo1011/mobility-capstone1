import argparse
import json
import sys

from .common import load_json, new_output, write_json
from .compatibility import assess_dataset
from .dataset import inspect_dataset
from .export import export_dataset
from .profiles import PROFILES
from .results import export_ground_truth, normalize_results
from .timeline import select_window


def parser():
    main = argparse.ArgumentParser(description="SCOUT input layouts and result normalization; no model inference")
    commands = main.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect", help="Inspect extracted metadata, including metadata-only roots")
    assess = commands.add_parser("assess", help="Compare SCOUT metadata and model input contracts; no inference")
    export = commands.add_parser("export", help="Export a bounded frame clip for one model")
    window = commands.add_parser("select-window", help="Select a continuous multi-camera clip without resampling")
    for command in (inspect, assess, export, window):
        command.add_argument("--root", required=True)
        command.add_argument("--sequence", type=int, default=1)
        command.add_argument("--cameras", nargs="+", help="Source names, e.g. cam_0 cam_1")
        command.add_argument("--timestamp-map", help="Optional JSON mapping source camera to timestamp key")
        command.add_argument("--output", required=(command in (export, window)), help="New output directory; never overwritten")
    window.add_argument("--frames", type=int, default=200)
    window.add_argument("--start-frame", type=int, default=0)
    window.add_argument("--sync-tolerance", type=float, default=0.05)
    window.add_argument("--max-step", type=float, default=0.15)
    export.add_argument("--model", choices=PROFILES, required=True)
    export.add_argument("--start-frame", type=int, default=0)
    export.add_argument("--max-frames", type=int, default=32)
    export.add_argument("--sync-tolerance", type=float, default=0.05)
    assess.add_argument("--sync-tolerance", type=float, default=0.05)
    export.add_argument("--materialize", choices=("copy", "manifest"), default="copy")
    export.add_argument("--videos", action="store_true", help="Explicitly encode lossless RGB MP4 and check decoded frame count")
    export.add_argument("--fps", type=float, help="Required for video; does not replace original timestamps")
    export.add_argument("--video-decoder", choices=("opencv", "ffmpeg"), default="opencv")
    results = commands.add_parser("normalize", help="Convert native results back to source camera/frame IDs")
    results.add_argument("--export", required=True)
    results.add_argument("--input", required=True)
    results.add_argument("--format", required=True)
    results.add_argument("--id-scope", choices=("camera", "sequence"), required=True)
    results.add_argument("--camera", help="Source camera for per-camera MOT results")
    results.add_argument("--output", required=True)
    gt = commands.add_parser("ground-truth", help="Export separate GT; never use it as model detections")
    gt.add_argument("--root", required=True)
    gt.add_argument("--export", required=True)
    gt.add_argument("--format", choices=("mot", "coco"), required=True)
    gt.add_argument("--output", required=True)
    for name in ("geometry-probe", "project-ground"):
        geometry = commands.add_parser(name, help="Raw pixel to ground-mesh geometry; no inference or native H")
        geometry.add_argument("--root", required=True)
        geometry.add_argument("--export", required=True)
        geometry.add_argument("--output", required=True)
        if name == "project-ground":
            geometry.add_argument("--input", required=True, help="CSV: camera,source_frame,point_id,pixel_x,pixel_y")
            geometry.add_argument("--input-kind", choices=("model_output", "synthetic_fixture", "unverified"), default="unverified")
        else:
            geometry.add_argument("--grid-columns", type=int, default=17)
            geometry.add_argument("--grid-rows", type=int, default=11)
    for name in ("model-ground-input", "model-ground"):
        model_ground = commands.add_parser(name, help="Native artifacts to explicit ground-point candidates; no inference")
        model_ground.add_argument("--export", required=True)
        model_ground.add_argument("--input", required=True)
        model_ground.add_argument("--output", required=True)
        model_ground.add_argument("--camera", required=True)
        model_ground.add_argument("--format", required=True, choices=("yachiyo-sct", "uwipl-mot16", "posetrack-pose57"))
        model_ground.add_argument("--policy", required=True, choices=("yachiyo-bbox-bottom", "uwipl-feet", "ankles-only"))
        model_ground.add_argument("--id-scope", required=True, choices=("camera", "sequence", "none"))
        model_ground.add_argument("--input-kind", choices=("model_output", "synthetic_fixture", "unverified"), default="unverified")
        model_ground.add_argument("--detections", help="Paired PoseTrack 7-column detection CSV")
        if name == "model-ground":
            model_ground.add_argument("--root", required=True)
            model_ground.add_argument("--emit-yachiyo-candidate", action="store_true")
    for name in ("yachiyo-pose-input", "yachiyo-pose-output"):
        bridge = commands.add_parser(name, help="Bridge YACHIYO detector/pose frame numbering; no inference")
        bridge.add_argument("--export", required=True)
        bridge.add_argument("--input", required=True)
        bridge.add_argument("--output", required=True)
        bridge.add_argument("--camera", required=True)
        bridge.add_argument("--input-kind", choices=("model_output", "synthetic_fixture", "unverified"), default="unverified")
        if name.endswith("output"):
            bridge.add_argument("--pose-input", required=True)
    return main


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action in ("inspect", "assess", "export", "select-window"):
        overrides = load_json(args.timestamp_map) if args.timestamp_map else None
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("--timestamp-map must contain a JSON object")
        if args.action == "inspect":
            result = inspect_dataset(args.root, args.sequence, args.cameras, overrides)
            if args.output:
                with new_output(args.output, args.root) as staging:
                    write_json(staging / "inspection.json", result)
        elif args.action == "assess":
            result = assess_dataset(args.root, args.sequence, args.cameras, overrides,
                                    args.sync_tolerance, args.output)
        elif args.action == "select-window":
            result = select_window(args.root, args.output, args.sequence, args.cameras,
                                   args.frames, args.start_frame, args.sync_tolerance, args.max_step, overrides)
        else:
            result = export_dataset(args.root, args.output, args.model, args.sequence, args.cameras,
                                    args.start_frame, args.max_frames, overrides, args.sync_tolerance,
                                    args.materialize, args.videos, args.fps, args.video_decoder)
    elif args.action == "normalize":
        result = normalize_results(args.export, args.input, args.output, args.format, args.id_scope, args.camera)
    elif args.action == "ground-truth":
        result = export_ground_truth(args.root, args.export, args.output, args.format)
    elif args.action in ("geometry-probe", "project-ground"):
        from .ground import project_ground
        if args.action == "project-ground":
            result = project_ground(args.root, args.export, args.output, args.input, args.input_kind)
        else:
            result = project_ground(args.root, args.export, args.output, columns=args.grid_columns, rows=args.grid_rows)
    elif args.action in ("model-ground-input", "model-ground"):
        from .model_ground import prepare_model_points, project_model_ground
        options = dict(camera=args.camera, format=args.format, policy=args.policy, id_scope=args.id_scope,
                       input_kind=args.input_kind, detections=args.detections)
        if args.action == "model-ground-input":
            result = prepare_model_points(args.export, args.input, args.output, **options)
        else:
            result = project_model_ground(args.root, args.export, args.input, args.output,
                                          emit_yachiyo_candidate=args.emit_yachiyo_candidate, **options)
    else:
        from .bridges.yachiyo import prepare_pose_csv, restore_pose_json
        if args.action == "yachiyo-pose-input":
            result = prepare_pose_csv(args.export, args.input, args.output, camera=args.camera, input_kind=args.input_kind)
        else:
            result = restore_pose_json(args.export, args.input, args.output, camera=args.camera,
                                       pose_input=args.pose_input, input_kind=args.input_kind)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print("Adapter error: " + str(error), file=sys.stderr)
        sys.exit(1)
