#!/usr/bin/env python3
"""Run the SCOUT adapter in an isolated CPU container with read-only inputs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "scout-mtmc/adapter:0.5"
BASE = "scout-mtmc/posetrack-all:ba727e6"
BASE_ID = "sha256:da99db46dd7d4cfbbd936899c7f3941f065df1725132d62417d0d5194d05bf3d"


def checked_path(value, directory=True):
    path = Path(value).expanduser().resolve()
    if "," in str(path) or "\n" in str(path):
        raise ValueError("Docker mount paths must not contain commas/newlines")
    if directory and not path.is_dir():
        raise ValueError("Input directory does not exist: " + str(path))
    if not directory and not path.is_file():
        raise ValueError("Input file does not exist: " + str(path))
    return path


def validate_adapter_arguments(arguments):
    """Use the stdlib CLI parser before creating any host output directory."""
    original_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "adapters"))
    try:
        from scout_adapter.__main__ import parser as adapter_parser
        adapter_parser().parse_args(arguments)
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = original_bytecode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "test", "inspect", "assess", "export", "normalize", "ground-truth",
                                           "select-window", "yachiyo-pose-input", "yachiyo-pose-output",
                                           "geometry-probe", "project-ground", "model-ground-input", "model-ground"))
    parser.add_argument("--data", help="Extracted SCOUT root, mounted read-only")
    parser.add_argument("--workspace", help="Existing adapter export to normalize, mounted read-only")
    parser.add_argument("--input", help="Native model results, mounted read-only")
    parser.add_argument("--timestamp-map", help="Optional JSON alias override, mounted read-only")
    parser.add_argument("--pose-input", help="Existing YACHIYO pose-input bridge workspace, mounted read-only")
    parser.add_argument("--detections", help="Paired PoseTrack detections, mounted read-only")
    parser.add_argument("--output", help="New output directory (must not already exist)")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    output, output_mount_index = None, None
    if args.action == "build":
        if extra:
            parser.error("build accepts no adapter arguments")
        if not args.dry_run:
            identity = subprocess.check_output(["docker", "image", "inspect", BASE, "--format", "{{.Id}}"], text=True).strip()
            if identity != BASE_ID:
                raise ValueError("Adapter base differs from the validated local image; inspect before rebuilding")
        command = ["docker", "build", "--network=none", "--pull=false", "-f", "docker/adapter/Dockerfile", "-t", IMAGE, "."]
    else:
        command = ["docker", "run", "--rm", "--pull=never", "--runtime=runc", "--network=none",
                   "--user", "{}:{}".format(os.getuid(), os.getgid()), "--cap-drop=ALL",
                   "--security-opt=no-new-privileges", "--cpus=2", "--memory=4g", "--shm-size=256m",
                   "--env", "HOME=/tmp", "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES="]
        inputs, invocation = [], [args.action]
        for value, destination, option, directory in (
                (args.data, "/data/scout", "--root", True),
                (args.workspace, "/workspace", "--export", True),
                (args.input, "/inputs/result", "--input", False),
                (args.timestamp_map, "/inputs/timestamp-map.json", "--timestamp-map", False),
                (args.pose_input, "/pose-input", "--pose-input", True),
                (args.detections, "/inputs/detections", "--detections", False)):
            if value:
                source = checked_path(value, directory)
                inputs.append(source)
                command += ["--mount", "type=bind,src={},dst={},readonly".format(source, destination)]
                invocation += [option, destination]
        if args.action in ("inspect", "assess", "export", "ground-truth", "select-window", "geometry-probe", "project-ground", "model-ground") and not args.data:
            parser.error("this action requires --data")
        if args.action in ("normalize", "ground-truth", "yachiyo-pose-input", "yachiyo-pose-output", "geometry-probe", "project-ground", "model-ground-input", "model-ground") and not args.workspace:
            parser.error("this action requires --workspace")
        if args.action in ("normalize", "yachiyo-pose-input", "yachiyo-pose-output", "project-ground", "model-ground-input", "model-ground") and not args.input:
            parser.error("this action requires --input")
        if args.action == "yachiyo-pose-output" and not args.pose_input:
            parser.error("yachiyo-pose-output requires --pose-input")
        if args.action in ("export", "normalize", "ground-truth", "select-window", "yachiyo-pose-input", "yachiyo-pose-output", "geometry-probe", "project-ground", "model-ground-input", "model-ground") and not args.output:
            parser.error("this action requires --output")
        if args.output:
            output = Path(args.output).expanduser().absolute()
            resolved = output.resolve()
            if output.exists() or output.is_symlink():
                raise ValueError("Output already exists; choose a new directory")
            if "," in str(resolved) or "\n" in str(resolved):
                raise ValueError("Unsupported output mount path")
            for source in inputs:
                if source == resolved or source in resolved.parents or resolved in source.parents:
                    raise ValueError("Input and output must not overlap")
            output = resolved
            output_mount_index = len(command)
            invocation += ["--output", "/outputs/result"]
        if args.action == "test":
            if extra or args.output or inputs:
                parser.error("test takes no data/output/extra arguments")
            command += ["--entrypoint", "python", IMAGE, "-m", "unittest", "discover", "-s", "/opt/scout-adapter/tests", "-v"]
        else:
            # Only the wrapper chooses mounted input/output paths.
            reserved = ("--root", "--export", "--input", "--output", "--timestamp-map", "--pose-input", "--detections")
            if any(item.split("=", 1)[0] in reserved for item in extra):
                parser.error("use wrapper --data/--workspace/--input/--output for paths")
            validate_adapter_arguments(invocation + extra)
            command += [IMAGE] + invocation + extra
    staging = None
    try:
        if output is not None:
            if args.dry_run:
                mounted_output = output.parent / ".adapter-run-DRY-RUN"
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=".adapter-run-", dir=str(output.parent)))
                mounted_output = staging
            # Expose only our fresh staging directory. Mounting output.parent
            # would make sibling read-only inputs writable through an alias.
            command[output_mount_index:output_mount_index] = [
                "--mount", "type=bind,src={},dst=/outputs".format(mounted_output)
            ]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        status = subprocess.call(command, cwd=str(ROOT))
        if status == 0 and staging is not None:
            result = staging / "result"
            if result.is_symlink() or not result.is_dir():
                raise ValueError("Container did not create a complete output directory")
            if output.exists() or output.is_symlink():
                raise ValueError("Output appeared during conversion; refusing overwrite")
            result.rename(output)
        return status
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("Adapter runner error: " + str(error), file=sys.stderr)
        sys.exit(1)
