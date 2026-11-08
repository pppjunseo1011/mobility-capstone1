#!/usr/bin/env python3
"""Dependency-free Docker CLI; does not run any model pipeline implicitly."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
ENVS = json.loads((ROOT / "environments.json").read_text())
SOURCES = json.loads((ROOT / "sources.lock.json").read_text())["repositories"]


def image_name(model, stage):
    return "scout-mtmc/{}-{}:{}".format(model, stage, SOURCES[model]["commit"][:7])


def checked_source(model):
    repo = ROOT / "repos" / model
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCES[model]["commit"]:
        raise ValueError("Source commit differs from sources.lock.json: " + model)
    changes = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if changes:
        raise ValueError("Source changes/untracked files found; review/update provenance before building: " + model)


def host_directory(value, allow_missing=False):
    path = Path(value).expanduser().resolve()
    if not path.is_dir() and (not allow_missing or path.exists()):
        raise ValueError("Not a directory: " + str(path))
    if "," in str(path) or "\n" in str(path):
        raise ValueError("Docker mount paths must not contain commas/newlines")
    return path


def execute(command, dry_run=False, log=None):
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    if log is None:
        return subprocess.call(command, cwd=str(ROOT))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        output.write("$ " + shlex.join(command) + "\n")
        output.flush()
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
        result = process.wait()
        output.write("\nexit_code={}\n".format(result))
    print("Log: " + str(log))
    return result


def main():
    argv = sys.argv[1:]
    override = []
    if "--" in argv:
        divider = argv.index("--")
        override, argv = argv[divider + 1:], argv[:divider]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    build = sub.add_parser("build")
    build.add_argument("model", choices=list(ENVS) + ["all"])
    build.add_argument("stage", nargs="?", default="all")
    build.add_argument("--build-arg", action="append", default=[])
    build.add_argument("--dry-run", action="store_true")
    for action in ("run", "smoke"):
        p = sub.add_parser(action)
        p.add_argument("model", choices=ENVS)
        p.add_argument("stage")
        device = p.add_mutually_exclusive_group()
        device.add_argument("--gpu", help="Assigned host GPU index/UUID, or comma-separated list; never implicit")
        device.add_argument("--cpu", action="store_true")
        p.add_argument("--data", help="Existing SCOUT directory, mounted read-only at /data/scout")
        p.add_argument("--weights", help="Existing weight directory, mounted read-only at /weights")
        p.add_argument("--output", help="Writable results directory; default outputs/<model>/<stage>")
        p.add_argument("--cpus", type=float, default=2.0)
        p.add_argument("--memory", default="8g")
        p.add_argument("--network", choices=["none", "bridge"], default="none")
        p.add_argument("--interactive", action="store_true")
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "list":
        for model, stages in ENVS.items():
            for stage, spec in stages.items():
                print(model, stage, image_name(model, stage), "CUDA" if spec["cuda"] else "CPU-only")
        return 0
    if override and args.action != "run":
        parser.error("Command override after -- is supported only by run")
    timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%dT%H%M%S%f")
    if args.action == "build":
        models = list(ENVS) if args.model == "all" else [args.model]
        for model in models:
            checked_source(model)
            if model == "yachiyo" and not args.dry_run:
                subprocess.run(["bash", str(ROOT / "docker/yachiyo/fetch-vendor.sh")], check=True)
            stages = list(ENVS[model]) if args.stage == "all" else [args.stage]
            for stage in stages:
                if stage not in ENVS[model]:
                    parser.error("Unknown stage: " + model + "/" + stage)
                spec = ENVS[model][stage]
                cmd = ["docker", "build", "--progress=plain", "--label", "project=scout-mtmc",
                       "--label", "org.opencontainers.image.revision=" + SOURCES[model]["commit"],
                       "--tag", image_name(model, stage), "--file", "docker/" + model + "/Dockerfile"]
                if spec["target"]:
                    cmd += ["--target", spec["target"]]
                for item in args.build_arg:
                    cmd += ["--build-arg", item]
                cmd += ["."]
                rc = execute(cmd, args.dry_run, ROOT / "logs" / (model + "-" + stage + "-build-" + timestamp + ".log"))
                if rc:
                    return rc
        return 0
    if args.stage not in ENVS[args.model]:
        parser.error("Unknown stage: " + args.stage)
    spec = ENVS[args.model][args.stage]
    if args.cpus <= 0:
        parser.error("--cpus must be positive")
    if args.gpu:
        if not spec["cuda"]:
            parser.error("This stage is CPU-only; use --cpu")
        if not re.fullmatch(r"(?:\d+|GPU-[0-9a-fA-F-]+)(?:,(?:\d+|GPU-[0-9a-fA-F-]+))*", args.gpu):
            parser.error("--gpu requires GPU index(es) or GPU UUID(s)")
    output = host_directory(args.output or ROOT / "outputs" / args.model / args.stage, allow_missing=True)
    mounts = [(output, "/outputs", False)]
    for value, destination in ((args.data, "/data/scout"), (args.weights, "/weights")):
        if value:
            path = host_directory(value)
            if path == output or path in output.parents or output in path.parents:
                parser.error("Input and output directories must be separate and must not contain one another")
            mounts.append((path, destination, True))
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--pull=never", "--label", "project=scout-mtmc",
           "--user", "{}:{}".format(os.getuid(), os.getgid()), "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--cpus", str(args.cpus), "--memory", args.memory,
           "--shm-size", "1g", "--network", args.network,
           "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/.cache",
           "--env", "SCOUT_DATA_DIR=/data/scout", "--env", "SCOUT_OUTPUT_DIR=/outputs",
           "--env", "SCOUT_WEIGHTS_DIR=/weights", "--env", "PYTHONDONTWRITEBYTECODE=1",
           "--env", "OMP_NUM_THREADS=2", "--env", "OPENBLAS_NUM_THREADS=2"]
    if args.gpu:
        cmd += ["--runtime=nvidia", "--gpus", '"device=' + args.gpu + '"',
                "--env", "NVIDIA_VISIBLE_DEVICES=" + args.gpu,
                "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
    else:
        cmd += ["--runtime=runc", "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES="]
    if args.interactive:
        cmd += ["-it"]
    for path, destination, readonly in mounts:
        cmd += ["--mount", "type=bind,src={},dst={}{}".format(path, destination, ",readonly" if readonly else "")]
    cmd += [image_name(args.model, args.stage)]
    cmd += override or ["python", "/opt/scout/smoke.py", "--device", "cuda" if args.gpu else "cpu"]
    log = None if args.action == "run" else ROOT / "logs" / (args.model + "-" + args.stage + "-" + ("gpu" if args.gpu else "cpu") + "-" + timestamp + ".log")
    return execute(cmd, args.dry_run, log)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
