#!/usr/bin/env python3
"""Restore pinned public upstreams without replacing existing work."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--check", action="store_true", help="Verify only; never fetch or change files")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = Path(__file__).resolve().parent.parent / "upstream-sources.lock.json"
    specs = json.loads(lock.read_text())["repositories"]
    missing = []
    for spec in specs:
        path = root / spec["path"]
        if root not in path.resolve().parents or path.is_symlink():
            raise SystemExit("Unsafe source path: " + str(path))
        if not re.fullmatch(r"[0-9a-f]{40}", spec["commit"]):
            raise SystemExit("Invalid pinned commit")
        if not spec["url"].startswith("https://github.com/"):
            raise SystemExit("Expected public GitHub HTTPS source")
        if not path.exists():
            missing.append((path, spec))
            continue
        if not (path / ".git").exists():
            raise SystemExit("Preserving existing non-repository path: " + str(path))
        if git(path, "rev-parse", "HEAD") != spec["commit"] or git(path, "status", "--porcelain"):
            raise SystemExit("Preserving modified or different checkout: " + str(path))
        print("Verified", spec["path"], spec["commit"])
    if args.check and missing:
        raise SystemExit("Missing sources: " + ", ".join(s["path"] for _, s in missing))
    for path, spec in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(path)], check=True)
        git(path, "remote", "add", "origin", spec["url"])
        git(path, "fetch", "--depth", "1", "origin", spec["commit"])
        git(path, "checkout", "--detach", spec["commit"])
        if git(path, "rev-parse", "HEAD") != spec["commit"]:
            raise SystemExit("Checkout verification failed: " + str(path))
        print("Restored", spec["path"], spec["commit"])


if __name__ == "__main__":
    main()
